import argparse
import math
import os
import shutil
import time
from importlib.machinery import SourceFileLoader
import cv2
import random
import torch
import droid_backends
from lietorch import SE3
from collections import OrderedDict
from loss.loss import get_loss
from utils.keyframe_selection import keyframe_selection_overlap
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from datasets.gradslam_datasets import (
    load_dataset_config,
    ReplicaDataset,
)
rng = np.random.default_rng(12345)
def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud

    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian ** 2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        pcd_mask = curr_data["pcd_mask"]

        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)& pcd_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None):
    # Get RGB-D Data & Camera Parameters

    # color, depth, intrinsics, pose = dataset[0]
    #
    # # Process RGB-D Data
    # color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    # depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    color = dataset["images"][0]
    depth = dataset["depth"][0]
    pose = dataset["pose"][0]+1e-7
    intrinsics = dataset["intrinsic"]
    pcd_mask = dataset["pcd_mask"][0]
    h,w = pcd_mask.shape
    pcd_mask = pcd_mask.view(1,h,w)
    color = color/ 255
    depth = depth.unsqueeze(dim=0)

    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]


    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    depth1 = depth[pcd_mask]
    mask = (depth > 0)
    mask = mask&pcd_mask# Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth1) / scene_radius_depth_ratio


    return params, variables, intrinsics, w2c, cam



def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx - 1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx - 2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx - 1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx - 2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx - 1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx - 1].detach()

    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

from torch import Tensor
def quat_to_rotmat(quat: Tensor) -> Tensor:
    """
    将四元数转换为旋转矩阵
    :param quat: 形状为(N, 4)的四元数张量
    :return: 形状为(N, 3, 3)的旋转矩阵张量
    """
    assert quat.shape[-1] == 4

    # 复制四元数以匹配维度
    quat = quat.unsqueeze(-2) if quat.dim() == 1 else quat

    # 提取四元数分量
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 计算旋转矩阵的组成部分
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    zx = z * x
    xw = x * w
    yw = y * w
    zw = z * w

    # 创建旋转矩阵
    r1 = torch.cat((1 - 2 * (yy + zz),2 * (xy - zw),2 * (zx + yw)))
    r1 = torch.unsqueeze(r1,dim=0)
    r2 = torch.cat((2 * (xy + zw), 1 - 2 * (zz + xx), 2 * (yz - xw)))
    r2 = torch.unsqueeze(r2, dim=0)
    r3 = torch.cat((2 * (zx - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)))
    r3 = torch.unsqueeze(r3, dim=0)
    rot_mat = torch.cat((r1,r2,r3),dim=0)

    # 移除额外的维度
    rot_mat = rot_mat.squeeze(0) if quat.dim() == 1 else rot_mat
    return rot_mat

def align_kf_traj(video_traj,video_timestamps,stream,return_full_est_traj=False):
    # offline_video = dict(np.load(npz_path))
    traj_ref = []
    traj_est = []

    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        val = stream.poses[timestamp].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        poset = stream.poses[timestamp].cpu().numpy()
        traj_est.append(video_traj[i])
        traj_ref.append(poset)
        timestamps.append(video_timestamps[i])

    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    if return_full_est_traj:
        from evo.core import lie_algebra as lie
        traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref

def traj_eval_and_plot(traj_est, traj_ref):
    import os
    from evo.core import metrics

    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    return ape_statistics


def imt_3dgsSlam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device(config["primary_device"])

    """
    load your reconstruction file
    """
    pose11 = np.load("xxxxxxxxxxxx/poses.npy")
    disps = np.load("xxxxxxxxxxxxx/disps.npy")
    images = np.load("xxxxxxxxxxxx/images.npy")
    tstamps = np.load("xxxxxxxxxxxx/tstamps.npy")
    intrinsics1 = np.load("xxxxxxxxxxxxx/intrinsics.npy")

    for i in range(images.shape[0]):
        img = images[i]
        img = np.transpose(img,(1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images[i] = np.transpose(img,(2,0,1))
    # cv2.imshow("qwe",img)
    # cv2.waitKey(1000)
    disps = (torch.from_numpy(disps)).cuda()
    images = (torch.from_numpy(images)).cuda()
    pose11 = (torch.from_numpy(pose11)).cuda()

    Ps = SE3(pose11).inv().matrix()
    poseRT = Ps.cuda()


    intrinsic = intrinsics1[0] * 8
    intrinsic = torch.from_numpy(intrinsic)
    intrinsic = intrinsic.to(torch.float32)
    intrinsic = intrinsic.cuda()
    cur_t = disps.shape[0]
    dirty_index = torch.arange(0, cur_t).long().to("cuda")

    poseRT = torch.index_select(poseRT.detach(), dim=0, index=dirty_index)
    disps = torch.index_select(disps.detach(), dim=0, index=dirty_index)

    thresh = 0.005 * torch.ones_like(disps.mean(dim=[1, 2]))

    count = droid_backends.depth_filter(
        pose11, disps, intrinsic, dirty_index, thresh)


    masks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))

    dataset = {}

    K = torch.eye(3)

    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]

    intrinsics = torch.eye(4).to(K)
    intrinsics[:3, :3] = K

    dataset["images"] = images
    dataset["pose"] = poseRT
    dataset["depth"] = 1.0 / (disps + 1e-7)# depths
    dataset["intrinsic"] = intrinsics.cuda()
    dataset["pcd_mask"] = masks

    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, cur_t,config['scene_radius_depth_ratio'],config['mean_sq_dist_method'])
    #

    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []

    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0



    for time_idx in tqdm(range(cur_t)):
        # Load RGBD frames incrementally instead of all frames
        color = dataset["images"][time_idx]
        depth = dataset["depth"][time_idx]
        gt_pose = dataset["pose"][time_idx]
        pcd_mask = dataset["pcd_mask"][time_idx].unsqueeze(dim=0)

        color = color / 255
        depth = depth.unsqueeze(dim=0)

        gt_w2c = torch.linalg.inv(gt_pose)

        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,'pcd_mask':pcd_mask}


        # Optimization Iterations

        rel_w2c = curr_gt_w2c[-1]
        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = rel_w2c[:3, 3].detach()

        with torch.no_grad():
            params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
            params['cam_trans'][..., time_idx] = rel_w2c_tran





        num_iters_mapping = 60#config['mapping']['num_iters']

        if time_idx > 0 and 1 :  # config['tracking']['use_gt_poses']
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran


        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:

                densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'])
                post_num_pts = params['means3D'].shape[0]

            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size'] - 2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                print(selected_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list) - 1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                    iter_pcd_mask = pcd_mask
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    iter_pcd_mask = keyframe_list[selected_rand_keyframe_idx]['pcd_mask']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx + 1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx,
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c,'iter_pcd_mask':iter_pcd_mask}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx,
                                                   config['mapping']['loss_weights'],
                                                   config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                   config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                   mapping=True,do_ba=True)
                print("mapping loss:" + str(loss))

                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter,
                                                            config['mapping']['pruning_dict'])

                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])

                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress

                    progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

        if time_idx == 0 or (time_idx + 1) % config['report_global_progress_every'] == 0:
            try:
                # Report Mapping Progress
                progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")

                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

    # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or \
            (time_idx == cur_t - 2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (
        not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth,'pcd_mask' : pcd_mask}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

    # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg * 1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg * 1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    """
    visualize pointcould with color in the final params
    """
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # colors = params['rgb_colors'].cpu()
    # pcls = params['means3D'].cpu()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcls.detach().numpy())
    # pcd.colors = o3d.utility.Vector3dVector(colors.detach().numpy())
    # vis.add_geometry(pcd)
    # #
    # vis.poll_events()
    # vis.update_renderer()
    # vis.get_render_option().point_size = 2.0
    # vis.get_view_control().set_lookat([0, 0, 0])
    # vis.get_view_control().set_zoom(0.5)
    # vis.run()

    # Save Parameters
    save_params(params, "/home/honsen")



def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.

        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, help="Path to experiment file")
    args = parser.parse_args(['--experiment', '/home/honsen/honsen/imt_3dgsSlam/configs/replica/splatam.py'])
    # args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    imt_3dgsSlam(experiment.config)
