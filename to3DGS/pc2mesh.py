import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import droid_backends
from lietorch import SE3
import os
from executeSlam import initialize_first_timestep
import trimesh


pose11 = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/poses.npy")
disps = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/disps.npy")
images = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/images.npy")
tstamps = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/tstamps.npy")
intrinsics1 = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/intrinsics.npy")


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

_, _, _, first_frame_w2c, cam = initialize_first_timestep(dataset, cur_t,3,"projective")

"""
    load the point_cloud with color
"""
params = dict(np.load("lgu.npz"))
params['means3D'] = torch.from_numpy(params['means3D']).cuda()
params['cam_unnorm_rots'] = torch.from_numpy(params['cam_unnorm_rots']).cuda()
params['cam_trans'] = torch.from_numpy(params['cam_trans']).cuda()
params['gt_w2c_all_frames'] = torch.from_numpy(params['gt_w2c_all_frames']).cuda()
params['rgb_colors'] = torch.from_numpy(params['rgb_colors']).cuda()
params['unnorm_rotations'] = torch.from_numpy(params['unnorm_rotations']).cuda()
params['logit_opacities'] = torch.from_numpy(params['logit_opacities']).cuda()
params['log_scales'] = torch.from_numpy(params['log_scales']).cuda()

volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i in range(cur_t):

    transformed_pts = transform_to_frame(params, i,
                                                 gaussians_grad=True,
                                                 camera_grad=False)

    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c,
                                                                     transformed_pts)

    im, radius, _, = Renderer(raster_settings=cam)(**rendervar)

    depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)

    depth = depth_sil[0, :, :].unsqueeze(0)
    image = torch.clamp(im, 0.0, 1.0)
    depth[dataset['depth'][i].unsqueeze(0) == 0] = 0
    depth_o3d = np.ascontiguousarray(depth.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
    depth_o3d = o3d.geometry.Image(depth_o3d)
    image = (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)

    """
        show the 3DGS rendered image
    """
    img1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imshow("qwe", img1)
    cv2.waitKey(100)

    color_o3d = np.ascontiguousarray(image)
    color_o3d = o3d.geometry.Image(color_o3d)

    w2c_o3d = params['gt_w2c_all_frames'][i].detach().cpu().numpy()  # convert from c2w to w2c

    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    W = depth.shape[-1]
    H = depth.shape[1]
    intrinsic2 = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=30,
        convert_rgb_to_intensity=False)
    # use gt pose for debugging
    # w2c_o3d = torch.linalg.inv(pose).cpu().numpy() @ dataset.w2c_first_pose
    volume.integrate(rgbd, intrinsic2, w2c_o3d)

mesh_out_file = os.path.join("/home/honsen", "mesh.ply")
o3d_mesh = volume.extract_triangle_mesh()
# o3d_mesh = clean_mesh(o3d_mesh)
o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
print('Meshing finished.')

"""
    load the mesh.ply
"""

# mesh = o3d.io.read_triangle_mesh("/home/honsen/mesh.ply")
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
