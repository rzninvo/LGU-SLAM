import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [1, -1, 1.5],
    [1, 1, 1.5],
    [-1, 1, 1.5],
    [-0.5, 1, 1.5],
    [0.5, 1, 1.5],
    [0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):#0.05
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization( device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    # droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.005

    vis = o3d.visualization.Visualizer()

    # cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    pose11 = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/poses.npy")
    disps = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/disps.npy")
    images = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/images.npy")
    tstamps = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/tstamps.npy")
    intrinsics = np.load("/home/honsen/honsen/imt_3dgsSlam/resconstruction1/intrinsics.npy")
    # disps = 1 / (disps + 0.000007)
    # cv2.imshow("qwe",disps[0])
    # cv2.waitKey(1000)
    # cv2.imshow("qwe", disps[1])
    # cv2.waitKey(1000)
    # cv2.imshow("qwe", disps[2])
    # cv2.waitKey(1000)
    # cv2.imshow("qwe", disps[3])
    # cv2.waitKey(1000)
    disps = (torch.from_numpy(disps)).cuda()
    pose11 = (torch.from_numpy(pose11)).cuda()
    images = (torch.from_numpy(images)).cuda()
    intrinsics = (torch.from_numpy(intrinsics)).cuda()
    with torch.no_grad():

        vis.create_window(height=540, width=960)

        # convert poses to 4x4 matrix
        cur_t = disps.shape[0]
        dirty_index = torch.arange(0, cur_t).long().to("cuda")

        poses = torch.index_select(pose11, 0, dirty_index)
        disps = torch.index_select(disps, 0, dirty_index)
        Ps = SE3(pose11).inv().matrix().cpu().numpy()
        # disps = 1/(disps+0.000007)
        images = torch.index_select(images, 0, dirty_index)
        images = images.cpu()[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0]*8).cpu()

        thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

        count = droid_backends.depth_filter(
            pose11, disps, intrinsics[0]*8, dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))

        for i in range(cur_t):
            pose = Ps[i]
            ix = dirty_index[i].item()

            if ix in droid_visualization.cameras:
                vis.remove_geometry(droid_visualization.cameras[ix])
                del droid_visualization.cameras[ix]

            if ix in droid_visualization.points:
                vis.remove_geometry(droid_visualization.points[ix])
                del droid_visualization.points[ix]

            ### add camera actor ###
            cam_actor = create_camera_actor(True)
            cam_actor.transform(pose)
            vis.add_geometry(cam_actor)
            droid_visualization.cameras[ix] = cam_actor

            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()#[mask]
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

            ## add point actor ###
            point_actor = create_point_actor(pts, clr)
            vis.add_geometry(point_actor)

            droid_visualization.points[ix] = point_actor

        # hack to allow interacting with vizualization during inference
        # if len(droid_visualization.cameras) >= droid_visualization.warmup:
        #     cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        droid_visualization.ix += 1
        vis.poll_events()
        vis.update_renderer()


    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()

if __name__ =="__main__":
    droid_visualization(device="cuda:0")