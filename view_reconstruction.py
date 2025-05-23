import sys
sys.path.append("droid_slam")

import torch
import argparse
import os
import numpy as np
import cv2
import json

import droid_backends
import argparse
import open3d as o3d

from droid_slam.visualization import create_camera_actor
from lietorch import SE3


def save_reconstruction_data(folder: str, images: torch.Tensor, disps: torch.Tensor, poses: torch.Tensor, intrinsics: torch.Tensor, point_cloud: o3d.geometry.PointCloud):
    color_dir = os.path.join(folder, "color")
    depth_dir = os.path.join(folder, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir,  exist_ok=True)

    fx, fy, cx, cy = intrinsics.cpu().tolist()

    K_flat = [
        fx,   0.0, cx,
        0.0,  fy,  cy,
        0.0,  0.0, 1.0
    ]

    # Convert SE(3) poses to 4×4 world-to-camera matrices (same convention as demo)
    T_wc = SE3(poses).inv().matrix().cpu().numpy()   # (N,4,4)

    for i in range(len(images)):
        rgb = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8) # H×W×3, RGB
        cv2.imwrite(
            os.path.join(color_dir, f"frame_{i:05d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)    # JPEG wants BGR
        )

        record = {
            "cameraPoseARFrame": T_wc[i].reshape(-1).tolist(),  # 16 numbers row-major
            "intrinsics": K_flat    # 9 numbers row-major
        }
        with open(os.path.join(color_dir, f"frame_{i:05d}.json"), "w") as fp:
            json.dump(record, fp, indent=2)

        depth_m = disps[i].cpu().numpy()    # if this is disparity, convert first!
        depth_mm_u16 = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"frame_{i:05d}.png"), depth_mm_u16)

    o3d.io.write_point_cloud(
        os.path.join(folder, "reconstruction.ply"),
        point_cloud
    )

    print(f"Reconstruction written to “{folder}”")

def view_reconstruction(filename: str, filter_thresh = 0.005, filter_count=2, output_folder="results"):
    reconstruction_blob = torch.load(filename)
    images = reconstruction_blob["images"].cuda()[...,::2,::2]
    disps = reconstruction_blob["disps"].cuda()[...,::2,::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

    points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:,[2,1,0]].permute(0,2,3,1) / 255.0

    counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    # Save to folder
    save_reconstruction_data(output_folder, images, disps, poses, intrinsics[0], point_cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.add_geometry(point_cloud)

    # get pose matrices as a nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    ### add camera actor ###
    for i in range(len(poses)):
        cam_actor = create_camera_actor(False)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to saved reconstruction .pth")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--output", type=str, default="results", help="output folder path")
    args = parser.parse_args()

    view_reconstruction(args.filename, args.filter_threshold, args.filter_count, args.output)