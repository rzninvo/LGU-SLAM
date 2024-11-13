from collections import OrderedDict

import cv2
from natsort import natsorted
import glob
import numpy as np
import os.path as osp
from lietorch import SE3
import torch
from scipy.spatial.transform import Rotation as R
from droid_net import DroidNet

poses = np.loadtxt(osp.join("/home/honsen/honsen", 'pose_left.txt'), delimiter=' ')
poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
pose = poses[0]
q = pose[2:-1]

rot = R.from_quat(q).as_matrix()
RT = np.eye(4)
RT[0:3,0:3]=rot
RT[0,3:4] = pose[0]
RT[1,3:4] = pose[1]
RT[2,3:4] = pose[2]

model = DroidNet()
state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load("/home/honsen/honsen/DROID-SLAM-main/droid.pth").items()])

state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

model.load_state_dict(state_dict)
model.cuda()
model.eval()

intrinsic = torch.as_tensor([600.0, 600.0, 599.5, 339.5])
intrinsic = intrinsic.to(torch.float32)
h1 = int(680 * np.sqrt((384 * 512) / (680 * 1200)))
w1 = int(1200 * np.sqrt((384 * 512) / (680 * 1200)))
intrinsic[0::2] *= (w1 / 1200)
intrinsic[1::2] *= (h1 / 680)

intrinsic0 = torch.Tensor(intrinsic)/8.0
intrinsics0 = torch.stack((intrinsic0,intrinsic0,intrinsic0,intrinsic0))
intrinsics0 = torch.unsqueeze(intrinsics0,dim=0).cuda()

poseT = torch.zeros(1, 7, device="cuda", dtype=torch.float)
poseT[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

poses = torch.cat((poseT,poseT,poseT,poseT),dim=0)[None]
poses = SE3(poses)
color_paths = natsorted(glob.glob(f"/media/honsen/PS2000/Replica/room0/results/frame*.jpg"))
images = []
for i in range(4):
    image = cv2.resize(cv2.imread(color_paths[i]), (w1, h1))
    image = image[:h1 - h1 % 8, :w1 - w1 % 8]
    images.append(image)

images = np.array(images)

images = torch.from_numpy(images)
images = images.permute(0,3,1,2)
images = images[None].cuda()
images = images/255
images = images[:,:,[2,1,0]]
MEAN = torch.as_tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
STDV = torch.as_tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

images = images.sub_(MEAN).div_(STDV)

disps = torch.ones(4, h1//8, w1//8, device="cuda", dtype=torch.float)[None].cuda()

graph = OrderedDict()
for i in range(4):
    graph[i] = [j for j in range(4) if i!=j and abs(i-j) <= 2]

poses_est, disps_est, residuals = model(poses, images, disps, intrinsics0,
                    graph, num_steps=2, fixedp=2)
print()



