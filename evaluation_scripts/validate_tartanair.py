import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse
from mpl_toolkits.mplot3d import Axes3D

from droid_slam.droid import Droid

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))
    images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    return data

def plot_traj(gtposes, vis=False, savefigname="/home/honsen/qwe1.png", title=''):
    import matplotlib.pyplot as plt
    fig = plt.figure()

    ax = fig.gca(projection='3d')

    gtposes = np.matrix(gtposes.astype(np.float64))
    ax.plot(gtposes[:, 2], gtposes[:, 1], gtposes[:, 0], linestyle='dashed', c='k')

    ax.set_xlabel('x (m)')
    ax.set_zlabel('z (m)')
    ax.set_ylabel('y (m)')
    ax.legend(['Ground Truth', 'Ours'])
    plt.title(title)

    # plt.axis('equal')

    if savefigname is not None:
        plt.savefig(savefigname)

    if 1:
        plt.show()

    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/home/honsen/tartan/test/tartanair-test-mono-release/mono")
    parser.add_argument("--weights", default="demo.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12) #12
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from droid_slam.data_readers.tartan import test_split
    from thirdparty.tartanair_tools.evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    if args.id >= 0:
        test_split = [ test_split[args.id] ]

    # test_split = [""/home/honsen/tartan/dataset/ocean/Hard/P009/image_left"",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME001",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME002",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME003",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME004",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME005",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME006",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME007",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH000",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH001",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH002",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH003",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH004",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH005",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH006",
    #               "/home/honsen/tartan/test/tartanair-test-mono-release/mono/MH007"]
    test_split = ["/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME004"]
    ate_list = []
    for scene in test_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()
        droid = Droid(args)

        scenedir = os.path.join(args.datapath, scene)

        gt_file = os.path.join(scenedir, "pose_left.txt")
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]]  # ned -> xyz
        # plot_traj(traj_ref)
        for (tstamp, image, intrinsics) in tqdm(image_stream(scenedir, stereo=args.stereo)):
            droid.track(tstamp, image, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(image_stream(scenedir))

        ### do evaluation ###
        evaluator = TartanAirEvaluator()

        traj_est1 = traj_est.tolist()
        with open("/home/honsen/honsen/tarT/droid_ME001.txt", "w") as file:
            for i in range(len(traj_est1)):
                traj_est1[i] = [str(x) for x in traj_est1[i]]
                file.write(" ".join(traj_est1[i]) + "\n")

        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
        
        print(results)
        ate_list.append(results["ate_score"])

    print("Results")
    print(ate_list)

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.show()


