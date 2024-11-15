import pyzed.sl as sl
import numpy as np
from autolab_core import CameraIntrinsics, PointCloud, RgbCloud
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
raft_stereo_path = os.path.join(dir_path,'RAFT-Stereo')
sys.path.append(raft_stereo_path)
raft_stereo_core_path = os.path.join(raft_stereo_path,'core')
sys.path.append(raft_stereo_core_path)
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2

DEVICE = 'cuda'
# Default is 1920x1080 resolution at 15 fps
class Zed:
    def __init__(self):
        init = sl.InitParameters()
        init.camera_image_flip = sl.FLIP_MODE.OFF
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.camera_resolution = sl.RESOLUTION.HD1080
        self.width_ = 1920
        self.height_ = 1080
        init.sdk_verbose = 1
        init.camera_fps = 15
        self.cam = sl.Camera()
        init.camera_disable_self_calib = True
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
            print("Camera Open : " + repr(status) + ". Exit program.")
            exit()
        else:
            print("Opened camera")
        left_cx = self.get_K(cam="left")[0, 2]
        right_cx = self.get_K(cam="right")[0, 2]
        self.cx_diff = right_cx - left_cx  # /1920
        self.f_ = self.get_K(cam="left")[0,0]
        self.cx_ = left_cx
        self.cy_ = self.get_K(cam="left")[1,2]
        self.Tx_ = self.get_stereo_transform()[0,3]

        # RAFT Parameters
        self.parser_ = argparse.ArgumentParser()
        self.parser_.add_argument(
            "--restore_ckpt",
            default=os.path.join(raft_stereo_path,'models/raftstereo-middlebury.pth'),
            help="restore checkpoint",
        )
        self.parser_.add_argument(
            "--save_numpy",
            action="store_true",
            help="save output as numpy arrays"
        )
        self.parser_.add_argument(
            "-l",
            "--left_imgs",
            help="path to all first (left) frames",
            default='/this/is/a/fake/path',
        )
        self.parser_.add_argument(
            "-r",
            "--right_imgs",
            help="path to all second (right) frames",
            default='/this/is/a/fake/path',
        )
        self.parser_.add_argument(
            "--mask_imgs",
            help="path to all mask frames",
            default='/this/is/a/fake/path',
        )
        self.parser_.add_argument(
            "--output_directory", help="directory to save output", default="demo_output"
        )
        self.parser_.add_argument(
            "--shared_backbone",
            action="store_true",
            default=False,
            help="use a single backbone for the context and feature encoders",
        )
        self.parser_.add_argument(
            "--mixed_precision",
            action="store_true",
            default=False,
            help="use mixed precision",
        )
        self.parser_.add_argument(
            "--slow_fast_gru",
            action="store_true",
            default=False,
            help="iterate the low-res GRUs more frequently",
        )
        self.parser_.add_argument(
            "--corr_implementation",
            default="reg",
            choices=["reg", "alt", "reg_cuda", "alt_cuda"],
            help="correlation volume implementation",
        )
        self.parser_.add_argument(
            "--context_norm",
            default="batch",
            choices=["group", "batch", "instance", "none"],
            help="normalization of context encoder",
        )
        # Architecture choices
        self.parser_.add_argument(
            "--valid_iters",
            type=int,
            default=64,
            help="number of flow-field updates during forward pass",
        )
        self.parser_.add_argument(
            "--hidden_dims",
            nargs="+",
            type=int,
            default=[128] * 3,
            help="hidden state and context dimensions",
        )
        self.parser_.add_argument(
            "--corr_levels",
            type=int,
            default=4,
            help="number of levels in the correlation pyramid",
        )
        self.parser_.add_argument(
            "--corr_radius",
            type=int,
            default=4,
            help="width of the correlation pyramid",
        )
        self.parser_.add_argument(
            "--n_downsample",
            type=int,
            default=2,
            help="resolution of the disparity field (1/2^K)",
        )
        self.parser_.add_argument(
            "--n_gru_layers",
            type=int,
            default=3,
            help="number of hidden GRU levels"
        )

        self.args_ = self.parser_.parse_args()
        self.model_ = torch.nn.DataParallel(RAFTStereo(self.args_), device_ids=[0])
        self.model_.load_state_dict(torch.load(self.args_.restore_ckpt))
        self.image_data_ = None
        self.model_ = self.model_.module
        self.model_.to(DEVICE)
        self.model_.eval()
        self.padder_ = InputPadder(torch.Size([1,3,self.height_,self.width_]))
        
    def get_K(self, cam="left"):
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ]
        )
        return K

    def get_intr(self, cam="left"):
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        return CameraIntrinsics(
            frame="zed",
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            width=1280,
            height=720,
        )

    def get_stereo_transform(self):
        transform = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        )
        transform[:3, 3] /= 1000  # convert to meters
        return transform
    
    def load_image(self,img):
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
    
    def get_frame(self,depth=True,cam='left'):
        with torch.no_grad():
            res = sl.Resolution()
            res.width = self.width_
            res.height = self.height_
            if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
                left_rgb = sl.Mat()
                right_rgb = sl.Mat()
                self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT, sl.MEM.CPU, res)
                self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT, sl.MEM.CPU, res)
                left_rgb_np = left_rgb.get_data()[..., :3]
                right_rgb_np = right_rgb.get_data()[...,:3]
                left_bgr_np = np.flip(left_rgb_np,axis=2).copy()
                right_bgr_np = np.flip(right_rgb_np,axis=2).copy()
                image1 = self.load_image(left_bgr_np)
                image2 = self.load_image(right_bgr_np)

                image1, image2 = self.padder_.pad(image1, image2)

                _, flow_up = self.model_(image1, image2, iters=self.args_.valid_iters, test_mode=True)
                flow_up = self.padder_.unpad(flow_up).squeeze()
                depth_image = (self.f_ * self.Tx_) / abs(flow_up + self.cx_diff)
                depth_image = depth_image.detach().cpu().numpy()
                plt.imshow(depth_image,cmap='jet')
                plt.show()
                return left_rgb_np,right_rgb_np,depth_image
            else:
                raise RuntimeError("Could not grab frame")
zed = Zed()
left,right,depth = zed.get_frame()
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
cv2.imwrite('output/left_img.png',left)
cv2.imwrite('output/right_img.png',right)
np.save('output/depth_img.npy',depth)