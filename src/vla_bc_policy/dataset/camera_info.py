from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from dataclasses import dataclass, asdict

from vla_bc_policy.dataset.utility import depth_to_pointcloud

@dataclass
class CameraInfo:
    camera_name: str
    camera_intrinsic_mat: Optional[Sequence] = None
    
    is_include_rgb: bool = False
    is_depth2pcl: bool = True
    sample_keys: Optional[list[str]] = None

    z_near: float = 0.01
    z_far: float = np.inf

    def __post_init__(self):
        if self.sample_keys is None:
            self.sample_keys = [
                self.camera_name + "_depth",
                self.camera_name + "_camera_t",
                self.camera_name + "_rgb",
            ]
        if self.camera_intrinsic_mat is not None:
            self.camera_intrinsic_mat_torch = torch.tensor(
                self.camera_intrinsic_mat, dtype = torch.float32
            )
        else:
            self.camera_intrinsic_mat_torch = torch.diag(torch.ones(3))

    def get_in_channels(self):
        in_channels = 0
        if self.is_depth2pcl:
            in_channels = 3
        else:
            in_channels = 1
        
        if self.is_include_rgb:
            in_channels += 3
        
        return in_channels

    def sample2img(self, sample: Mapping[str, torch.Tensor]):
        
        assert self.sample_keys is not None
        
        # 处理深度图
        depth_part = sample[self.sample_keys[0]]
        depth_part = depth_part.type(torch.float32) / 1000.0
        if self.is_depth2pcl:
            # 将深度转为点云
            depth_part = depth_to_pointcloud(
                depth = depth_part, 
                K = self.camera_intrinsic_mat_torch,
                T_wc = sample[self.sample_keys[1]], 

                znear = self.z_near,
                zfar = self.z_far
            )
        else:
            # 纯深度图但使用 tanh 依据 z_far 压缩到 [0, 1] (类 mshab 处理方法)
            depth_part = 1 - torch.tanh(depth_part / self.z_far)
        # 从 h5 中读取, 还需要将 H, W, C 转为 C, H, W
        depth_part = torch.permute(depth_part, (2, 0, 1))
        # 拼接 rgb (Lerobot 读取已转为 C, H, W, float)
        if self.is_include_rgb:
            rgb_part = sample[self.sample_keys[2]]
            res = torch.cat([depth_part, rgb_part], dim = 0)
        else:
            res = depth_part
        return res
    
def camera_info_list_to_dict_list(camera_info_list: list[CameraInfo]):
    
    dict_list = []
    for camera_info in camera_info_list:
        dict_list.append(asdict(camera_info))
    return dict_list

###

FetchStandardCameraInfos = [
    CameraInfo(
        "head", camera_intrinsic_mat = [
            [41.0939,  0.0000, 64.0000],
            [ 0.0000, 41.0939, 64.0000],
            [ 0.0000,  0.0000,  1.0000]
        ], z_near = 0.01, z_far = 1.0, is_depth2pcl = False,
    ),
    CameraInfo(
        "gripper", camera_intrinsic_mat = [
            [41.0939,  0.0000, 64.0000],
            [ 0.0000, 41.0939, 64.0000],
            [ 0.0000,  0.0000,  1.0000]
        ], z_near = 0.01, z_far = 1.0, is_depth2pcl = False
    )
]

###