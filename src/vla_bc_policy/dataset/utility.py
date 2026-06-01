from typing import Literal, Optional
import numpy as np
import torch

ACTION_SAMPLE_KEY = "policy_action"
VECTION_OBS_KEY = "vector"

ROT_TYPE = Literal["axis_angle", "rot_6d", "quat", "rpy_euler"]
STATE_ROT_LENGTH_DICT = {
    "axis_angle": 3,
    "rot_6d": 6,
    "quat": 4,
    "rpy_euler": 3
}

def depth_to_pointcloud(
    depth: torch.Tensor,
    K: torch.Tensor,
    T_wc: torch.Tensor,
    znear: float = 0.0,
    zfar: float = np.inf,
) -> torch.Tensor:
    """
    将深度图转换为世界坐标系点云。

    Args:
        depth: (H, W, 1)，深度图，单位与相机位姿/内参一致
        K:     (3, 3)，相机内参矩阵
        T_wc:  (4, 4)，相机坐标系到世界坐标系的齐次变换矩阵
               即 X_w = T_wc @ X_c_h
        znear: 最小深度
        zfar: 最大深度

    Returns:
        points_world: (H, W, 3)，世界坐标系点云
                      无效深度对应点为 (0, 0, 0)
    """
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise ValueError(f"depth should have shape (H, W, 1), got {tuple(depth.shape)}")

    if K.shape != (3, 3):
        raise ValueError(f"K should have shape (3, 3), got {tuple(K.shape)}")

    if T_wc.shape != (4, 4):
        raise ValueError(f"T_wc should have shape (4, 4), got {tuple(T_wc.shape)}")

    H, W, _ = depth.shape
    device = depth.device
    dtype = depth.dtype

    K = K.to(device=device, dtype=dtype)
    T_wc = T_wc.to(device=device, dtype=dtype)

    # 提取深度: (H, W)
    z = depth[..., 0]

    # 无效深度掩码
    invalid_mask = (~torch.isfinite(z)) | (z <= znear) | (z >= zfar)
    valid_mask = ~invalid_mask

    # 像素坐标网格
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # v: (H, W), u: (H, W)

    # 内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 反投影到相机坐标系
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    z_cam = z

    # 无效点先置零，避免传播 NaN/Inf
    x_cam = torch.where(valid_mask, x_cam, torch.zeros_like(x_cam))
    y_cam = torch.where(valid_mask, y_cam, torch.zeros_like(y_cam))
    z_cam = torch.where(valid_mask, z_cam, torch.zeros_like(z_cam))

    ones = torch.ones_like(z_cam)

    # 转换为 sapien 标准
    # 原始相机坐标为 [x_cam, y_cam, z_cam]
    # sapien 标准这里使用 [z_cam, -x_cam, -y_cam]
    points_cam_h = torch.stack(
        [z_cam, -x_cam, -y_cam, ones],
        dim=-1,
    )  # (H, W, 4)

    # 展平后做矩阵乘法
    points_cam_h_flat = points_cam_h.reshape(H * W, 4)          # (N, 4)
    points_world_h = points_cam_h_flat @ T_wc.T                 # (N, 4)

    points_world = points_world_h[:, :3].reshape(H, W, 3)

    # 再次保证无效点严格为 (0, 0, 0)
    points_world = torch.where(
        valid_mask.unsqueeze(-1),
        points_world,
        torch.zeros_like(points_world),
    )

    return points_world

def get_random_pi0_lerobot_batch(
    output_dim: Optional[int],
    vec_obs_dim: int,
    img_obs_dim: dict[str, int],
    b: int,

    image_size: tuple[int, int] = (128, 128),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    根据 get_pi0_lerobot_dataspec 的输出结果，随机生成一个 mock batch
    """

    h, w = image_size

    obs = {}

    obs[VECTION_OBS_KEY] = torch.randn(
        b, vec_obs_dim,
        device = device, dtype = dtype,
    )

    for camera_name, in_channels in img_obs_dim.items():
        obs[camera_name] = torch.randn(
            b, in_channels, h, w,
            device = device, dtype = dtype,
        )

    if output_dim is not None:
        action = torch.randn(
            b, output_dim, device = device, dtype = dtype,
        )
    else:
        action = None

    return obs, action
