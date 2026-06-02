from vla_bc_policy.dataset.utility import VECTION_OBS_KEY
from vla_bc_policy.dataset.camera_info import CameraInfo, camera_info_list_to_dict_list
from vla_bc_policy.dataset.pi0_lerobot_dataset import Pi0LeRobotDataset
from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy.dataset.normalizer import JsonNormalizer

import torch

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    SAMPLE_IDX = 0 # 28387

    TestCameraInfos = [
        CameraInfo(
            "gripper", camera_intrinsic_mat = [
                [41.0939,  0.0000, 64.0000],
                [ 0.0000, 41.0939, 64.0000],
                [ 0.0000,  0.0000,  1.0000]
            ], z_near = 0.01, z_far = 100.0, is_include_rgb = True
        ),
    ]
    
    print("#### Test Pi0LeRobotDataset Start ####")
    ds = Pi0LeRobotDataset(
        "trajectories_tidy_house_all_bc_fix_la_state_rot_6d_action_axis_angle",
        TestCameraInfos
    )
    output_dim, vec_obs_dim, img_obs_dim = ds.get_sample_size()
    print(f"output_dim: {output_dim}, vec_obs_dim: {vec_obs_dim}, img_obs_dim: {img_obs_dim}")

    obs_0, action_0 = ds[0]
    obs_1, action_1 = ds[1]
    obs_2, action_2 = ds[2]

    print(f"t = 0: last action {obs_0['vector']} action: {action_0}")
    print(f"t = 1: last action {obs_1['vector']} action: {action_1}")
    print(f"t = 2: last action {obs_2['vector']} action: {action_2}")

    obs, action = ds[SAMPLE_IDX]

    print(f"action: {action}")
    print(f"vec: {obs[VECTION_OBS_KEY]}")
    img = torch.as_tensor(obs[TestCameraInfos[0].camera_name])
    img = torch.permute(img, (1, 2, 0))
    img_xyz = img[:, :, :3]
    img_rgb = img[:, :, 3:]

    fig, axes = plt.subplot_mosaic([[0, 1, "color_bar"], [2, 3, "color_bar"]], width_ratios = [1, 1, 0.2])
    fig.set_layout_engine("compressed")
    
    # vmin = img_xyz.min().item()
    # vmax = img_xyz.max().item()
    for i in range(3):
        axe_image = axes[i].imshow(
            img_xyz[:, :, i].detach().cpu().numpy(), cmap="viridis",
            # vmin = vmin, vmax = vmax
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.colorbar(axe_image, cax = axes["color_bar"], label="Pixel value") # type: ignore

    axes[3].imshow(img_rgb.detach().cpu().numpy())
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    fig.savefig("output/sample.png")
    plt.close(fig)
    del ds
    print("#### Test Pi0LeRobotDataset Done ####")

    ###

    print("#### Test Pi0LeRobotDataModule Start ####")

    dm = Pi0LeRobotDataModule(
        "trajectories_tidy_house_all_bc_fix_la_state_rot_6d_action_axis_angle",
        camera_info_list_to_dict_list(TestCameraInfos),
        debug_config = False
    )
    print(f"output_dim: {dm.output_dim}, vec_obs_dim: {dm.vec_obs_dim}, img_obs_dim: {dm.img_obs_dim}")
    dm.setup()
    train_dataloader = dm.train_dataloader()
    obs_batch, action_batch = next(iter(train_dataloader))
    
    print(f"action_batch shape: {action_batch.shape}")
    for key, val in obs_batch.items():
        print(f"{key} shape: {val.shape}")
    
    print("#### Test Pi0LeRobotDataModule Done ####")

    ###

    print("#### Test JsonNormalizer Start ####")

    normalizer = JsonNormalizer(
        "assets/train_stats_depth.json"
    )
    obs_norm_batch = normalizer.normalize_obs(obs_batch)
    action_norm_batch = normalizer.normalize_action(action_batch)

    print(f"before norm vector: {obs_batch[VECTION_OBS_KEY][0]}")
    print(f"after norm vector: {obs_norm_batch[VECTION_OBS_KEY][0]}")

    print("#### Test JsonNormalizer End ####")
