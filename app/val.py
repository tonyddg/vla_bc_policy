import torch

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list

from vla_bc_policy.model.lit_model import PolicyModule
from vla_bc_policy.train import get_trainer

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    dm = Pi0LeRobotDataModule(
        repo_id = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle",

        camera_info_list = camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False,

        batch_size = 1024,
        num_workers = 32,

        vec_obs_keys = ["pi0_actions_ref", "pi0_state", "qpos", "qvel"],
        vec_obs_compress_key = {
            "qvel": 1.5 
        },
    )
    pm = PolicyModule.load_from_checkpoint(
        checkpoint_path = "/mnt/dataset/xzmyuq/vla_bc_policy/full_res_mlp/version_1/checkpoints/best-epoch-epoch=137-acc-val_RMSE=0.130.ckpt",
        stats_path = "assets/train_stats_compress_qvel_old.json"
    )
    
    trainer = get_trainer(
        "full_res_mlp", precision = "32", 
        max_epochs = 150, patience = None,
        gradient_clip_val = 2.0
    )
    res = trainer.validate(pm, datamodule = dm)
    print(res)
