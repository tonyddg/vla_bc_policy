import torch

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list

from vla_bc_policy.model.lit_model import PolicyModule
from vla_bc_policy.train import get_trainer

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    dm = Pi0LeRobotDataModule(
        repo_id = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle_train",
        val_repo_id = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle_val",

        camera_info_list = camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False,

        batch_size = 1024,
        num_workers = 32,

        vec_obs_keys = ["pi0_actions_ref", "pi0_state", "qpos", "qvel"],
        vec_obs_compress_key = {
            "qvel": 1.5 
        },
    )
    pm = PolicyModule(
        "post_mix",
        extractor_kwargs = dict(
            image_backbone_cfg = {
                "head": (
                    "resnet", {}, 512
                ),
                "gripper": (
                    "resnet", {}, 512
                ),
            },
            dropout_rate = 0.2,

            vector_out_feat = 1024,
            vector_encoder_type = "res_mlp",
            vector_encoder_kwargs = dict(
                hidden_dim = 1024,
                num_blocks = 2,
                expansion = 2,
                dropout_rate = 0.2,
                is_output_proj = False
            ),
        ),

        decoder_type = "res_mlp",
        decoder_kwargs = dict(
            hidden_dim = 2048,
            num_blocks = 2,
            expansion = 2,
            dropout_rate = 0.15,
            # 接输出头
            is_output_proj = False,
            num_out_feats = 2048,
        ),

        use_multi_head_decoder = True,
        multi_head_decoder_config = dict(head_decoder_config = dict(
            joint = (
                (0, 7),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.1
                )
            ),
            torso = (
                (7, 8),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.1
                )
            ),
            base = (
                (8, 10),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.1
                )
            )
        )),

        stats_path = "assets/train_stats_compress_qvel.json",
        data_module = dm,
        loss_type = "smooth_l1",

        lr = 3e-4,
        lr_schedule_type = "warmup_cos",
        lr_schedule_kwargs = dict(
            warmup_steps_ratio = 0.05,
            min_lr_ratio = 0.01
        ),
        weight_decay = 1e-4
    )
    
    trainer = get_trainer(
        "full_res_mlp", precision = "32", 
        max_epochs = 150, patience = None,
        gradient_clip_val = 2.0
    )
    trainer.fit(pm, datamodule = dm)
