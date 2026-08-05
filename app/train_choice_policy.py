import torch

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list

from vla_bc_policy.model.choice_policy import ChoicePolicy
from vla_bc_policy.train import get_trainer

def main(
    repo_id: str,
    stats_path: str,
    exp_name: str = "full_res_mlp",
    # 启动低精度但更快的训练方式
    is_fast_train: bool = False
):
    if is_fast_train:
        torch.set_float32_matmul_precision('medium')
    else:
        torch.set_float32_matmul_precision('high')

    dm = Pi0LeRobotDataModule(
        repo_id = repo_id + "_train",
        val_repo_id = repo_id + "_val",

        camera_info_list = camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False,

        batch_size = 512,
        num_workers = 32,

        vec_obs_keys = ["pi0_actions_ref", "pi0_state", "qpos", "qvel"],
        vec_obs_compress_key = {
            "qvel": 1.5 
        },
    )
    pm = ChoicePolicy(
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
            dropout_rate = 0.1,

            vector_out_feat = 1024,
            vector_encoder_type = "res_mlp",
            vector_encoder_kwargs = dict(
                hidden_dim = 1024,
                num_blocks = 2,
                expansion = 2,
                dropout_rate = 0.1,
                is_output_proj = False
            ),
        ),

        decoder_type = "res_mlp",
        decoder_kwargs = dict(
            hidden_dim = 2048,
            num_blocks = 2,
            expansion = 2,
            dropout_rate = 0.1,
            # 接输出头
            is_output_proj = False,
            num_out_feats = 2048,
        ),

        action_head_type = "multi",
        action_head_kwargs = dict(head_decoder_config = dict(
            joint = (
                (0, 7),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.0,
                    is_output_proj = True
                )
            ),
            torso = (
                (7, 8),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.0,
                    is_output_proj = True
                )
            ),
            base = (
                (8, 10),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.0,
                    is_output_proj = True
                )
            )
        )),

        num_proposals = 4,
        score_pred_kwargs = dict(
            mlp_layers = [256],
            dropout_rate = 0.0,
            is_output_proj = True
        ),

        stats_path = stats_path,
        data_module = dm,

        distance_loss_type = "smooth_l1",
        score_loss_type = "mse",

        key_metrics = "MAE",

        lr = 3e-4,
        lr_schedule_type = "warmup_cos",
        lr_schedule_kwargs = dict(
            warmup_steps_ratio = 0.05,
            min_lr_ratio = 0.01
        ),
        weight_decay = 1e-4
    )
    
    trainer = get_trainer(
        exp_name, precision = "32" if not is_fast_train else "bf16-mixed", 
        max_epochs = 80, patience = None,
        gradient_clip_val = 2.0
    )
    trainer.fit(pm, datamodule = dm)

if __name__ == "__main__":

    import tyro
    tyro.cli(main)
