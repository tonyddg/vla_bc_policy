import torch

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list

from vla_bc_policy.model.bet_policy import BeTPolicy
from vla_bc_policy.train import get_trainer

def main(
    repo_id: str,
    stats_path: str,
    artifact_path: str,

    use_cluster_class_weights: bool = False,
    exp_name: str = "full_res_mlp",
    num_clusters: int = 8,

    is_shrink_vector: bool = False,
    is_shrink_decoder: bool = False,
    is_strong_regulation: bool = False,

    batch_size: int = 256,

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

        batch_size = batch_size,
        num_workers = 32,

        vec_obs_keys = ["pi0_actions_ref", "pi0_state", "qpos", "qvel"],
        vec_obs_compress_key = {
            "qvel": 1.5 
        },
    )
    pm = BeTPolicy(
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
            dropout_rate = 0.1 if not is_strong_regulation else 0.15,

            vector_out_feat = 1024 if not is_shrink_vector else 512,
            vector_encoder_type = "res_mlp",
            vector_encoder_kwargs = dict(
                hidden_dim = 1024 if not is_shrink_vector else 512,
                num_blocks = 2,
                expansion = 2,
                dropout_rate = 0.1 if not is_strong_regulation else 0.15,
                is_output_proj = False
            ),
        ),

        decoder_type = "res_mlp",
        decoder_kwargs = dict(
            hidden_dim = 2048 if not is_shrink_decoder else 1024,
            num_blocks = 2 if not is_shrink_decoder else 1,
            expansion = 2,
            dropout_rate = 0.1 if not is_strong_regulation else 0.2,
            # 接输出头
            is_output_proj = False,
            num_out_feats = 2048 if not is_shrink_decoder else 1024,
        ),

        action_head_type = "multi",
        action_head_kwargs = dict(head_decoder_config = dict(
            joint = (
                (0, 7),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.05,
                    is_output_proj = True
                )
            ),
            torso = (
                (7, 8),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.05,
                    is_output_proj = True
                )
            ),
            base = (
                (8, 10),
                dict(
                    mlp_layers = [256],
                    dropout_rate = 0.05,
                    is_output_proj = True
                )
            )
        )),

        num_clusters = num_clusters,
        artifact_path = artifact_path,
        cluster_cls_pred_kwargs = dict(
            mlp_layers = [256],
            dropout_rate = 0.05,
            is_output_proj = True
        ),
        use_cluster_class_weights = use_cluster_class_weights,
        cluster_cls_loss_weight = 0.25,
        is_soft_cls_target = True,
        soft_temperature = 0.09, # 基于 K8 统计结果

        stats_path = stats_path,
        data_module = dm,

        distance_loss_type = "smooth_l1",

        key_metrics = "MAE",

        lr = 1e-4,
        lr_schedule_type = "warmup_cos",
        lr_schedule_kwargs = dict(
            warmup_steps_ratio = 0.05,
            min_lr_ratio = 0.01
        ),
        weight_decay = 1e-4 if not is_strong_regulation else 5e-4
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
