import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Type, Union
import warnings
import torch
from torch import nn
from torch import optim
import lightning.pytorch as pl

from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy.dataset.normalizer import JsonNormalizer, NormalizeMethod

from vla_bc_policy.model.extractor import ExtractorType, ExtractorDict
from vla_bc_policy.model.mlp_decoder import MlpDecoder
from vla_bc_policy.model.res_mlp_decoder import ResMlpDecoder
from vla_bc_policy.model.action_head import MultiActionHead, SingleActionHead
from vla_bc_policy.model.utility import regression_metrics, LRScheduleType, WarmupCosineLR, LossFnDict, LossFnType
from vla_bc_policy.model.kmeans_codebook import KMeansCodebook

class BeTPolicy(pl.LightningModule):
    def __init__(
        self,

        # 模型参数
        extractor_type: ExtractorType,
        extractor_kwargs: Dict[str, Any],
        decoder_type: Literal["mlp", "res_mlp"],
        decoder_kwargs: Dict[str, Any],
        action_head_type: Literal["multi", "single"],
        action_head_kwargs: Dict[str, Any],

        # 标准化信息参数
        stats_path: Union[str, Path],

        # 输出分组
        action_group: Dict[str, tuple[int, int]] = dict(
            joint = (0, 7),
            torso = (7, 8),
            base = (8, 10)
        ),

        # BeT Kmeans 参数
        num_clusters: int = 8,
        cluster_groups: Dict[str, tuple[int, int]] = {
            "joint": (0, 7),
            "base": (8, 10),
        },
        artifact_path: Optional[Union[str, Path]] = None,
        # 类型总是 mlp decoder, 默认直接将特征映射为类别
        cluster_cls_pred_kwargs: Dict[str, Any] = dict(
            is_output_proj = True
        ),
        # 是否使用聚类权重
        use_cluster_class_weights: bool = True,
        cluster_cls_loss_weight: float = 0.5,

        # 优化器参数
        lr: float = 2e-3,
        optim_type: Literal["AdamW", "Adam", "SGD"] = "AdamW",
        weight_decay: float = 5e-6,
        sgd_momentum: float = 0.99,

        distance_loss_type: LossFnType = "mse",
        key_metrics: str = "RMSE", # 核心指标, 根号均方误差

        # 学习率调度参数
        lr_schedule_type: LRScheduleType = "none",
        lr_schedule_kwargs: Dict[str, Any] = {},

        # 标准化信息参数
        image_method: NormalizeMethod = "none",
        vector_method: NormalizeMethod = "standard",
        action_method: NormalizeMethod = "minmax",

        # 数据集相关的输入信息
        data_module: Optional[Pi0LeRobotDataModule] = None,

    ) -> None:
        super().__init__()

        assert num_clusters > 1, "num_clusters must > 1"

        # 在 self.save_hyperparameters 前保存输入信息, 确保没有数据集模型也可以加载
        if data_module is not None:
            vec_obs_dim = data_module.vec_obs_dim
            img_obs_dim = data_module.img_obs_dim
            output_dim = data_module.output_dim

            # 在 self.save_hyperparameters 前注入输入相关参数, 确保没有数据集模型也可以加载
            extractor_kwargs = dict(extractor_kwargs)
            extractor_kwargs["vec_obs_dim"] = vec_obs_dim
            extractor_kwargs["img_obs_dim"] = img_obs_dim

            # 自动配置 action head 的输入输出维度
            action_head_kwargs = dict(action_head_kwargs)
            action_head_kwargs["num_out_feats"] = output_dim
            action_head_kwargs["num_in_feats"] = decoder_kwargs["num_out_feats"]

            # 自动配置 action head 的输入输出维度
            cluster_cls_pred_kwargs = dict(cluster_cls_pred_kwargs)
            cluster_cls_pred_kwargs["num_out_feats"] = num_clusters
            cluster_cls_pred_kwargs["num_in_feats"] = decoder_kwargs["num_out_feats"]

        # 将 __init__ 参数保存到字典 self.hparams 中
        self.save_hyperparameters(ignore = ["data_module", "artifact_path"])

        # 创建模型
        self.extractor = ExtractorDict[extractor_type](**extractor_kwargs)
        decoder_kwargs["num_in_feats"] = self.extractor.get_out_feat()
        if decoder_type == "mlp":
            self.decoder = MlpDecoder(**decoder_kwargs)
        elif decoder_type == "res_mlp":
            self.decoder = ResMlpDecoder(**decoder_kwargs)
        else: 
            raise RuntimeError(f"Unknown decoder type: {decoder_type}")

        # 创建动作头
        self.num_clusters = num_clusters
        self.action_group = action_group
        def make_head():
            if action_head_type == "multi":
                action_head = MultiActionHead(**action_head_kwargs)
            elif action_head_type == "single":
                action_head = SingleActionHead(**action_head_kwargs)
            else: 
                raise RuntimeError(f"Unknown action_head type: {action_head_type}")
            return action_head
        self.action_head_list = nn.ModuleList([
            make_head() for _ in range(
                self.num_clusters
            )
        ])

        # 创建动作分类头
        self.cluster_cls_pred = MlpDecoder(**cluster_cls_pred_kwargs)

        # 创建动作聚类模型
        cluster_feature_dim = 0
        for idx_s, idx_e in cluster_groups.values():
            cluster_feature_dim += idx_e - idx_s
        self.kmeans_codebook = KMeansCodebook(
            num_clusters, action_head_kwargs["num_out_feats"],
            cluster_feature_dim, artifact_path
        )

        # 优化器参数
        self.lr = lr
        self.optim_type: Literal["AdamW", "Adam", "SGD"] = optim_type
        self.weight_decay = weight_decay
        self.sgd_momentum = sgd_momentum

        # 学习率调度参数
        self.lr_schedule_type: LRScheduleType = lr_schedule_type
        self.lr_schedule_kwargs = lr_schedule_kwargs
        self.key_metrics = key_metrics

        # loss 函数 (reduction = "none", 输出不改变形状)
        self.distance_loss_fn = LossFnDict[distance_loss_type](reduction = "none")
        # self.cls_loss_fn = nn.CrossEntropyLoss() # 使用 nn.founctional 防止 class weight 没有正确读取时出错
        self.use_cluster_class_weights = use_cluster_class_weights
        self.cluster_cls_loss_weight = cluster_cls_loss_weight

        # 标准化器
        self.normalizer = JsonNormalizer(
            stats_path, 
            image_method = image_method,
            vector_method = vector_method,
            action_method = action_method
        )

        # 示例输入
        if data_module is not None:
            obs, action = data_module.get_random_batch()
            self.example_input_array = dict(obs = obs)

    def configure_optimizers(self): # type: ignore

        if self.optim_type == "AdamW":
            use_optimizer = optim.AdamW(
                self.parameters(), 
                lr = self.lr,
                weight_decay = self.weight_decay
            )
        elif self.optim_type == "Adam":
            use_optimizer = optim.Adam(
                self.parameters(), 
                lr = self.lr,
                weight_decay = self.weight_decay
            )
        else:
            use_optimizer = optim.SGD(
                self.parameters(), 
                lr = self.lr, 
                momentum = self.sgd_momentum,
                weight_decay = self.weight_decay
            )

        use_lr_schedule_kwargs = dict(self.lr_schedule_kwargs)
        use_lr_schedule_kwargs["optimizer"] = use_optimizer
        if self.lr_schedule_type == "warmup_cos":
            use_lr_schedule_kwargs["total_steps"] = self.trainer.estimated_stepping_batches
            return {
                "optimizer": use_optimizer,
                "lr_scheduler": {
                    "scheduler": WarmupCosineLR(**use_lr_schedule_kwargs),
                    "interval": "step",   # 每个 step 更新学习率
                    "frequency": 1,
                },
            }
        elif self.lr_schedule_type == "reduce":
            return {
                "optimizer": use_optimizer,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(**use_lr_schedule_kwargs),
                    "monitor": f"val_{self.key_metrics}", # 重点：必须指定
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.lr_schedule_type == "none":
            # 单个优化器
            return use_optimizer
        else:
            raise ValueError(
                f"Unknown lr_schedule_type: {self.lr_schedule_type}"
            )

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # print(f"obs type {type(obs)}")
        obs = self.normalizer.normalize_obs(obs)

        feat = self.extractor(obs)
        feat = self.decoder(feat)

        proposal_action_residuals = torch.stack([
            # (B, A)
            action_head(feat) for action_head in self.action_head_list
        ], 1) # (B, P, A)
        cluster_cls_logits = self.cluster_cls_pred(feat) # (B, P)

        # (B, P, A), (B, P)
        return proposal_action_residuals, cluster_cls_logits

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        proposal_action_residuals, cluster_cls_logits = self.forward(obs)

        best_idx = torch.argmax(cluster_cls_logits, 1)
        selected_action_residuals = proposal_action_residuals[
            # argmin 结果元素所在的索引对应了该结果所对应的 batch 维坐标
            torch.arange(best_idx.shape[0], device = proposal_action_residuals.device), 
            best_idx
        ]
        action_norm = self.kmeans_codebook.decode_action(
            best_idx,
            selected_action_residuals,
        )

        return self.normalizer.denormalize_action(
            action_norm
        )

    def compute_loss(self, obs, gt_action: torch.Tensor):
        (
            proposal_action_residuals, # (B, P, A)
            cluster_cls_logits         # (B, P)  
        ) = self.forward(obs)
        gt_action_norm = self.normalizer.normalize_action(gt_action) # (B, A)
        batch_size = gt_action.shape[0]
        batch_rows = torch.arange(batch_size, device = gt_action.device)

        with torch.no_grad():
            gt_cluster_idx = self.kmeans_codebook.assign_action_cluster(
                gt_action_norm
            )  # (B,)
            gt_center = self.kmeans_codebook.action_centers[
                gt_cluster_idx
            ] 
            # 除以标准差标准化为 N(0, 1)
            gt_residual = (
                gt_action_norm - gt_center
            ) / self.kmeans_codebook.residual_scale # (B, A)

        # 选择对应动作模式头的输出作为预测值
        selected_action_residual = proposal_action_residuals[
            batch_rows,
            gt_cluster_idx,
        ] # (B, A)
        action_loss = torch.mean(self.distance_loss_fn(selected_action_residual, gt_residual))

        class_weights = None
        if self.use_cluster_class_weights:
            class_weights = self.kmeans_codebook.cluster_class_weights
        cluster_cls_loss = nn.functional.cross_entropy(
            cluster_cls_logits,
            gt_cluster_idx,
            weight = class_weights,
        )
        # 用 log(K) 归一化，使不同 K 的初始 CE 更可比。
        normalized_cluster_loss = (
            cluster_cls_loss
            / math.log(self.num_clusters)
        )

        loss = (
            action_loss
            + self.cluster_cls_loss_weight
            * normalized_cluster_loss
        )

        with torch.no_grad():
            # 模型预测动作
            pred_cluster_idx = cluster_cls_logits.argmax(
                dim=1
            )
            selected_residual = proposal_action_residuals[
                batch_rows,
                pred_cluster_idx,
            ]
            pred_action_norm = self.kmeans_codebook.decode_action(
                pred_cluster_idx,
                selected_residual,
            )
            # 聚类精确率
            cluster_accuracy = (
                pred_cluster_idx == gt_cluster_idx
            ).float().mean()

        loss_info = {
            "action_loss": action_loss.detach(),
            "normalized_cluster_loss": normalized_cluster_loss.detach(),
            "cluster_accuracy": cluster_accuracy.detach()
        }

        return loss, pred_action_norm, loss_info
    
    def training_step(self, batch: tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):
        obs, action = batch

        target_action_norm = self.normalizer.normalize_action(action)
        train_loss, pred_action_norm, loss_info = self.compute_loss(obs, action)

        self.log(
            "train_loss", train_loss, 
            batch_size = action.size(0),
            prog_bar = True, logger = True
        )
        if loss_info is not None:
            for key, val in loss_info.items():
                self.log(
                    f"train_{key}", val,
                    batch_size = action.size(0),
                    prog_bar = False, logger = True
                )

        train_metrics = regression_metrics(pred_action_norm, target_action_norm, groups = self.action_group)
        for key, val in train_metrics.items():
            self.log(
                f"train_{key}", val,
                batch_size = action.size(0),
                prog_bar = False, logger = True
            )

        # 默认自动优化, 仅需返回带有梯度的 loss
        return train_loss

    def validation_step(self, batch: tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):
        obs, action = batch

        target_action_norm = self.normalizer.normalize_action(action)
        valid_loss, pred_action_norm, loss_info = self.compute_loss(obs, action)

        self.log(
            "valid_loss", valid_loss, 
            batch_size = action.size(0),
            prog_bar = True, logger = True
        )
        if loss_info is not None:
            for key, val in loss_info.items():
                self.log(
                    f"valid_{key}", val,
                    batch_size = action.size(0),
                    prog_bar = False, logger = True
                )

        valid_metrics = regression_metrics(pred_action_norm, target_action_norm, groups = self.action_group)
        for key, val in valid_metrics.items():
            is_key_metrics = (self.key_metrics == key)
            self.log(
                f"val_{key}", val,
                batch_size = action.size(0),
                prog_bar = is_key_metrics, logger = True
            )