from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Type, Union
import torch
from torch import nn
from torch import optim
import lightning.pytorch as pl

from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy.dataset.normalizer import JsonNormalizer, NormalizeMethod

from vla_bc_policy.model.extractor import ExtractorType, ExtractorDict
from vla_bc_policy.model.mlp_decoder import MlpDecoder
from vla_bc_policy.model.res_mlp_decoder import ResMlpDecoder
from vla_bc_policy.model.multi_head_decoder import MultiHeadDecoder
from vla_bc_policy.model.utility import regression_metrics, LRScheduleType, WarmupCosineLR

class PolicyModule(pl.LightningModule):
    def __init__(
        self,

        # 模型参数
        extractor_type: ExtractorType,
        extractor_kwargs: Dict[str, Any],
        decoder_type: Literal["mlp", "res_mlp"],
        decoder_kwargs: Dict[str, Any],
        # 标准化信息参数
        stats_path: Union[str, Path],

        # 输出分组
        action_group: dict[str, tuple[int, int]] = dict(
            joint = (0, 7),
            torso = (7, 8),
            base = (8, 10)
        ),
        # 使用多头输出
        use_multi_head_decoder: bool = False,
        # 多头输出配置
        multi_head_decoder_config: Optional[Dict[str, Any]] = None,

        # 优化器参数
        lr: float = 2e-3,
        is_use_adam: bool = True,
        weight_decay: float = 5e-6,
        sgd_momentum: float = 0.99,

        loss_type: Literal["mse", "l1", "smooth_l1"] = "mse",
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

        # 在 self.save_hyperparameters 前保存输入信息, 确保没有数据集模型也可以加载
        if data_module is not None:
            vec_obs_dim = data_module.vec_obs_dim
            img_obs_dim = data_module.img_obs_dim
            output_dim = data_module.output_dim

            # 在 self.save_hyperparameters 前注入输入相关参数, 确保没有数据集模型也可以加载
            extractor_kwargs = dict(extractor_kwargs)
            extractor_kwargs["vec_obs_dim"] = vec_obs_dim
            extractor_kwargs["img_obs_dim"] = img_obs_dim

            if use_multi_head_decoder:
                assert multi_head_decoder_config is not None
                multi_head_decoder_config = dict(multi_head_decoder_config)
                multi_head_decoder_config["num_out_feats"] = output_dim
                multi_head_decoder_config["num_in_feats"] = decoder_kwargs["num_out_feats"]
            else:
                decoder_kwargs = dict(decoder_kwargs)
                decoder_kwargs["num_out_feats"] = output_dim

        # 将 __init__ 参数保存到字典 self.hparams 中
        self.save_hyperparameters(ignore = ["data_module", ])

        # 创建模型
        self.extractor = ExtractorDict[extractor_type](**extractor_kwargs)
        decoder_kwargs["num_in_feats"] = self.extractor.get_out_feat()
        if decoder_type == "mlp":
            self.decoder = MlpDecoder(**decoder_kwargs)
        elif decoder_type == "res_mlp":
            self.decoder = ResMlpDecoder(**decoder_kwargs)
        else: 
            raise RuntimeError(f"Unknown decoder type: {decoder_type}")
        
        self.action_group = action_group
        if use_multi_head_decoder:
            assert multi_head_decoder_config is not None
            self.multi_head_decoder = MultiHeadDecoder(**multi_head_decoder_config)
        else:
            self.multi_head_decoder = None

        # 优化器参数
        self.lr = lr
        self.is_use_adam = is_use_adam
        self.weight_decay = weight_decay
        self.sgd_momentum = sgd_momentum

        # 学习率调度参数
        self.lr_schedule_type: LRScheduleType = lr_schedule_type
        self.lr_schedule_kwargs = lr_schedule_kwargs
        self.key_metrics = key_metrics

        # loss 函数
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise RuntimeError(f"Unknown loss type: {loss_type}")

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

        if self.is_use_adam:
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
        else: 
            # 单个优化器
            return use_optimizer

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # print(f"obs type {type(obs)}")
        obs = self.normalizer.normalize_obs(obs)

        feat = self.extractor(obs)
        feat = self.decoder(feat)

        if self.multi_head_decoder is None:
            return feat
        else:
            return self.multi_head_decoder(feat)

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_norm = self.forward(obs)
        return self.normalizer.denormalize_action(action_norm)
    
    def training_step(self, batch: tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):
        obs, action = batch

        pred_action_norm = self(obs)
        target_action_norm = self.normalizer.normalize_action(action)
        train_loss = self.loss_fn(pred_action_norm, target_action_norm)

        self.log(
            "train_loss", train_loss, 
            batch_size = action.size(0),
            prog_bar = True, logger = True
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

        pred_action_norm = self(obs)
        target_action_norm = self.normalizer.normalize_action(action)
        valid_loss = self.loss_fn(pred_action_norm, target_action_norm)

        self.log(
            "valid_loss", valid_loss, 
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
