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

class ChoicePolicy(pl.LightningModule):
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

        # choice policy 参数
        # 等于 0 时退化为 BC
        # 等于 1 时仅预测距离
        num_proposals: int = 4,
        # 类型总是 mlp decoder, 默认直接将特征映射为分数
        score_pred_kwargs: Dict[str, Any] = dict(
            is_output_proj = True
        ),

        # 优化器参数
        lr: float = 2e-3,
        optim_type: Literal["AdamW", "Adam", "SGD"] = "AdamW",
        weight_decay: float = 5e-6,
        sgd_momentum: float = 0.99,

        distance_loss_type: LossFnType = "mse",
        score_loss_type: LossFnType = "mse",
        key_metrics: str = "RMSE", # 核心指标, 根号均方误差

        # loss 超参数
        score_loss_weight: float = 1.0,
        score_loss_warmup_ratio: float = 0.05, # 前 0.05 的 step 逐渐增大
        # # 风格损失, 暂时排除绝对位置动作
        # style_loss_weight: Optional[float] = 1e-3, # 所有头的风格损失
        # style_loss_type: LossFnType = "mse",
        # style_group: Dict[str, tuple[int, int]] = dict(
        #     joint = (0, 7),
        #     base = (8, 10)
        # ),
        # # 幅度过小的动作不进入风格监督
        # energy_threshold_q: float = 0.10,
        # target_style: list[list[float]] = [
        #     [0.8, 0.2],
        #     [0.6, 0.4],
        #     [0.4, 0.6],
        #     [0.2, 0.8],
        # ],

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

        assert num_proposals >= 0, f"num_proposals must >= 0, current is {num_proposals}"

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
            score_pred_kwargs = dict(score_pred_kwargs)
            score_pred_kwargs["num_out_feats"] = num_proposals
            score_pred_kwargs["num_in_feats"] = decoder_kwargs["num_out_feats"]

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

        # 创建动作头
        self.num_proposals = num_proposals
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
                max(self.num_proposals, 1) # 0 纯 BC; 1 带有距离预测的 BC
            )
        ])

        # 创建动作评分头 (使用独立模型防止数量级偏差, 分数越小与真样本距离越近, 越好)
        if self.num_proposals > 0:
            self.score_pred = MlpDecoder(**score_pred_kwargs)
        else:
            self.score_pred = None

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
        self.score_loss_fn = LossFnDict[score_loss_type](reduction = "none")
        # self.style_loss_fn = LossFnDict[style_loss_type](reduction = "none")

        # loss 超参数
        self.score_loss_weight = score_loss_weight
        self.score_loss_warmup_ratio = score_loss_warmup_ratio
        if self.num_proposals > 1:
            self.max_freq_std = math.sqrt(self.num_proposals - 1) / self.num_proposals
        else:
            self.max_freq_std = 1.0

        # self.style_loss_weight = style_loss_weight
        # self.style_group = style_group
        # self.target_style = target_style
        # self.energy_threshold_q = energy_threshold_q

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

        proposal_actions = torch.stack([
            # (B, A)
            action_head(feat) for action_head in self.action_head_list
        ], 1) # (B, P, A)

        if self.score_pred is not None:
            # (B, P)
            score = self.score_pred(feat)
        else:
            score = proposal_actions.new_zeros(
                (proposal_actions.shape[0], 1),
            )

        # (B, P, A), (B, P)
        return proposal_actions, score

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        proposal_actions_norm, pred_score = self.forward(obs)

        if self.num_proposals > 0:
            best_idx = torch.argmin(pred_score, 1)
            selected_actions_norm = proposal_actions_norm[
                # argmin 结果元素所在的索引对应了该结果所对应的 batch 维坐标
                torch.arange(best_idx.shape[0], device = proposal_actions_norm.device), 
                best_idx
            ]
        else:
            selected_actions_norm = proposal_actions_norm.squeeze(dim = 1)

        return self.normalizer.denormalize_action(
            selected_actions_norm
        )

    def compute_loss(self, obs, gt_action: torch.Tensor):
        # pred_action, pred_score = self.model(data)
        # if self.num_proposals > 1:
        #     _gt = gt_action[:, None].repeat(1, self.num_proposals, 1, 1)
        #     loss = nn.functional.mse_loss(_gt, pred_action, reduction='none').mean(dim=(2, 3))
        #     score_loss = ((pred_score - loss.detach()) ** 2)
        #     if self.clip_score_loss:
        #         score_loss = torch.clip(score_loss, max=self.clip_score_loss_max)
        #     score_loss = score_loss.mean()
        #     loss_mask = loss.argmin(dim=1)
        #     loss = loss[torch.arange(loss.shape[0]), loss_mask]
        #     loss = loss.mean() + score_loss
        # else:
        #     pred_action = pred_action.reshape(-1, self.pred_horizon, self.action_dim)
        #     loss = nn.functional.mse_loss(gt_action, pred_action)
        # 动作模式权重 (鼓励特定动作模式) 选择正则 (模式均等)

        (
            pred_action,    # (B, P, A)
            pred_score      # (B, P)
        ) = self.forward(obs)
        gt_action = self.normalizer.normalize_action(gt_action) # (B, A)
        batch_size = pred_action.shape[0]
        batch_rows = torch.arange(batch_size, device = pred_action.device)

        if self.num_proposals > 0:
            action_error = torch.as_tensor(self.distance_loss_fn(
                pred_action, 
                # 复制扩展为相同形状计算 loss
                gt_action.unsqueeze(1).expand(-1, self.num_proposals, -1)
            )) # (B, P, A)
            action_distance = torch.mean(action_error, dim = 2) # (B, P)
            gt_score = action_distance.detach()

            # TODO: action_error 计算 action loss 时设置各个维度的权重

            # 基于距离取出选用动作的距离计算动作 loss
            oracle_proposal_idx = torch.argmin(gt_score, 1) # (B)
            oracle_action_distance = action_distance[
                batch_rows, 
                oracle_proposal_idx
            ] # (B)
            action_loss = oracle_action_distance.mean()

            # 将距离作为真实样本，计算预测 loss
            score_loss = torch.as_tensor(self.score_loss_fn(pred_score, gt_score))
            score_loss = torch.mean(score_loss)

            # # 风格正则损失 (总是对各个预测动作施加) (风格损失与 WTA 的动作损失冲突)
            # proposal_style, proposal_total_energy = compute_group_energy(
            #     pred_action, self.style_group
            # )
            # oracle_style, oracle_total_energy = compute_group_energy(
            #     gt_action[:, None, :], self.style_group
            # )
            # oracle_total_energy = oracle_total_energy.squeeze(1) # (B,)
            # oracle_energy_threshold = torch.quantile(oracle_total_energy, self.energy_threshold_q)
            # energy_mask = oracle_total_energy > oracle_energy_threshold # (B,)

            # target_style = torch.tensor(self.target_style, dtype = torch.float32, device = self.device).unsqueeze(0)
            # style_error = self.style_loss_fn(proposal_style, target_style).mean(dim = (1, 2)) # (B,)
            # if energy_mask.any():
            #     style_loss = style_error[energy_mask].mean()
            # else:
            #     style_loss = pred_action.new_zeros(())

            # 总 loss
            loss = action_loss + score_loss * self.score_loss_weight
            # if self.style_loss_weight is not None:
            #     loss += self.style_loss_weight * style_loss

            ##################
            # 额外信息
            selected_proposal_idx = torch.argmin(pred_score, 1) # (B)
            # 生成选择动作时, 依然基于 pred_score 而不是 gt_score
            selected_actions_norm = pred_action[
                batch_rows, 
                selected_proposal_idx
            ] # (B, A)
            selected_action_distance = action_distance[
                batch_rows, 
                selected_proposal_idx
            ] # (B)

            # 模型输出选择的分支, 检查哪个分支被经常使用到
            selected_freq = torch.bincount(
                selected_proposal_idx, minlength = self.num_proposals,
            ).float() / selected_proposal_idx.numel()
            selected_freq_std = torch.std(selected_freq, correction = 0)
            # 正确预测被训练的分支, 检查哪个分支被经常训练到
            oracle_freq = torch.bincount(
                oracle_proposal_idx, minlength = self.num_proposals,
            ).float() / oracle_proposal_idx.numel()
            oracle_freq_std = torch.std(oracle_freq, correction = 0)

            # pred_action: (B, P, A)
            pairwise_distance = torch.cdist(
                pred_action.float(),
                pred_action.float(),
                p=2,
            )  # (B, P, P)

            pair_mask = torch.triu(
                torch.ones(
                    self.num_proposals,
                    self.num_proposals,
                    dtype=torch.bool,
                    device=pred_action.device,
                ),
                diagonal=1,
            )
            mean_pairwise_distance = pairwise_distance[:, pair_mask].mean()
            min_pairwise_distance = pairwise_distance[:, pair_mask].min(dim=1).values.mean()

            loss_info = dict(
                action_loss = action_loss.detach(),
                score_loss = score_loss.detach(),
                # style_loss = style_loss.detach(),
                # 基于预测距离选择动作时, 相对真实距离选择动作时, 额外的误差大小
                selector_regret = torch.mean(selected_action_distance - oracle_action_distance).detach(),
                # 模型输出选择的分支, 越接近 1 越说明其他分支无法发挥作用
                selected_freq_collapse = selected_freq_std.detach() / self.max_freq_std,
                # 正确预测被训练的分支, 越接近 1 越说明其他分支无法发挥作用
                oracle_freq_collapse = oracle_freq_std.detach() / self.max_freq_std,
                # 所有 proposal 的两两距离的平均值
                mean_pairwise_distance = mean_pairwise_distance.detach(),
                # 所有 proposal 的两两距离的最小值
                min_pairwise_distance = min_pairwise_distance.detach(),

                # # 能量下限
                # oracle_energy_threshold = oracle_energy_threshold.detach(),
                # # 平均能量
                # oracle_energy_mean = oracle_total_energy.mean().detach(),
                # # 受到风格监督的动作数量
                # active_ratio = energy_mask.float().mean().detach(),
            )

        else:
            pred_action = torch.squeeze(pred_action, 1)
            loss = torch.mean(self.distance_loss_fn(pred_action, gt_action))

            selected_actions_norm = pred_action.detach()
            loss_info = None

        return loss, selected_actions_norm, loss_info
    
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