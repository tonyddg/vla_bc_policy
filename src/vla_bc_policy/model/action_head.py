from typing import Any, Dict, Optional
import torch
from torch import nn

from vla_bc_policy.model.mlp_decoder import MlpDecoder

class MultiActionHead(nn.Module):
    def __init__(
        self,
        num_in_feats: int,
        head_decoder_config: Dict[str, tuple[
            tuple[int, int], # 输出位置索引, 从小到大
            Dict[str, Any] # 解码器超参数
        ]],

        # 仅用于校验
        num_out_feats: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        self.output_idx: Dict[str, tuple[int, int]] = {}
        self.num_out_feats = 0

        used_ranges: list[tuple[int, int, str]] = []

        for key, cfg in head_decoder_config.items():

            idx_range = cfg[0]
            self.output_idx[key] = idx_range
            decoder_cfg = dict(cfg[1])
            self.num_out_feats = max(cfg[0][1], self.num_out_feats)

            # 合法性检查
            start, end = idx_range
            if start < 0:
                raise ValueError(f"{key}: start index must be >= 0, got {start}")
            if end <= start:
                raise ValueError(
                    f"{key}: invalid output range {idx_range}, expected end > start"
                )
            for prev_start, prev_end, prev_key in used_ranges:
                overlap = not (end <= prev_start or start >= prev_end)
                if overlap:
                    raise ValueError(
                        f"Output range of {key} {idx_range} overlaps with "
                        f"{prev_key} {(prev_start, prev_end)}"
                    )
            used_ranges.append((start, end, key))

            # 创建模型
            decoder_cfg["num_in_feats"] = num_in_feats
            decoder_cfg["num_out_feats"] = cfg[0][1] - cfg[0][0]
            self.heads[key] = MlpDecoder(**decoder_cfg)
        
        if num_out_feats is not None:
            assert num_out_feats == self.num_out_feats, "num_out_feats by config is not equal to given num_out_feats"

    def forward(self, X: torch.Tensor):
        batch_size = X.size(0)
        y = X.new_zeros((batch_size, self.num_out_feats))

        for key, idx_range in self.output_idx.items():
            start, end = idx_range
            y[:, start : end] = self.heads[key](X)
        return y

from typing import TypeAlias
SingleActionHead: TypeAlias = MlpDecoder
