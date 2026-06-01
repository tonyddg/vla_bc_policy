from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Union

import torch
from torch import Tensor, nn

from vla_bc_policy.dataset.utility import VECTION_OBS_KEY


NormalizeMethod = Literal["none", "standard", "minmax", "minmax_0_1"]


class JsonNormalizer(nn.Module):
    """
    约定：
    - obs["vector"] 是一维状态，shape:
        [D] / [B, D] / [B, T, D]
    - obs 中除 "vector" 外的所有 key 都是 channel-first 图像，shape:
        [C, H, W] / [B, C, H, W] / [B, T, C, H, W]
    - action 是动作向量，shape:
        [A] / [B, A] / [B, T, A]

    JSON 格式：
    {
      "obs": {
        "head": {"mean": [...], "std": [...], "min": [...], "max": [...]},
        "gripper": {...},
        "vector": {...}
      },
      "action": {"mean": [...], "std": [...], "min": [...], "max": [...]}
    }
    """

    STAT_NAMES = ("mean", "std", "min", "max")

    def __init__(
        self,
        stats_path: Union[str, Path],
        image_method: NormalizeMethod = "none",
        vector_method: NormalizeMethod = "standard",
        action_method: NormalizeMethod = "minmax",
        method_by_key: Optional[Mapping[str, NormalizeMethod]] = None,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.stats_path = str(stats_path)
        self.image_method: NormalizeMethod = image_method
        self.vector_method: NormalizeMethod = vector_method
        self.action_method: NormalizeMethod = action_method
        self.method_by_key: Dict[str, NormalizeMethod] = dict(method_by_key or {})
        self.eps = eps
        self.dtype = dtype

        self._buffer_names = {}

        with Path(stats_path).open("r", encoding="utf-8") as f:
            stats = json.load(f)

        self._register_stats(stats)

    def _safe_name(self, key: str, stat_name: str):
        name = f"{key}_{stat_name}"
        return re.sub(r"[^0-9a-zA-Z_]", "_", name)

    def _register_stats(self, stats: Mapping[str, Any]):
        """
        将 JSON 中的 mean/std/min/max 注册为 buffer。
        """
        for obs_key, stat in stats.get("obs", {}).items():
            path_key = f"obs.{obs_key}"
            self._register_one(path_key, stat)

        if "action" in stats:
            self._register_one("action", stats["action"])

    def _register_one(self, key: str, stat: Mapping[str, Any]):
        self._buffer_names[key] = {}

        for stat_name in self.STAT_NAMES:
            if stat_name not in stat:
                continue

            tensor = torch.as_tensor(stat[stat_name], dtype=self.dtype)
            buffer_name = self._safe_name(key, stat_name)

            self.register_buffer(buffer_name, tensor)
            self._buffer_names[key][stat_name] = buffer_name

    def _has_stats(self, key: str):
        return key in self._buffer_names

    def _get_stat(self, key: str, stat_name: str):
        if key not in self._buffer_names:
            raise KeyError(f"Missing stats for key: {key}")

        if stat_name not in self._buffer_names[key]:
            raise KeyError(f"Missing stat {stat_name!r} for key: {key}")

        return getattr(self, self._buffer_names[key][stat_name])

    def _resolve_method(self, key: str, is_image: bool):
        if key in self.method_by_key:
            return self.method_by_key[key]

        if key == "action":
            return self.action_method

        if key == f"obs.{VECTION_OBS_KEY}":
            return self.vector_method

        if is_image:
            return self.image_method

        return self.vector_method

    def _broadcast_vector_stat(self, stat: Tensor, x: Tensor):
        """
        vector/action:
            stat [D]
            x [D]       -> [D]
            x [B, D]    -> [1, D]
            x [B, T, D] -> [1, 1, D]
        """
        stat = stat.to(device=x.device, dtype=x.dtype).flatten()

        if stat.numel() == 1:
            return stat.reshape([1] * x.ndim)

        if x.shape[-1] != stat.numel():
            raise ValueError(
                f"Vector/action stats mismatch: "
                f"stat length={stat.numel()}, x.shape={tuple(x.shape)}"
            )

        shape = [1] * x.ndim
        shape[-1] = stat.numel()
        return stat.reshape(shape)

    def _broadcast_image_stat(self, stat: Tensor, x: Tensor):
        """
        channel-first image:
            stat [C]
            x [C, H, W]       -> [C, 1, 1]
            x [B, C, H, W]    -> [1, C, 1, 1]
            x [B, T, C, H, W] -> [1, 1, C, 1, 1]
        """
        stat = stat.to(device=x.device, dtype=x.dtype).flatten()

        if stat.numel() == 1:
            return stat.reshape([1] * x.ndim)

        if x.ndim < 3:
            raise ValueError(f"Image tensor should be channel-first, got shape={tuple(x.shape)}")

        channel_axis = x.ndim - 3

        if x.shape[channel_axis] != stat.numel():
            raise ValueError(
                f"Image channel stats mismatch: "
                f"stat length={stat.numel()}, x.shape={tuple(x.shape)}, "
                f"expected channel axis={channel_axis}"
            )

        shape = [1] * x.ndim
        shape[channel_axis] = stat.numel()
        return stat.reshape(shape)

    def _broadcast_stat(self, stat: Tensor, x: Tensor, is_image: bool):
        if is_image:
            return self._broadcast_image_stat(stat, x)
        return self._broadcast_vector_stat(stat, x)

    def _normalize_tensor(self, x: Tensor, key: str, is_image: bool, method: Optional[NormalizeMethod] = None):
        if not self._has_stats(key):
            return x

        if not torch.is_floating_point(x):
            x = x.float()

        method = method or self._resolve_method(key, is_image)

        if method == "none":
            return x

        if method == "standard":
            mean = self._broadcast_stat(self._get_stat(key, "mean"), x, is_image)
            std = self._broadcast_stat(self._get_stat(key, "std"), x, is_image)
            std = torch.clamp(std, min=self.eps)
            return (x - mean) / std

        if method == "minmax":
            min_v = self._broadcast_stat(self._get_stat(key, "min"), x, is_image)
            max_v = self._broadcast_stat(self._get_stat(key, "max"), x, is_image)
            denom = torch.clamp(max_v - min_v, min=self.eps)
            return 2.0 * (x - min_v) / denom - 1.0

        if method == "minmax_0_1":
            min_v = self._broadcast_stat(self._get_stat(key, "min"), x, is_image)
            max_v = self._broadcast_stat(self._get_stat(key, "max"), x, is_image)
            denom = torch.clamp(max_v - min_v, min=self.eps)
            return (x - min_v) / denom

        raise ValueError(f"Unknown normalize method: {method}")

    def _denormalize_tensor(self, x: Tensor, key: str, is_image: bool, method: Optional[NormalizeMethod] = None):
        if not self._has_stats(key):
            return x

        method = method or self._resolve_method(key, is_image)

        if method == "none":
            return x

        if method == "standard":
            mean = self._broadcast_stat(self._get_stat(key, "mean"), x, is_image)
            std = self._broadcast_stat(self._get_stat(key, "std"), x, is_image)
            std = torch.clamp(std, min=self.eps)
            return x * std + mean

        if method == "minmax":
            min_v = self._broadcast_stat(self._get_stat(key, "min"), x, is_image)
            max_v = self._broadcast_stat(self._get_stat(key, "max"), x, is_image)
            return (x + 1.0) * 0.5 * (max_v - min_v) + min_v

        if method == "minmax_0_1":
            min_v = self._broadcast_stat(self._get_stat(key, "min"), x, is_image)
            max_v = self._broadcast_stat(self._get_stat(key, "max"), x, is_image)
            return x * (max_v - min_v) + min_v

        raise ValueError(f"Unknown denormalize method: {method}")

    def normalize_obs(self, obs: Mapping[str, Tensor], method: Optional[NormalizeMethod] = None):
        out = {}

        for obs_key, value in obs.items():
            stat_key = f"obs.{obs_key}"
            is_image = obs_key != VECTION_OBS_KEY

            out[obs_key] = self._normalize_tensor(
                value,
                key=stat_key,
                is_image=is_image,
                method=method,
            )

        return out

    def denormalize_obs(self, obs: Mapping[str, Tensor], method: Optional[NormalizeMethod] = None):
        out = {}

        for obs_key, value in obs.items():
            stat_key = f"obs.{obs_key}"
            is_image = obs_key != VECTION_OBS_KEY

            out[obs_key] = self._denormalize_tensor(
                value,
                key=stat_key,
                is_image=is_image,
                method=method,
            )

        return out

    def normalize_action(self, action: Tensor, method: Optional[NormalizeMethod] = None):
        return self._normalize_tensor(
            action,
            key="action",
            is_image=False,
            method=method,
        )

    def denormalize_action(self, action: Tensor, method: Optional[NormalizeMethod] = None):
        return self._denormalize_tensor(
            action,
            key="action",
            is_image=False,
            method=method,
        )

    def forward(
        self,
        obs: Mapping[str, Tensor],
        action: Optional[Tensor] = None,
        method: Optional[NormalizeMethod] = None,
    ):
        obs = self.normalize_obs(obs, method=method)

        if action is None:
            return obs

        action = self.normalize_action(action, method=method)
        return obs, action
