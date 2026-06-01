from pathlib import Path
from typing import Optional, Union, Any
import json
import torch
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy.dataset.utility import VECTION_OBS_KEY

class RunningStats:
    """
    流式统计 mean / std / min / max。

    支持两种模式：
    1. feature: 输入 [B, D]，统计每个 feature dim
    2. channel: 输入 [B, C, H, W] 或 [B, C, ...]，统计每个 channel
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

        self.count: int = 0
        self.sum: Optional[torch.Tensor] = None
        self.sq_sum: Optional[torch.Tensor] = None
        self.min: Optional[torch.Tensor] = None
        self.max: Optional[torch.Tensor] = None

    def _init_if_needed(self, shape: torch.Size):
        if self.sum is not None:
            return

        self.sum = torch.zeros(shape, dtype=torch.float64)
        self.sq_sum = torch.zeros(shape, dtype=torch.float64)
        self.min = torch.full(shape, float("inf"), dtype=torch.float64)
        self.max = torch.full(shape, float("-inf"), dtype=torch.float64)

    @torch.no_grad()
    def update_feature(self, x: torch.Tensor):
        """
        x: [B, D] 或 [B]
        """
        x = torch.as_tensor(x).detach().cpu().to(torch.float64)

        if x.ndim == 1:
            # [B] -> 统计一个 scalar feature
            reduce_dims = (0,)
            stat_shape = torch.Size([])
            n = x.shape[0]
        else:
            # [B, D]
            reduce_dims = (0,)
            stat_shape = x.shape[1:]
            n = x.shape[0]

        self._init_if_needed(stat_shape)

        assert self.sum is not None
        assert self.sq_sum is not None
        assert self.min is not None
        assert self.max is not None

        self.sum += x.sum(dim=reduce_dims)
        self.sq_sum += (x * x).sum(dim=reduce_dims)
        self.min = torch.minimum(self.min, x.amin(dim=reduce_dims))
        self.max = torch.maximum(self.max, x.amax(dim=reduce_dims))
        self.count += int(n)

    @torch.no_grad()
    def update_channel(self, x: torch.Tensor):
        """
        x: [B, C, H, W] 或 [B, C, ...]
        按 channel 统计，输出 shape 为 [C]
        """
        x = torch.as_tensor(x).detach().cpu().to(torch.float64)

        if x.ndim < 2:
            raise ValueError(f"Expected image-like tensor with shape [B, C, ...], got {x.shape}")

        # 保留 channel dim=1，其他维度都 reduce
        reduce_dims = tuple(i for i in range(x.ndim) if i != 1)
        stat_shape = torch.Size([x.shape[1]])

        n = 1
        for dim in reduce_dims:
            n *= x.shape[dim]

        self._init_if_needed(stat_shape)

        assert self.sum is not None
        assert self.sq_sum is not None
        assert self.min is not None
        assert self.max is not None

        self.sum += x.sum(dim=reduce_dims)
        self.sq_sum += (x * x).sum(dim=reduce_dims)
        self.min = torch.minimum(self.min, x.amin(dim=reduce_dims))
        self.max = torch.maximum(self.max, x.amax(dim=reduce_dims))
        self.count += int(n)

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("RunningStats is empty.")

        assert self.sum is not None
        assert self.sq_sum is not None
        assert self.min is not None
        assert self.max is not None

        mean = self.sum / self.count
        var = self.sq_sum / self.count - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + self.eps)

        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

@torch.no_grad()
def compute_pi0_lerobot_stats_from_datamodule(
    dm: Pi0LeRobotDataModule,

    output_path: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_batches: Optional[int] = None,
    compute_image_stats: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """
    基于 Pi0LeRobotDataModule 已经划分好的 train_dataset 计算 stats。

    注意：
    - 不重新 random_split。
    - 不统计 val_dataset。
    - 使用 dm.stats_dataloader()，而不是直接用 dm.train_dataloader()。
    """

    dm.setup(stage = "fit")

    loader = dm.stats_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    obs_stats: dict[str, RunningStats] = {}
    action_stats = RunningStats()

    iterator = loader
    if progress:
        iterator = tqdm(loader, desc="Computing train stats")

    num_batches = 0
    num_samples_seen = 0

    for obs_batch, action_batch in iterator:
        num_batches += 1
        num_samples_seen += int(action_batch.shape[0])

        action_stats.update_feature(action_batch)

        for key, value in obs_batch.items():
            if key not in obs_stats:
                obs_stats[key] = RunningStats()

            if key == VECTION_OBS_KEY:
                obs_stats[key].update_feature(value)
            else:
                if compute_image_stats:
                    obs_stats[key].update_channel(value)

        if max_batches is not None and num_batches >= max_batches:
            break

    assert dm.full_dataset is not None
    stats = {
        "repo_id": dm.repo_id,
        "split": {
            "val_ratio": dm.val_ratio,
            "split_seed": dm.split_seed,
            "full_len": len(dm.full_dataset), # type: ignore
            "train_len": len(dm.train_dataset), # type: ignore
            "val_len": len(dm.val_dataset), # type: ignore
            "num_samples_seen": num_samples_seen,
            "num_batches_seen": num_batches,
            "max_batches": max_batches,

            "vec_obs_keys": dm.full_dataset.vec_obs_keys
        },
        "obs": {
            key: stat.to_dict()
            for key, stat in obs_stats.items()
        },
        "action": action_stats.to_dict(),
    }

    if output_path is None:
        output_path = HF_LEROBOT_HOME / dm.repo_id / "train_stats.json"
    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(stats, f, indent = 2)

    print(f"Saved train stats to: {output_path}")

    return stats

if __name__ == "__main__":
    from vla_bc_policy.dataset.camera_info import FetchStandardCameraInfos, camera_info_list_to_dict_list

    dm = Pi0LeRobotDataModule(
        "trajectories_tidy_house_all_bc_fix_la_state_rot_6d_action_axis_angle",
        camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False
    )

    stats = compute_pi0_lerobot_stats_from_datamodule(
        dm,
        output_path = "output/train_stats.json",
        batch_size = 512,
        num_workers = 16,

        # 调试时可以先开：
        # max_batches = 10,
    )