from __future__ import annotations

from vla_bc_policy.dataset.normalizer import JsonNormalizer
from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch

from tqdm import tqdm


def build_cluster_transform(
    # 用于检查
    action_dim: int,
    # 聚类动作分组、排除不参与聚类的动作, 将不同类别的动作除以 sqrt(l) 保证各类别动作在聚类空间上计算聚类距离 (MSE) 时平衡
    cluster_groups: Dict[str, tuple[int, int]],
    # 各组在聚类空间上的权重
    group_weights: Optional[Dict[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造将动作转为聚类特征的变换
    返回：
        feature_indices: (F,)
        feature_scales:  (F,)

    feature = action[..., feature_indices] * feature_scales
    """
    group_weights = group_weights or {
        name: 1.0 for name in cluster_groups
    }

    indices: list[int] = []
    scales: list[float] = []

    for name, (start, end) in cluster_groups.items():
        if not (0 <= start < end <= action_dim):
            raise ValueError(
                f"Invalid group {name}: {(start, end)} "
                f"for action_dim={action_dim}"
            )

        width = end - start
        weight = float(group_weights[name])

        if weight <= 0:
            raise ValueError(
                f"Group weight must be positive, got {name}={weight}"
            )

        # 令该组距离贡献为 weight * group MSE
        per_dim_scale = math.sqrt(weight / width)

        indices.extend(range(start, end))
        scales.extend([per_dim_scale] * width)

    return (
        # 参与聚类的动作索引
        torch.tensor(indices, dtype=torch.long),
        # 聚类空间权重
        torch.tensor(scales, dtype=torch.float32),
    )

def encode_cluster_feature(
    action_norm: torch.Tensor,
    feature_indices: torch.Tensor,
    feature_scales: torch.Tensor,
) -> torch.Tensor:
    """
    将标准化动作转为聚类空间动作
    action_norm: (..., A)
    return:      (..., F)
    """
    indices = feature_indices.to(action_norm.device)
    scales = feature_scales.to(
        device=action_norm.device,
        dtype=action_norm.dtype,
    )

    return action_norm.index_select(
        dim=-1,
        index=indices,
    ) * scales

@torch.no_grad()
def collect_normalized_actions(
    loader: Iterable,
    normalizer: JsonNormalizer,
    max_batches: Optional[int] = None,

    progress: bool = True,
) -> torch.Tensor:
    '''
    将整个 LeRobot 数据集的动作加载到内存中
    '''
    actions: list[torch.Tensor] = []

    if progress:
        iterator = tqdm(loader, desc="Load all actions")
    else:
        iterator = loader

    for batch_idx, (_, action_batch) in enumerate(iterator):
        action_norm = normalizer.normalize_action(
            action_batch
        ).detach().cpu().float()

        if action_norm.ndim != 2:
            raise ValueError(
                f"Expected action shape (B, A), got {action_norm.shape}"
            )

        actions.append(action_norm)

        if (
            max_batches is not None
            and batch_idx + 1 >= max_batches
        ):
            break

    if not actions:
        raise RuntimeError("Action dataloader was empty")

    return torch.cat(actions, dim=0)

@torch.no_grad()
def fit_and_save_action_kmeans(
    actions_norm: torch.Tensor,  # (N, A)
    output_path: str | Path,
    num_clusters: int = 8,
    cluster_groups: Dict[str, tuple[int, int]] = {
        "joint": (0, 7),
        "base": (8, 10),
    },
    group_weights: Optional[Dict[str, float]] = None,
    random_state: int = 42,
    n_init: int = 20,

    verbose: int = 0
) -> dict:
    """
    在 group-balanced 特征空间拟合 K-means，
    再计算每类的完整动作中心。
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score  
    
    ###

    if actions_norm.ndim != 2:
        raise ValueError(
            f"Expected (N, A), got {actions_norm.shape}"
        )

    actions_norm = actions_norm.detach().cpu().float()
    num_samples, action_dim = actions_norm.shape

    if num_clusters <= 1:
        raise ValueError("num_clusters must be greater than 1")

    if num_clusters > num_samples:
        raise ValueError(
            "num_clusters cannot exceed number of samples"
        )

    ###

    # 获取到聚类特征的变换
    feature_indices, feature_scales = (
        build_cluster_transform(
            action_dim=action_dim,
            cluster_groups=cluster_groups,
            group_weights=group_weights,
        )
    )

    # 将数据集全部动作转为聚类特征
    cluster_features = encode_cluster_feature(
        actions_norm,
        feature_indices,
        feature_scales,
    )  # (N, F)

    # 显式设置 n_init，避免不同 sklearn 默认配置导致结果变化。
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=random_state,
        algorithm="lloyd",

        verbose = verbose
    )

    # 记录各个动作所属类别
    labels_np = kmeans.fit_predict(
        cluster_features.numpy()
    ) # (N,)
    labels = torch.from_numpy(labels_np).long()
    # 各个类别中的动作数量
    cluster_counts = torch.bincount(
        labels,
        minlength=num_clusters,
    )
    # 检查没有任何元素的类别
    if (cluster_counts == 0).any():
        empty_clusters = (
            torch.where(cluster_counts == 0)[0]
            .tolist()
        )
        raise RuntimeError(
            f"Empty clusters found: {empty_clusters}"
        )

    # 根据最终标签重新计算精确均值。
    action_centers = torch.zeros(
        num_clusters,
        action_dim,
        dtype=torch.float32,
    )
    # 在 action_centers[labels[i]] += actions_norm[i]
    # 即计算各个标签对应所用动作之和, 为后续计算均值做准备
    # 屏蔽的动作维度不参与聚类, 但参与中心动作的计算
    action_centers.index_add_(
        dim=0,
        index=labels,
        source=actions_norm,
    )
    action_centers /= cluster_counts[:, None].float()

    # 变换是线性的，因此完整动作均值映射后就是特征中心。
    cluster_feature_centers = encode_cluster_feature(
        action_centers,
        feature_indices,
        feature_scales,
    )

    # BeT 风格的残差目标 (动作头预测相对所属风格中心动作的残差)
    residuals = (
        actions_norm
        - action_centers[labels]
    )

    # 作为 residual target 的每维尺度。
    residual_scale = residuals.std(
        dim=0,
        correction=0,
    ).clamp_min(1e-3)

    # 各个类别中样本数占比
    cluster_fraction = (
        cluster_counts.float()
        / cluster_counts.sum()
    )
    # 缓和的类别重加权, 提高稀有类别的分类损失权重, 同时避免稀有类权重过大。
    cluster_class_weights = (
        cluster_fraction.mean()
        / cluster_fraction.clamp_min(1e-8)
    ).sqrt()
    cluster_class_weights /= (
        cluster_class_weights.mean()
    )

    # 衡量聚类效果
    # 残差目标 RMSE, 越小聚类越紧密
    quantization_rmse = (
        residuals.square()
        .mean()
        .sqrt()
    )

    artifact = {
        "schema_version": 1,
        "num_clusters": num_clusters,
        "action_dim": action_dim,
        "cluster_feature_dim": int(
            # 聚类特征维度
            feature_indices.numel()
        ),
        "normalization_space": "model_normalized_action",

        "feature_indices": feature_indices,
        "feature_scales": feature_scales,

        "cluster_feature_centers":
            cluster_feature_centers,

        "action_centers": action_centers,
        "residual_scale": residual_scale,

        "cluster_counts": cluster_counts,
        "cluster_fraction": cluster_fraction,
        "cluster_class_weights":
            cluster_class_weights,

        "quantization_rmse": quantization_rmse,

        "cluster_groups": {
            name: [start, end]
            for name, (start, end)
            in cluster_groups.items()
        },

        "random_state": random_state,
        "n_init": n_init,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(artifact, output_path)

    print(f"Saved K-means artifact to: {output_path}")
    print(
        f"K={num_clusters}, "
        f"N={num_samples}, "
        f"quantization RMSE={quantization_rmse.item():.6f}"
    )
    print(
        "Cluster fraction:",
        cluster_fraction.tolist(),
    )

    return artifact

def main(
    repo_id: str,
    stats_path: str,

    output_path: Union[str, Path] = "output/action_kmeans.pt",
    num_clusters: int = 8,
    cluster_groups: Dict[str, tuple[int, int]] = {
        "joint": (0, 7),
        "base": (8, 10),
    },

    is_split_val: bool = False,
    batch_size: int = 512,
    num_workers: int = 16,
    verbose: bool = True
):
    from vla_bc_policy.dataset.camera_info import FetchStandardCameraInfos, camera_info_list_to_dict_list

    dm = Pi0LeRobotDataModule(
        repo_id = repo_id + "_train",
        val_repo_id = None,
        is_split_val = is_split_val,
        # 后续改为通过独立的配置文件加载
        camera_info_list = camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False,

        vec_obs_keys = ["pi0_actions_ref", "pi0_state", "qpos", "qvel"],
        vec_obs_compress_key = { "qvel": 1.5 },
    )
    
    dm.setup(stage="fit")
    loader = dm.stats_dataloader(
        batch_size = batch_size,
        num_workers = num_workers,
    )

    normalizer = JsonNormalizer(
        stats_path,
        image_method = "none",
        vector_method = "standard",
        action_method = "minmax",
    )

    actions_norm = collect_normalized_actions(
        loader = loader,
        normalizer = normalizer,
        progress = verbose
    )

    fit_and_save_action_kmeans(
        actions_norm = actions_norm,
        output_path = output_path,
        num_clusters = num_clusters,
        cluster_groups = cluster_groups,
        random_state = 42,
        n_init = 20,
        # verbose = 1 if verbose else 0
    )

# uv run -m vla_bc_policy.dataset.fit_kmeans --repo_id trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle --stats_path /root/vla_bc_policy/assets/tidy_house/train_stats_compress_qvel.json
if __name__ == "__main__":
    import tyro
    tyro.cli(main)