from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

class KMeansCodebook(nn.Module):

    def __init__(
            self,
            num_clusters: int,
            action_dim: int,
            cluster_feature_dim: int,
            artifact_path: Optional[Union[str, Path]] = None,
        ) -> None:
        super().__init__()

        self.num_clusters = int(num_clusters)
        self.action_dim = int(action_dim)
        self.cluster_feature_dim = int(
            cluster_feature_dim
        )

        # 原始动作中, 取出哪几个索引构成聚类特征
        self.cluster_feature_indices = nn.Buffer(
            torch.zeros(
                cluster_feature_dim,
                dtype=torch.long,
            ),
            persistent=True,
        )
        # 原始动作特征到聚类特征的缩放因子
        self.cluster_feature_scales = nn.Buffer(
            torch.ones(
                cluster_feature_dim,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # 聚类中心对应的聚类特征
        self.cluster_feature_centers = nn.Buffer(
            torch.zeros(
                num_clusters,
                cluster_feature_dim,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # 聚类中心对应的原始动作
        self.action_centers = nn.Buffer(
            torch.zeros(
                num_clusters,
                action_dim,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # 动作残差的标准差, 用于标准化作为目标的动作残差
        self.residual_scale = nn.Buffer(
            torch.ones(
                action_dim,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # 各个类别中的动作数量
        self.cluster_counts = nn.Buffer(
            torch.zeros(
                num_clusters,
                dtype=torch.long,
            ),
            persistent=True,
        )
        # 缓和的类别重加权, 提高稀有类别的分类损失权重
        self.cluster_class_weights = nn.Buffer(
            torch.ones(
                num_clusters,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # 是否正确加载了 Kmeans 参数
        self.codebook_ready = nn.Buffer(
            torch.tensor(False),
            persistent=True,
        )

        if artifact_path is not None:
            self.load_codebook_artifact(artifact_path)

    @torch.no_grad()
    def load_codebook_artifact(
        self,
        path: Union[str, Path],
    ) -> None:
        artifact = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if artifact["schema_version"] != 1:
            raise ValueError(
                "Unsupported codebook schema version: "
                f"{artifact['schema_version']}"
            )

        expected = (
            self.num_clusters,
            self.action_dim,
            self.cluster_feature_dim,
        )

        received = (
            int(artifact["num_clusters"]),
            int(artifact["action_dim"]),
            int(artifact["cluster_feature_dim"]),
        )

        if received != expected:
            raise ValueError(
                "Codebook shape mismatch. "
                f"Model expects {expected}, artifact has {received}"
            )

        self.cluster_feature_indices.copy_(
            artifact["feature_indices"]
        )
        self.cluster_feature_scales.copy_(
            artifact["feature_scales"]
        )
        self.cluster_feature_centers.copy_(
            artifact["cluster_feature_centers"]
        )
        self.action_centers.copy_(
            artifact["action_centers"]
        )
        self.residual_scale.copy_(
            artifact["residual_scale"]
        )
        self.cluster_counts.copy_(
            artifact["cluster_counts"]
        )
        self.cluster_class_weights.copy_(
            artifact["cluster_class_weights"]
        )

        self.codebook_ready.fill_(True)

    def _check_codebook(self) -> None:
        if not bool(self.codebook_ready.item()):
            raise RuntimeError(
                "K-means codebook has not been initialized, \nrun self.load_codebook_artifact() to load"
            )

    def encode_action_feature(
        self,
        action_norm: torch.Tensor,
    ) -> torch.Tensor:
        '''
        将动作转为聚类特征
        '''
        self._check_codebook()

        return (
            action_norm.index_select(
                dim=-1,
                index=self.cluster_feature_indices,
            )
            * self.cluster_feature_scales
        )

    @torch.no_grad()
    def assign_action_cluster(
        self,
        action_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取动作 Batch 所属的聚类类别
        action_norm: (B, A)
        return:      (B,)
        """
        feature = self.encode_action_feature(
            action_norm.float()
        )  # (B, F)

        # squared Euclidean distance: (B, K)
        distance = (
            feature[:, None, :]
            - self.cluster_feature_centers[None, :, :]
        ).square().sum(dim=-1)

        return distance.argmin(dim=1)

    def decode_action(
        self,
        cluster_idx: torch.Tensor,
        residual_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于聚类类别与相对聚类中心的残差动作, 还原出原始动作
        cluster_idx:         (B,)
        residual_normalized: (B, A)
        return:              (B, A)
        """
        center = self.action_centers[
            cluster_idx
        ]

        return (
            center
            + residual_normalized
            * self.residual_scale
        )
