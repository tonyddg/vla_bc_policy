from pathlib import Path
from typing import Optional, Sequence, Union

import json
from typing import Any, Sequence

import torch
from torch.utils.data import random_split, DataLoader

from lightning import pytorch as pl

from vla_bc_policy.dataset.camera_info import CameraInfo, dict_list_to_camera_info_list
from vla_bc_policy.dataset.pi0_lerobot_dataset import Pi0LeRobotDataset
from vla_bc_policy.dataset.utility import ACTION_SAMPLE_KEY, get_random_pi0_lerobot_batch

def get_pi0_lerobot_dataspec(
    repo_id: str, 
    camera_info_list: list[CameraInfo],

    root: Optional[Union[str, Path]] = None,
    vec_obs_keys: Sequence[str] = [
        "pi0_actions_ref", "pi0_state", "qpos", "last_action"
    ],
):
    '''
    从 Lerobot 元数据中获取一个 sample 的特征维度
    '''
    if root is None:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        root = HF_LEROBOT_HOME
    if isinstance(root, str):
        root = Path(root)
    info_json_path = root / repo_id / "meta/info.json"

    with open(info_json_path) as f:
        dataset_info = dict(json.load(f))

    vec_obs_dim = 0
    for vec_obs_key in vec_obs_keys:
        vec_single_dim = int(dataset_info["features"][vec_obs_key]["shape"][0])
        vec_obs_dim += vec_single_dim
    
    img_obs_dim = {}
    for camera_info in camera_info_list:
        img_obs_dim[camera_info.camera_name] = camera_info.get_in_channels()
    
    output_dim = int(dataset_info["features"][ACTION_SAMPLE_KEY]["shape"][0])
    return output_dim, vec_obs_dim, img_obs_dim

class Pi0LeRobotDataModule(pl.LightningDataModule):
    def __init__(
        self,

        repo_id: str, 
        camera_info_list: list[dict], # list[CameraInfo],
        vec_obs_keys: Sequence[str] = [
            "pi0_actions_ref", "pi0_state", "qpos", "last_action"
        ],
        # 使用 tanh 压缩具有显著离群点的特征 tanh(v/s), 键值为 s
        vec_obs_compress_key: dict[str, float] = {
            "qvel": 1.5 
        },

        batch_size: int = 512,
        num_workers: int = 8,

        # 使用独立的验证集
        val_repo_id: Optional[str] = None, 
        # 没有独立验证集时将从训练集划分
        val_ratio: float = 0.1,
        split_seed: int = 42,

        # 使用保守的 dataloader 配置
        debug_config: bool = False,

    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # self.camera_info_list = camera_info_list
        self.camera_info_list = dict_list_to_camera_info_list(camera_info_list)
        self.repo_id = repo_id
        self.vec_obs_keys = vec_obs_keys
        self.vec_obs_compress_key = vec_obs_compress_key

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.val_repo_id = val_repo_id
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        self.debug_config = debug_config

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        # 获取样本输入尺寸
        output_dim, vec_obs_dim, img_obs_dim = get_pi0_lerobot_dataspec(
            self.repo_id, self.camera_info_list, vec_obs_keys = self.vec_obs_keys
        )
        self.output_dim = output_dim
        self.vec_obs_dim = vec_obs_dim
        self.img_obs_dim = img_obs_dim

    def get_dataspec(self):
        return (
            self.output_dim,
            self.vec_obs_dim,
            self.img_obs_dim,
        )

    def get_random_batch(self, batch_size: int = 1):
        output_dim, vec_obs_dim, img_obs_dim = self.get_dataspec()
        return get_random_pi0_lerobot_batch(
            output_dim, vec_obs_dim, img_obs_dim, b = batch_size
        )

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str | None = None) -> None:
        
        if self.val_repo_id is None:
            if self.full_dataset is None:
                self.full_dataset = Pi0LeRobotDataset(
                    self.repo_id, self.camera_info_list, 
                    vec_obs_keys = self.vec_obs_keys,
                    vec_obs_compress_key = self.vec_obs_compress_key
                )

            if self.train_dataset is None or self.val_dataset is None:
                full_len = len(self.full_dataset)
                val_len = int(full_len * self.val_ratio)
                print(f"setup lerobot dataset with full len: {full_len}, val len: {val_len}")

                generator = torch.Generator().manual_seed(self.split_seed)
                self.train_dataset, self.val_dataset = random_split(
                    self.full_dataset,
                    [full_len - val_len, val_len],
                    generator = generator
                )
        else:
            if self.train_dataset is None or self.val_dataset is None:
                self.train_dataset = Pi0LeRobotDataset(
                    self.repo_id, self.camera_info_list, 
                    vec_obs_keys = self.vec_obs_keys,
                    vec_obs_compress_key = self.vec_obs_compress_key
                )
                self.val_dataset = Pi0LeRobotDataset(
                    self.val_repo_id, self.camera_info_list, 
                    vec_obs_keys = self.vec_obs_keys,
                    vec_obs_compress_key = self.vec_obs_compress_key
                )

    def train_dataloader(self) -> Any:
        assert self.train_dataset is not None, "Not setup for train dataset"

        if self.debug_config:
            return DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = 0,
                pin_memory = False,
                persistent_workers = False,
                prefetch_factor = None,
                drop_last = True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers = self.num_workers > 0,
                prefetch_factor = 2 if self.num_workers > 0 else None,
                drop_last = True,
            )

    def val_dataloader(self) -> Any:
        assert self.val_dataset is not None, "Not setup for val dataset"

        if self.debug_config:
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = 0,
                pin_memory = False,
                persistent_workers = False,
                prefetch_factor = None,
                drop_last = False,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers = self.num_workers > 0,
                prefetch_factor = 2 if self.num_workers > 0 else None,
                drop_last = False,
            )
    
    def stats_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        '''
        非接口实现, 用于统计训练集信息
        '''
        assert self.train_dataset is not None, "Call dm.setup() before stats_dataloader()"

        if batch_size is None:
            batch_size = self.batch_size

        if num_workers is None:
            num_workers = self.num_workers

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,                 # 统计不需要 shuffle
            num_workers=num_workers,
            pin_memory=False,              # 统计阶段通常不需要 pin_memory
            persistent_workers=num_workers > 0,
            prefetch_factor=1 if num_workers > 0 else None,
            drop_last=False,               # 统计必须保留最后一个不完整 batch
        )