from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from vla_bc_policy.dataset.camera_info import CameraInfo
from vla_bc_policy.dataset.utility import ACTION_SAMPLE_KEY, VECTION_OBS_KEY
from vla_bc_policy.dataset.sample_to_obs_config import Sample2ObsConfig

class Pi0LeRobotDataset(Dataset):
    def __init__(
        self, 
        repo_id: str, 
        camera_info_list: list[CameraInfo],

        root: Optional[Union[str, Path]] = None,
        vec_obs_keys: Sequence[str] = [
            "pi0_actions_ref", "pi0_state", "qpos", "last_action"
        ],
        # 使用 tanh 压缩具有显著离群点的特征 tanh(v/s), 键值为 s
        vec_obs_compress_key: dict[str, float] = {
            "qvel": 1.5 
        },
        # MSHAB 数据不会主动剪切动作到 [-1, 1]
        clip_action: bool = True,
    ):
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
        except:
            raise ImportError("For dataset loading, install [train] optional group")

        self.ds = LeRobotDataset(repo_id, root = root)
        self.sample_to_obs = Sample2ObsConfig(
            camera_info_list = camera_info_list, 
            vec_obs_keys = vec_obs_keys,
            vec_obs_compress_key = vec_obs_compress_key
        )

        self.clip_action = clip_action

        if root is None:
            root = HF_LEROBOT_HOME
        if isinstance(root, str):
            root = Path(root)
        self.dataset_dir_path = root / repo_id

        self._h5 = None

    def __len__(self):
        return len(self.ds)
    
    def _get_h5(self):
        try:
            import h5py
        except:
            raise ImportError("For dataset loading, install [train] optional group")

        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_dir_path / "detach.h5", "r", swmr=True)
        return self._h5

    def _load_detach_sample(self, sample):
        h5 = self._get_h5()

        # 从 h5 中读取数据整合到 sample 中
        episode_idx = int(sample["episode_idx"].item())
        frame_idx = int(sample["frame_idx"].item())

        episode_group: h5py.Group = h5[f"episode_{episode_idx:06d}"] # type: ignore
        for key, dataset in episode_group.items():
            # 读取未处理的原始数据
            sample[key] = torch.from_numpy(dataset[frame_idx])
        return sample

    def __getitem__(self, idx):

        sample = self.ds[idx]
        sample = self._load_detach_sample(sample)    

        obs = self.sample_to_obs.get_obs(sample)
        # 策略真实动作
        action = sample[ACTION_SAMPLE_KEY]
        if self.clip_action:
            action = torch.clip(action, -1, 1)

        return obs, action

    def get_sample_size(self):
        '''
        debug, 主动从采样中获取一个 sample 的特征维度
        '''
        obs, action = self[0]

        output_dim = torch.as_tensor(action).size(0)
        vec_obs_dim = torch.as_tensor(obs[VECTION_OBS_KEY]).size(0)
        img_obs_dim = {}
        for key, img in obs.items():
            if key == VECTION_OBS_KEY:
                continue
            img_obs_dim[key] = torch.as_tensor(img).size(0)
        
        return output_dim, vec_obs_dim, img_obs_dim
