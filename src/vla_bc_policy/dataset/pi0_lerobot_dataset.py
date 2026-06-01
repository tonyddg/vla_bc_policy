from pathlib import Path
from typing import Optional, Sequence, Union

import h5py

import torch
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

from vla_bc_policy.dataset.camera_info import CameraInfo
from vla_bc_policy.dataset.utility import ACTION_SAMPLE_KEY, VECTION_OBS_KEY

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
    ):
        self.ds = LeRobotDataset(repo_id, root = root)
        self.vec_obs_keys = vec_obs_keys
        self.vec_obs_compress_key = vec_obs_compress_key
        self.camera_info_list = camera_info_list

        if root is None:
            root = HF_LEROBOT_HOME
        if isinstance(root, str):
            root = Path(root)
        self.dataset_dir_path = root / repo_id

        self._h5 = None

    def __len__(self):
        return len(self.ds)
    
    def _get_h5(self):
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

    def _get_image_obs(self, sample):
        image_obs = {}
        for camera_info in self.camera_info_list:
            # obs[camera_info.camera_name] = camera_info.sample2img(sample)
            camera_img = camera_info.sample2img(sample)
            image_obs[camera_info.camera_name] = camera_img
        return image_obs
    
    def _get_1d_vector_obs(self, sample):
        # 合并一维特征
        state_list = []
        for vec_obs_key in self.vec_obs_keys:
            state = sample.get(vec_obs_key)
            assert state is not None, f"{vec_obs_key} may not in sample with {list(sample.keys())}"

            state = torch.as_tensor(state).float().flatten()
            if vec_obs_key in self.vec_obs_compress_key:
                state = torch.tanh(state / self.vec_obs_compress_key[vec_obs_key])

            state_list.append(state)
        return torch.cat(state_list)

    def _get_dict_vector_obs(self, sample):
        # 以字典形式组织一维特征
        vec_obs = {}
        for vec_obs_key in self.vec_obs_keys:
            state = sample.get(vec_obs_key)
            assert state is not None, f"{vec_obs_key} may not in sample with {list(sample.keys())}"

            state = torch.as_tensor(state).float().flatten()
            vec_obs[vec_obs_key] = state
        return vec_obs

    def __getitem__(self, idx):

        sample = self.ds[idx]
        sample = self._load_detach_sample(sample)        
        obs = self._get_image_obs(sample)
        vec_obs = self._get_1d_vector_obs(sample)

        # 整合一维特征与图像
        obs[VECTION_OBS_KEY] = vec_obs
        # 策略真实动作
        action = sample[ACTION_SAMPLE_KEY]

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
