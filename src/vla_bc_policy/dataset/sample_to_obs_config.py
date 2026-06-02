from typing import Sequence, Mapping

import torch

from vla_bc_policy.dataset.camera_info import CameraInfo
from vla_bc_policy.dataset.utility import VECTION_OBS_KEY

SampleType = Mapping[str, torch.Tensor]

class Sample2ObsConfig:
    def __init__(
        self,
        camera_info_list: list[CameraInfo],
        vec_obs_keys: Sequence[str], 
        vec_obs_compress_key: dict[str, float],

        is_concat_vec_obs: bool = True
    ) -> None:
        self.camera_info_list = camera_info_list
        self.vec_obs_keys = vec_obs_keys
        self.vec_obs_compress_key = vec_obs_compress_key

        self.is_concat_vec_obs = is_concat_vec_obs

    def _get_image_obs(
        self,
        sample: SampleType,
    ):
        image_obs = {}
        for camera_info in self.camera_info_list:
            # obs[camera_info.camera_name] = camera_info.sample2img(sample)
            camera_img = camera_info.sample2img(sample)
            image_obs[camera_info.camera_name] = camera_img
        return image_obs

    def _get_1d_vector_obs(
        self,
        sample: SampleType,
    ):
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

    def _get_dict_vector_obs(
        self, 
        sample: SampleType,
    ):
        # 以字典形式组织一维特征
        vec_obs = {}
        for vec_obs_key in self.vec_obs_keys:
            state = sample.get(vec_obs_key)
            assert state is not None, f"{vec_obs_key} may not in sample with {list(sample.keys())}"

            state = torch.as_tensor(state).float().flatten()
            vec_obs[vec_obs_key] = state
        return vec_obs

    def get_obs(
        self, 
        sample: SampleType,
    ):
        obs = self._get_image_obs(sample)

        if self.is_concat_vec_obs:
            vec_obs = self._get_1d_vector_obs(sample)
        else:
            vec_obs = self._get_dict_vector_obs(sample)

        # 整合一维特征与图像
        obs[VECTION_OBS_KEY] = vec_obs
        return obs
