from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import nn

from vla_bc_policy.encoder import get_encoder, EncoderType
from vla_bc_policy.dataset.pi0_lerobot_datamodule import get_random_pi0_lerobot_batch
from vla_bc_policy.dataset.utility import VECTION_OBS_KEY

from vla_bc_policy.model.res_mlp_decoder import ResMlpDecoder
from vla_bc_policy.model.utility import NormType, ActivateFnType, get_model_output_shape
from vla_bc_policy.model.mlp_decoder import MlpDecoder

class MulViewPostMixedExtractor(nn.Module):
    def __init__(
        self,
 
        vec_obs_dim: int, 
        img_obs_dim: Dict[str, int],

        image_backbone_cfg: Dict[str, Tuple[
            EncoderType, # 特征提取网络类 
            Dict, # 特征提取网络参数 
            int, # 调整特征数
        ]],

        # 一维特征调整数
        vector_out_feat: int = 128,
        # 一维特征提取器类型
        vector_encoder_type: Literal["mlp", "res_mlp"] = "mlp",
        # 一维特征调整网络的额外 MLP 隐藏层维度
        vector_encoder_kwargs: dict[str, Any] = {},

        # 图像全连接层参数
        dropout_rate: Optional[float] = None,
        norm_type: NormType = "layer",
        activate_fn_type: ActivateFnType = "gelu",
    ):
        super().__init__()

        self.example_input_array, _ = get_random_pi0_lerobot_batch(
            None, vec_obs_dim, img_obs_dim, 1
        )
        self.out_feat = 0

        self.image_backbone = nn.ModuleDict()
        for key, val in image_backbone_cfg.items():
            encoder_type = get_encoder(val[0])
            encoder_in_channels = img_obs_dim[key]
            # 传入实际图像通道参数
            encoder_cfg = dict(val[1])
            encoder_cfg["in_channels"] = encoder_in_channels

            encoder_model = encoder_type(**encoder_cfg)
            encoder_output_shape = get_model_output_shape(
                encoder_model, self.example_input_array[key]
            )
            # 使用 MLP 调整特征通道, 防止特征比重过大
            shrink_model = MlpDecoder(
                encoder_output_shape[0], val[2], 
                is_pure_output = False,
                dropout_rate = dropout_rate,
                norm_type = norm_type,
                activate_fn_type = activate_fn_type
            )
            # 
            self.image_backbone[key] = nn.Sequential(
                encoder_model, shrink_model
            )

            self.out_feat += val[2]

        vector_encoder_kwargs = dict(vector_encoder_kwargs)
        vector_encoder_kwargs["num_in_feats"] = vec_obs_dim
        vector_encoder_kwargs["num_out_feats"] = vector_out_feat
        if vector_encoder_type == "mlp":
            self.vector_backbone = MlpDecoder(**vector_encoder_kwargs)
        elif vector_encoder_type == "res_mlp":
            self.vector_backbone = ResMlpDecoder(**vector_encoder_kwargs)
        else: 
            raise RuntimeError(f"Unknown vector_encoder_type: {vector_encoder_type}")
        self.out_feat += vector_out_feat

    def forward(self, X: Dict[str, torch.Tensor]):

        feat_vec = [self.vector_backbone(X[VECTION_OBS_KEY])]

        for key, val in self.image_backbone.items():
            feat_vec.append(val(X[key]))

        return torch.cat(feat_vec, dim=1)

    def get_out_feat(self):
        return self.out_feat

class VectorOnlyExtractor(nn.Module):
    def __init__(
        self,
 
        vec_obs_dim: int, 

        # 一维特征调整数
        vector_modified_feat: int = 128,
        # 一维特征调整网络的额外 MLP 隐藏层维度
        vector_mlp_layers: Optional[list[int]] = None,

        # 全连接层参数
        dropout_rate: Optional[float] = None,
        norm_type: NormType = "layer",
        activate_fn_type: ActivateFnType = "silu",
    ):
        super().__init__()

        self.out_feat = vector_modified_feat

        self.vector_backbone = MlpDecoder(
            vec_obs_dim,
            vector_modified_feat,
            mlp_layers = vector_mlp_layers,

            is_pure_output = False,
            dropout_rate = dropout_rate,
            norm_type = norm_type,
            activate_fn_type = activate_fn_type
        )

    def forward(self, X: Dict[str, torch.Tensor]):
        return self.vector_backbone(X[VECTION_OBS_KEY])

    def get_out_feat(self):
        return self.out_feat

ExtractorType = Literal["post_mix", "vector_only"]
ExtractorDict = {
    "post_mix": MulViewPostMixedExtractor,
    "vector_only": VectorOnlyExtractor
}

if __name__ == "__main__":
    from vla_bc_policy.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
    from vla_bc_policy.dataset.camera_info import FetchStandardCameraInfos, camera_info_list_to_dict_list

    dm = Pi0LeRobotDataModule(
        "trajectories_tidy_house_all_bc_state_rot_6d_action_axis_angle",
        camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False
    )
    model = MulViewPostMixedExtractor(
        dm.vec_obs_dim, dm.img_obs_dim,
        image_backbone_cfg = {
            "head": (
                "resnet", {}, 64
            ),
            "gripper": (
                "resnet", {}, 64
            ),
        },
        vector_out_feat = 128,
        vector_mlp_layers = [256],
        dropout_rate = 0.05
    )

    random_input, _ = dm.get_random_batch(1)
    output = model(random_input)
    print(f"output shape: {output.shape}")