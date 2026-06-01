from typing import Any

import torch
import optuna

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list
from vla_bc_policy.model.lit_model import PolicyModule
from vla_bc_policy.train import get_trainer

def sample_param(
    trial: optuna.Trial
):
    param = {}

    vector_only = trial.suggest_categorical("vector_only", [True, False])
    if vector_only:
        param["extractor_type"] = "vector_only"
        param["extractor_kwargs"] = {}
    else:
        param["extractor_type"] = "post_mix"

        encoder_type = trial.suggest_categorical("encoder_type", ["impala", "resnet"])
        param["extractor_kwargs"] = dict(
            image_backbone_cfg = {
                "head": (
                    encoder_type, {}, 64
                ),
                "gripper": (
                    encoder_type, {}, 64
                ),
            },
        )

def make_model(
    param: dict[str, Any]
):
    pass