from vla_bc_policy.encoder.convnext import ConvNeXtEncoder
from vla_bc_policy.encoder.efficient_net import EfficientNetEncoder
from vla_bc_policy.encoder.impala import ImpalaEncoder
from vla_bc_policy.encoder.resnet import ResNetEncoder

from typing import Literal
EncoderType = Literal["convnext", "efficient_net", "impala", "resnet"]
EncoderDict = {
    "convnext": ConvNeXtEncoder,
    "efficient_net": EfficientNetEncoder,
    "impala": ImpalaEncoder,
    "resnet": ResNetEncoder
}

def get_encoder(encoder: EncoderType):
    return EncoderDict[encoder]
