import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ViT_B_16_Weights,
    EfficientNet_V2_S_Weights,
)


def get_convnext_model():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)

    return model


def get_vit_model():
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)

    return model


def get_efficientnetv2s_model():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    return model