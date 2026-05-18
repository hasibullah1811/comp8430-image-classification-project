"""
MobileNetV3 fine-tuning utilities for fine-grained classification.
"""

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights


def create_mobilenet(
    num_classes: int = 20,
    pretrained: bool = True,
    freeze_features: bool = True,
    model_size: Literal["small", "large"] = "small",
) -> nn.Module:
    """
    Load a pretrained MobileNetV3 and adapt the final layer for our class count.

    The early convolutional blocks act as a generic feature extractor trained on
    ImageNet. We freeze those weights and only train the classifier head, which is
  a standard transfer-learning strategy for small, specialised datasets.
    """
    if model_size == "small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
    elif model_size == "large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
    else:
        raise ValueError("model_size must be 'small' or 'large'")

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace only the final linear layer; keep the preceding dropout / activation blocks.
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
