"""ResNet18 pré-treinada no ImageNet-1K para transferência de aprendizado."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ..config import DATA_CONFIG


def build() -> nn.Module:
    """Cria uma ResNet18 ImageNet-1K com ajuste fino integral para oito classes."""

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, DATA_CONFIG.num_classes)
    model.requires_grad_(True)
    return model
