"""CNN compacta treinada do zero para o BloodMNIST."""

from __future__ import annotations

from torch import nn

from ..config import DATA_CONFIG


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Extrai características locais e reduz a resolução espacial pela metade."""

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def build() -> nn.Module:
    """Cria uma CNN compacta, sem pesos pré-treinados, para oito classes."""

    return nn.Sequential(
        _conv_block(DATA_CONFIG.channels, 32),
        _conv_block(32, 64),
        _conv_block(64, 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Dropout(p=0.3),
        nn.Linear(128, DATA_CONFIG.num_classes),
    )
