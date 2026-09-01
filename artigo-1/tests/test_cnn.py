from __future__ import annotations

import torch
from torch import nn

from src.config import DATA_CONFIG
from src.models.cnn import build


def test_cnn_has_three_convolutional_blocks_and_regularized_head() -> None:
    model = build()
    layers = list(model.children())

    assert [type(layer) for layer in layers] == [
        nn.Sequential,
        nn.Sequential,
        nn.Sequential,
        nn.AdaptiveAvgPool2d,
        nn.Flatten,
        nn.Dropout,
        nn.Linear,
    ]

    expected_channels = ((3, 32), (32, 64), (64, 128))
    for block, (in_channels, out_channels) in zip(
        layers[:3], expected_channels, strict=True
    ):
        operations = list(block.children())
        assert [type(operation) for operation in operations] == [
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.MaxPool2d,
        ]
        first_conv, _, _, second_conv, _, _, pool = operations
        assert (first_conv.in_channels, first_conv.out_channels) == (
            in_channels,
            out_channels,
        )
        assert (second_conv.in_channels, second_conv.out_channels) == (
            out_channels,
            out_channels,
        )
        assert first_conv.kernel_size == second_conv.kernel_size == (3, 3)
        assert first_conv.padding == second_conv.padding == (1, 1)
        assert first_conv.bias is second_conv.bias is None
        assert pool.kernel_size == pool.stride == 2

    assert layers[3].output_size == (1, 1)
    assert layers[4].start_dim == 1
    assert layers[5].p == 0.3
    assert layers[6].in_features == 128
    assert layers[6].out_features == DATA_CONFIG.num_classes
    assert not any(isinstance(layer, nn.Softmax) for layer in model.modules())


def test_cnn_produces_one_finite_logit_per_class() -> None:
    images = torch.zeros(
        4,
        DATA_CONFIG.channels,
        DATA_CONFIG.dataset_size,
        DATA_CONFIG.dataset_size,
    )

    logits = build()(images)

    assert logits.shape == (4, DATA_CONFIG.num_classes)
    assert torch.isfinite(logits).all()


def test_cnn_is_compact_and_fully_trainable() -> None:
    model = build()

    assert sum(parameter.numel() for parameter in model.parameters()) == 288_488
    assert all(parameter.requires_grad for parameter in model.parameters())
