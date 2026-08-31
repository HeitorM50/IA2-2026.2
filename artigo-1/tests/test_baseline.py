from __future__ import annotations

import torch
from torch import nn

from src.config import DATA_CONFIG
from src.models.baseline import build


def test_baseline_is_exactly_multinomial_logistic_regression() -> None:
    model = build()
    layers = list(model.children())

    assert [type(layer) for layer in layers] == [nn.Flatten, nn.Linear]
    assert layers[0].start_dim == 1
    assert layers[1].in_features == 3 * 64 * 64
    assert layers[1].out_features == DATA_CONFIG.num_classes
    assert sum(parameter.numel() for parameter in model.parameters()) == 98_312
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_baseline_produces_one_logit_per_class() -> None:
    images = torch.zeros(
        4,
        DATA_CONFIG.channels,
        DATA_CONFIG.dataset_size,
        DATA_CONFIG.dataset_size,
    )

    logits = build()(images)

    assert logits.shape == (4, DATA_CONFIG.num_classes)
    assert torch.isfinite(logits).all()
