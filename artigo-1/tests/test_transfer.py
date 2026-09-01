from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from src.config import DATA_CONFIG, TRAINING_CONFIGS
from src.models import transfer


def _build_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[nn.Module, list[ResNet18_Weights | None]]:
    original_builder = transfer.resnet18
    received_weights: list[ResNet18_Weights | None] = []

    def build_uninitialized(*, weights: ResNet18_Weights | None) -> nn.Module:
        received_weights.append(weights)
        return original_builder(weights=None)

    monkeypatch.setattr(transfer, "resnet18", build_uninitialized)
    return transfer.build(), received_weights


def test_transfer_uses_exact_imagenet_weights_and_replaces_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, received_weights = _build_without_download(monkeypatch)

    assert received_weights == [ResNet18_Weights.IMAGENET1K_V1]
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == DATA_CONFIG.num_classes


def test_transfer_is_fully_trainable_with_expected_parameter_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _build_without_download(monkeypatch)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert total_parameters == 11_180_616
    assert trainable_parameters == total_parameters


def test_transfer_produces_one_finite_logit_per_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _build_without_download(monkeypatch)
    images = torch.zeros(
        2,
        DATA_CONFIG.channels,
        DATA_CONFIG.dataset_size,
        DATA_CONFIG.dataset_size,
    )

    model.eval()
    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (2, DATA_CONFIG.num_classes)
    assert torch.isfinite(logits).all()


def test_transfer_has_separate_backbone_and_head_learning_rates() -> None:
    config = TRAINING_CONFIGS["resnet18"]

    assert config.learning_rate == 1e-4
    assert config.head_learning_rate == 1e-3
    assert config.head_parameter_prefix == "fc."
    assert config.weight_decay == 1e-4
