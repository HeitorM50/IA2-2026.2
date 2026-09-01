from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.run import _toy_experiment
from src.train import _build_optimizer, _synchronize_device, train_eval


class CountingLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterations = 0
        self.batch_size = loader.batch_size

    def __iter__(self) -> Iterator:
        self.iterations += 1
        return iter(self.loader)


def _assert_no_null_or_non_finite(value) -> None:
    assert value is not None
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            _assert_no_null_or_non_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_null_or_non_finite(child)


def test_linear_model_passes_through_train_eval_and_tests_once(capsys) -> None:
    model, (train_loader, val_loader, raw_test_loader) = _toy_experiment(42)
    test_loader = CountingLoader(raw_test_loader)
    config = TrainingConfig(
        learning_rate=1e-30,
        head_learning_rate=1e-30,
        weight_decay=0,
        max_epochs=8,
        patience=2,
        min_delta=1.0,
    )

    result = train_eval(
        model,
        (train_loader, val_loader, test_loader),  # type: ignore[arg-type]
        42,
        model_name="toy",
        config=config,
    )

    assert result["schema_version"] == 1
    assert result["model"] == "toy"
    assert result["seed"] == 42
    assert result["training"]["epochs_ran"] == 3
    assert result["training"]["best_epoch"] == 1
    assert result["parameters"] == {"total": 72, "trainable": 72}
    assert test_loader.iterations == 1
    assert len(result["test"]["confusion_matrix"]) == 8
    assert "toy seed=42 época=1/8" in capsys.readouterr().out
    _assert_no_null_or_non_finite(result)


def test_optimizer_uses_separate_learning_rate_for_head() -> None:
    class ModelWithHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.fc = nn.Linear(4, 8)

    config = TrainingConfig(
        learning_rate=1e-4,
        head_learning_rate=1e-3,
        head_parameter_prefix="fc.",
    )
    optimizer = _build_optimizer(ModelWithHead(), config)

    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-3]


def test_cuda_synchronization_only_runs_for_cuda(monkeypatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))

    _synchronize_device(torch.device("cpu"))
    _synchronize_device(torch.device("cuda"))

    assert calls == [torch.device("cuda")]


def test_training_rejects_model_without_trainable_parameters() -> None:
    class EmptyModel(nn.Module):
        def forward(self, images):
            return images

    with pytest.raises(ValueError, match="parâmetros treináveis"):
        _build_optimizer(EmptyModel(), TrainingConfig())
