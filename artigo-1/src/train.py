"""Loop único de treinamento, validação e teste do Artigo 1."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .config import DATA_CONFIG, TRAINING_CONFIGS, TrainingConfig
from .data import set_seed
from .metrics import classification_metrics

Loaders = tuple[DataLoader, DataLoader, DataLoader]


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("O modelo não possui parâmetros treináveis.")

    if not config.head_parameter_prefix:
        return AdamW(
            (parameter for _, parameter in named_parameters),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    backbone = [
        parameter
        for name, parameter in named_parameters
        if not name.startswith(config.head_parameter_prefix)
    ]
    head = [
        parameter
        for name, parameter in named_parameters
        if name.startswith(config.head_parameter_prefix)
    ]
    if not backbone or not head:
        raise ValueError(
            "O prefixo da cabeça não separou parâmetros de backbone e cabeça: "
            f"{config.head_parameter_prefix!r}."
        )
    return AdamW(
        [
            {"params": backbone, "lr": config.learning_rate},
            {"params": head, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def _batch_limit_reached(batch_index: int, limit: int) -> bool:
    return limit > 0 and batch_index >= limit


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int,
) -> float:
    model.train()
    loss_sum = 0.0
    example_count = 0

    for batch_index, (images, targets) in enumerate(loader):
        if _batch_limit_reached(batch_index, max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != DATA_CONFIG.num_classes:
            raise RuntimeError(
                "O modelo deve produzir logits com formato "
                f"(N, {DATA_CONFIG.num_classes}); recebido {tuple(logits.shape)}."
            )
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        loss_sum += float(loss.detach().item()) * batch_size
        example_count += batch_size

    if example_count == 0:
        raise RuntimeError("O loader de treino não produziu nenhum exemplo.")
    return loss_sum / example_count


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    true_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []

    for batch_index, (images, targets) in enumerate(loader):
        if _batch_limit_reached(batch_index, max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != DATA_CONFIG.num_classes:
            raise RuntimeError(
                "O modelo deve produzir logits com formato "
                f"(N, {DATA_CONFIG.num_classes}); recebido {tuple(logits.shape)}."
            )

        loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        batch_size = targets.shape[0]

        loss_sum += float(loss.item()) * batch_size
        example_count += batch_size
        true_batches.append(targets.cpu().numpy())
        predicted_batches.append(predictions.cpu().numpy())
        score_batches.append(probabilities.cpu().numpy())

    if example_count == 0:
        raise RuntimeError("O loader de avaliação não produziu nenhum exemplo.")

    metrics = classification_metrics(
        np.concatenate(true_batches),
        np.concatenate(predicted_batches),
        np.concatenate(score_batches),
        num_classes=DATA_CONFIG.num_classes,
    )
    return loss_sum / example_count, metrics


def _git_metadata() -> tuple[str, bool]:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unavailable", False


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    commit, dirty = _git_metadata()
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device_type": device.type,
        "device_name": device_name,
        "cuda_version": torch.version.cuda or "not_available",
        "platform": sys.platform,
    }


def train_eval(
    model: nn.Module,
    loaders: Loaders,
    seed: int,
    *,
    model_name: str = "toy",
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Treina um modelo, escolhe a melhor validação e avalia o teste uma vez."""

    if len(loaders) != 3:
        raise ValueError("loaders deve conter treino, validação e teste.")
    if config is None:
        try:
            config = TRAINING_CONFIGS[model_name]
        except KeyError as error:
            raise ValueError(f"Configuração desconhecida: {model_name!r}.") from error

    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    device = _resolve_device()
    model = model.to(device)
    parameters = _parameter_counts(model)
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_metrics: dict[str, Any] | None = None
    best_score = float("-inf")
    epochs_without_improvement = 0

    _synchronize_device(device)
    started_at = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.max_train_batches,
        )
        validation_loss, validation_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            config.max_eval_batches,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1_macro": validation_metrics["f1_macro"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
            }
        )

        score = validation_metrics["f1_macro"]
        if score > best_score + config.min_delta:
            best_score = score
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_validation_metrics = validation_metrics
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    _synchronize_device(device)
    elapsed_seconds = time.perf_counter() - started_at

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("O treinamento terminou sem uma época válida.")
    model.load_state_dict(best_state)
    test_loss, test_metrics = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        config.max_eval_batches,
    )

    return {
        "schema_version": 1,
        "model": model_name,
        "seed": seed,
        "dataset": {
            "name": DATA_CONFIG.dataset_flag,
            "image_size": DATA_CONFIG.dataset_size,
            "num_classes": DATA_CONFIG.num_classes,
            "split_sizes": DATA_CONFIG.expected_split_sizes,
        },
        "hyperparameters": {
            **config.as_dict(),
            "loss": "CrossEntropyLoss",
            "batch_size": int(train_loader.batch_size or 0),
        },
        "training": {
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_validation_f1_macro": best_score,
            "elapsed_seconds": elapsed_seconds,
            "history": history,
        },
        "validation": {
            "loss": best_validation_loss,
            **best_validation_metrics,
        },
        "test": {"loss": test_loss, **test_metrics},
        "parameters": parameters,
        "environment": _environment_metadata(device),
    }
