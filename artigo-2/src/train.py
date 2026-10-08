"""Treino, calibração e avaliação comuns aos detectores do Artigo 2.

O conjunto de teste aparece em um único ponto deste módulo: depois do ajuste,
da restauração da melhor época e da calibração sobre validação normal.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import (
    DATA_CONFIG,
    DETECTION_CONFIG,
    FAULT_NAMES,
    TRAINING_CONFIG,
    DetectionConfig,
    TrainingConfig,
    set_seed,
)
from src.faults import EvaluationSet
from src.metrics import detection_metrics
from src.models.base import Detector


def _as_windows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(
            f"{name} deve ter formato (janelas, amostras, canais); "
            f"recebido {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contém valor não finito.")
    return array


def _as_binary_labels(
    values: Sequence[int] | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size != size:
        raise ValueError(f"{name} deve conter um rótulo por janela.")
    if not (
        np.issubdtype(labels.dtype, np.integer)
        or np.issubdtype(labels.dtype, np.bool_)
    ):
        raise TypeError(f"{name} deve conter rótulos binários inteiros.")
    labels = labels.astype(np.int8, copy=False)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} deve conter somente 0 e 1.")
    return labels


def _validate_evaluation_set(evaluation: EvaluationSet, name: str) -> np.ndarray:
    X = _as_windows(evaluation.X, f"{name}.X")
    n, samples, _ = X.shape
    expected_vectors = {
        "y": evaluation.y,
        "fault": evaluation.fault,
        "start": evaluation.start,
    }
    for field, values in expected_vectors.items():
        if np.asarray(values).shape != (n,):
            raise ValueError(f"{name}.{field} deve ter formato ({n},).")
    if np.asarray(evaluation.mask).shape != (n, samples):
        raise ValueError(
            f"{name}.mask deve ter formato ({n}, {samples})."
        )
    return _as_binary_labels(evaluation.y, n, f"{name}.y")


def calibrate_threshold(
    window_scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    sigma_multiplier: float,
) -> dict[str, Any]:
    """Calcula ``μ + λσ`` exclusivamente sobre validação normal."""

    scores = np.asarray(window_scores, dtype=np.float64)
    y = np.asarray(labels)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("window_scores deve ser um vetor não vazio.")
    if y.ndim != 1 or y.shape != scores.shape:
        raise ValueError("labels deve ter o mesmo formato de window_scores.")
    if not np.isfinite(scores).all():
        raise ValueError("window_scores contém valor não finito.")
    if not (
        np.issubdtype(y.dtype, np.integer)
        or np.issubdtype(y.dtype, np.bool_)
    ) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels deve conter somente 0 e 1.")
    if isinstance(sigma_multiplier, bool) or not np.isfinite(sigma_multiplier):
        raise ValueError("sigma_multiplier deve ser finito e positivo.")
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier deve ser finito e positivo.")

    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        raise ValueError("A calibração exige ao menos uma janela normal.")

    mean = float(normal_scores.mean())
    std = float(normal_scores.std(ddof=0))
    multiplier = float(sigma_multiplier)
    threshold = mean + multiplier * std
    if not np.isfinite(threshold):
        raise ValueError("A calibração produziu um limiar não finito.")
    return {
        "source": "validation_normal",
        "normal_windows": int(normal_scores.size),
        "score_mean": mean,
        "score_std": std,
        "sigma_multiplier": multiplier,
        "threshold": float(threshold),
    }


def _batch_limit_reached(batch_index: int, limit: int) -> bool:
    return limit > 0 and batch_index >= limit


def _fit_torch_mse(
    model: Detector,
    train_X: np.ndarray,
    validation_X: np.ndarray,
    seed: int,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Ajusta um ``nn.Module`` por reconstrução, importando PyTorch sob demanda."""

    try:
        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "O treino MSE requer PyTorch; instale a dependência declarada."
        ) from error

    if not isinstance(model, nn.Module):
        raise TypeError(
            "Detector com fit_mode='torch_mse' também deve herdar de nn.Module."
        )

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("O autoencoder não possui parâmetros treináveis.")

    optimizer = AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_X)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    def epoch_loss(loader: Any, *, training: bool, max_batches: int) -> float:
        model.train(training)
        loss_sum = 0.0
        example_count = 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for batch_index, (batch,) in enumerate(loader):
                if _batch_limit_reached(batch_index, max_batches):
                    break
                batch = batch.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                if reconstruction.shape != batch.shape:
                    raise RuntimeError(
                        "O autoencoder deve reconstruir o formato de entrada; "
                        f"entrada={tuple(batch.shape)}, "
                        f"saída={tuple(reconstruction.shape)}."
                    )
                loss = criterion(reconstruction, batch)
                if training:
                    loss.backward()
                    optimizer.step()
                batch_size = batch.shape[0]
                loss_sum += float(loss.detach().item()) * batch_size
                example_count += batch_size
        if example_count == 0:
            split = "treino" if training else "validação"
            raise RuntimeError(f"O loader de {split} não produziu exemplos.")
        return loss_sum / example_count

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = epoch_loss(
            train_loader,
            training=True,
            max_batches=config.max_train_batches,
        )
        validation_loss = epoch_loss(
            validation_loader,
            training=False,
            max_batches=config.max_validation_batches,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"{model.name} seed={seed} época={epoch}/{config.max_epochs} "
            f"train_loss={train_loss:.6f} val_loss={validation_loss:.6f}",
            flush=True,
        )

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("O treino terminou sem uma época válida.")
    model.load_state_dict(best_state)
    model._mark_fitted()

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    return (
        {
            "method": "gradient_mse",
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        {
            "torch_version": torch.__version__,
            "device_type": device.type,
            "device_name": device_name,
            "cuda_version": torch.version.cuda or "not_available",
        },
    )


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


def _environment_metadata(torch_metadata: dict[str, str]) -> dict[str, Any]:
    commit, dirty = _git_metadata()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch_metadata.get("torch_version", "not_used"),
        "device_type": torch_metadata.get("device_type", "cpu"),
        "device_name": torch_metadata.get("device_name", "CPU"),
        "cuda_version": torch_metadata.get("cuda_version", "not_available"),
        "platform": sys.platform,
    }


def _evaluate_once(
    model: Detector,
    evaluation: EvaluationSet,
    threshold: float,
    fault_names: Sequence[str],
) -> dict[str, Any]:
    """Pontua todo o split em uma chamada e reutiliza os escores nas métricas."""

    sample_scores = np.asarray(model.score_samples(evaluation.X), dtype=np.float64)
    window_scores = model.aggregate_scores(sample_scores)
    return detection_metrics(
        evaluation.y,
        window_scores,
        sample_scores,
        threshold,
        evaluation.fault,
        evaluation.mask,
        fault_names=fault_names,
    )


def train_eval(
    model: Detector,
    train_X: np.ndarray,
    train_y: Sequence[int] | np.ndarray,
    validation: EvaluationSet,
    test: EvaluationSet,
    seed: int,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    detection_config: DetectionConfig = DETECTION_CONFIG,
    fault_names: Sequence[str] = FAULT_NAMES,
) -> dict[str, Any]:
    """Ajusta, calibra e avalia um detector sem consultar o teste antes da hora."""

    X_train = _as_windows(train_X, "train_X")
    y_train = _as_binary_labels(train_y, len(X_train), "train_y")
    validation_y = _validate_evaluation_set(validation, "validation")
    _validate_evaluation_set(test, "test")

    # Invariante metodológica explícita: qualquer anomalia interrompe a execução
    # antes que o modelo tenha oportunidade de vê-la.
    assert np.all(y_train == 0), "Nenhum modelo pode ver anomalias no treino."

    validation_normal = validation_y == 0
    if not validation_normal.any():
        raise ValueError("A validação precisa conter janelas normais para calibração.")
    X_validation_normal = np.asarray(
        validation.X[validation_normal], dtype=np.float32
    )

    set_seed(seed)
    started_at = time.perf_counter()
    torch_metadata: dict[str, str] = {}
    if model.fit_mode == "closed_form":
        model.fit(X_train)
        training: dict[str, Any] = {
            "method": "closed_form",
            "epochs_ran": 0,
            "history": [],
        }
    elif model.fit_mode == "torch_mse":
        training, torch_metadata = _fit_torch_mse(
            model,
            X_train,
            X_validation_normal,
            seed,
            training_config,
        )
    else:
        raise ValueError(f"fit_mode desconhecido: {model.fit_mode!r}.")
    training["elapsed_seconds"] = time.perf_counter() - started_at

    # A calibração ocorre antes de qualquer acesso ao teste e usa somente as
    # linhas normais da validação, mesmo que o EvaluationSet contenha falhas.
    validation_normal_samples = model.score_samples(X_validation_normal)
    validation_normal_scores = model.aggregate_scores(validation_normal_samples)
    calibration = calibrate_threshold(
        validation_normal_scores,
        np.zeros(len(validation_normal_scores), dtype=np.int8),
        detection_config.calibration_sigmas,
    )
    threshold = float(calibration["threshold"])

    validation_metrics = _evaluate_once(
        model, validation, threshold, fault_names
    )
    # Único acesso ao split de teste em todo o protocolo.
    test_metrics = _evaluate_once(model, test, threshold, fault_names)

    total_parameters = model.n_parameters()
    if model.fit_mode == "torch_mse":
        trainable_parameters = sum(
            int(parameter.numel())
            for parameter in model.parameters()  # type: ignore[attr-defined]
            if parameter.requires_grad
        )
    else:
        trainable_parameters = 0

    return {
        "schema_version": 1,
        "model": model.name,
        "seed": seed,
        "dataset": {
            "name": DATA_CONFIG.dataset_name,
            "window_shape": list(X_train.shape[1:]),
            "split_sizes": {
                "train": len(X_train),
                "validation": len(validation.X),
                "test": len(test.X),
            },
        },
        "hyperparameters": {
            "training": training_config.as_dict(),
            "detection": detection_config.as_dict(),
            "loss": "MSELoss" if model.fit_mode == "torch_mse" else "not_applicable",
        },
        "training": training,
        "calibration": calibration,
        "validation": validation_metrics,
        "test": test_metrics,
        "parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
        },
        "environment": _environment_metadata(torch_metadata),
    }
