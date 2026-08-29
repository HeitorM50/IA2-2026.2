"""Métricas comuns da comparação entre os três classificadores."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def _as_vector(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} deve ser um vetor; recebido formato {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} não pode estar vazio.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} deve conter rótulos inteiros.")
    return array.astype(np.int64, copy=False)


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_classes: int = 8,
) -> dict[str, Any]:
    """Calcula as métricas do artigo a partir de alvos, predições e probabilidades."""

    if num_classes <= 1:
        raise ValueError("num_classes deve ser maior que um.")

    true = _as_vector(y_true, "y_true")
    pred = _as_vector(y_pred, "y_pred")
    score = np.asarray(y_score, dtype=np.float64)

    if true.shape != pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo número de exemplos.")
    if score.shape != (true.size, num_classes):
        raise ValueError(
            "y_score deve ter formato "
            f"({true.size}, {num_classes}); recebido {score.shape}."
        )
    if not np.isfinite(score).all():
        raise ValueError("y_score contém valor não finito.")
    if (score < 0).any() or (score > 1).any():
        raise ValueError("y_score deve conter probabilidades no intervalo [0, 1].")
    if not np.allclose(score.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("Cada linha de y_score deve somar um.")

    valid_labels = np.arange(num_classes)
    if not np.isin(true, valid_labels).all() or not np.isin(pred, valid_labels).all():
        raise ValueError(f"Os rótulos devem estar no intervalo [0, {num_classes}).")
    if not np.array_equal(score.argmax(axis=1), pred):
        raise ValueError("y_pred deve corresponder ao argmax de y_score.")

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "f1_macro": float(
            f1_score(
                true,
                pred,
                labels=valid_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "confusion_matrix": confusion_matrix(
            true, pred, labels=valid_labels
        ).astype(int).tolist(),
    }
