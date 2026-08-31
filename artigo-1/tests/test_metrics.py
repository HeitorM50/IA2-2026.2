from __future__ import annotations

import numpy as np
import pytest

from src.metrics import classification_metrics


def test_perfect_predictions_return_perfect_metrics_and_fixed_matrix() -> None:
    y_true = np.arange(8, dtype=np.int64)
    y_score = np.eye(8, dtype=np.float64)

    metrics = classification_metrics(y_true, y_true.copy(), y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == np.eye(8, dtype=int).tolist()


def test_metrics_keep_all_classes_in_imbalanced_case() -> None:
    y_true = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    y_pred = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
    y_score = np.full((10, 8), 0.01 / 7)
    y_score[np.arange(10), y_pred] = 0.99

    metrics = classification_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert len(metrics["confusion_matrix"]) == 8
    assert all(len(row) == 8 for row in metrics["confusion_matrix"])
    assert metrics["f1_macro"] == pytest.approx(0.9333333333)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_score", "error"),
    [
        ([], [], np.empty((0, 8)), ValueError),
        ([0, 1], [0], np.eye(2, 8), ValueError),
        ([0, 1], [0, 1], np.ones((2, 7)) / 7, ValueError),
        ([0, 8], [0, 1], np.eye(2, 8), ValueError),
        ([0.0, 1.0], [0, 1], np.eye(2, 8), TypeError),
    ],
)
def test_metrics_reject_invalid_inputs(y_true, y_pred, y_score, error) -> None:
    with pytest.raises(error):
        classification_metrics(y_true, y_pred, y_score)


def test_metrics_require_prediction_to_match_probability_argmax() -> None:
    with pytest.raises(ValueError, match="argmax"):
        classification_metrics([0], [1], np.eye(1, 8))
