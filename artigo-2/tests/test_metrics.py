"""Testes das metricas de deteccao de anomalias do Artigo 2."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.metrics import (
    binary_detection_metrics,
    detection_latency,
    detection_metrics,
    latency_summary,
)


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected"),
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.1], 0.5),
        ([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9, 0.8], 0.5),
        ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
    ],
)
def test_auc_pr_bate_com_sklearn_em_casos_conhecidos(
    y_true: list[int],
    y_score: list[float],
    expected: float,
) -> None:
    metrics = binary_detection_metrics(y_true, y_score, threshold=0.5)

    assert metrics["auc_pr"] == pytest.approx(expected)
    assert metrics["auc_pr"] == pytest.approx(
        average_precision_score(y_true, y_score)
    )


def test_metricas_no_limiar_tem_valores_conhecidos() -> None:
    metrics = binary_detection_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.7, 0.9],
        threshold=0.75,
    )

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_escore_igual_ao_limiar_nao_dispara() -> None:
    metrics = binary_detection_metrics([0, 1], [0.1, 0.5], threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_detector_perfeito_tem_latencia_zero() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    mask = np.array([False, False, True, True])

    assert detection_latency(scores, mask, threshold=0.5) == 0


def test_disparo_anterior_a_falha_e_ignorado() -> None:
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    mask = np.array([False, False, True, True])

    assert detection_latency(scores, mask, threshold=0.5) == 1


def test_detector_que_nunca_dispara_tem_latencia_indefinida() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    mask = np.array([False, True, True, True])

    assert detection_latency(scores, mask, threshold=0.5) is None

    summary = latency_summary(scores[None, :], mask[None, :], threshold=0.5)
    assert summary == {
        "defined": False,
        "detected": 0,
        "undetected": 1,
    }


def test_resumo_de_latencia_nao_esconde_janelas_nao_detectadas() -> None:
    scores = np.array(
        [
            [0.1, 0.6, 0.2, 0.2],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.6, 0.7],
        ]
    )
    masks = np.array(
        [
            [False, True, True, True],
            [False, True, True, True],
            [False, True, True, True],
        ]
    )

    summary = latency_summary(scores, masks, threshold=0.5)

    assert summary["defined"] is True
    assert summary["detected"] == 2
    assert summary["undetected"] == 1
    assert summary["mean_detected_samples"] == pytest.approx(0.5)
    assert summary["median_detected_samples"] == pytest.approx(0.5)


def test_desagregacao_compara_cada_falha_somente_com_normais() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.95])
    sample_scores = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.1, 0.9, 0.8],
            [0.1, 0.95, 0.1],
        ]
    )
    masks = np.array(
        [
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, False],
        ]
    )
    faults = np.array(["", "", "gain", "spike"])

    metrics = detection_metrics(
        y_true,
        y_score,
        sample_scores,
        threshold=0.5,
        fault=faults,
        fault_masks=masks,
        fault_names=("gain", "spike"),
    )

    assert list(metrics["per_fault"]) == ["gain", "spike"]
    assert metrics["per_fault"]["gain"]["auc_pr"] == 1.0
    assert metrics["per_fault"]["gain"]["normal_support"] == 2
    assert metrics["per_fault"]["gain"]["anomaly_support"] == 1
    assert metrics["per_fault"]["gain"]["latency"]["mean_detected_samples"] == 0.0


def test_saida_e_um_dicionario_json_estrito_sem_nan() -> None:
    metrics = detection_metrics(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.4, 0.3],
        sample_scores=np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.1, 0.4, 0.3],
                [0.1, 0.3, 0.2],
            ]
        ),
        threshold=0.5,
        fault=["", "", "gain", "gain"],
        fault_masks=np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [False, True, True],
            ]
        ),
        fault_names=("gain",),
    )

    encoded = json.dumps(metrics, allow_nan=False)
    decoded = json.loads(encoded)

    assert isinstance(metrics, dict)
    assert decoded["latency"]["defined"] is False
    assert decoded["latency"]["undetected"] == 2
    assert decoded["per_fault"]["gain"]["latency"]["defined"] is False


@pytest.mark.parametrize(
    ("y_true", "y_score", "threshold", "error"),
    [
        ([], [], 0.5, ValueError),
        ([0, 1], [0.1], 0.5, ValueError),
        ([0, 2], [0.1, 0.9], 0.5, ValueError),
        ([0.0, 1.0], [0.1, 0.9], 0.5, TypeError),
        ([0, 1], [0.1, np.nan], 0.5, ValueError),
        ([0, 1], [0.1, 0.9], np.inf, ValueError),
        ([0, 0], [0.1, 0.2], 0.5, ValueError),
    ],
)
def test_metricas_rejeitam_entradas_invalidas(
    y_true,
    y_score,
    threshold,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        binary_detection_metrics(y_true, y_score, threshold)


def test_latencia_rejeita_mascara_vazia() -> None:
    with pytest.raises(ValueError, match="ao menos uma"):
        detection_latency(
            [0.1, 0.9],
            [False, False],
            threshold=0.5,
        )


def test_metricas_rejeitam_mascara_em_janela_normal() -> None:
    with pytest.raises(ValueError, match="Janelas normais"):
        detection_metrics(
            y_true=[0, 1],
            y_score=[0.1, 0.9],
            sample_scores=np.array([[0.1, 0.2], [0.1, 0.9]]),
            threshold=0.5,
            fault=["", "gain"],
            fault_masks=np.array([[True, False], [False, True]]),
            fault_names=("gain",),
        )
