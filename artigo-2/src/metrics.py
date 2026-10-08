"""Metricas de avaliacao dos detectores de anomalia do Artigo 2.

As funcoes deste modulo recebem somente vetores NumPy. A calibracao do limiar,
o carregamento dos dados e a execucao dos modelos ficam fora daqui, o que torna
o protocolo testavel com escores sinteticos e impede que as metricas consultem
o conjunto de teste para escolher o ponto de operacao.

Embora o artigo use o nome AUC-PR, a definicao operacional e a precisao media
(`average_precision_score`), conforme fixado nos criterios de aceite.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FAULT_NAMES


def _as_binary_vector(
    values: Sequence[int] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} deve ser um vetor; recebido formato {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} nao pode estar vazio.")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} deve conter rotulos binarios inteiros.")

    array = array.astype(np.int8, copy=False)
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} deve conter somente os rotulos 0 e 1.")
    return array


def _as_score_vector(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} deve ser um vetor; recebido formato {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} nao pode estar vazio.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contem valor nao finito.")
    return array


def _as_sample_scores(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"{name} deve ter formato (janelas, amostras); recebido {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} nao pode estar vazio.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contem valor nao finito.")
    return array


def _as_masks(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"{name} deve ter formato (janelas, amostras); recebido {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} nao pode estar vazio.")
    if not np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} deve conter valores booleanos.")
    return array.astype(bool, copy=False)


def _as_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("threshold deve ser numerico.")
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError("threshold deve ser finito.")
    return threshold


def binary_detection_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcula AUC-PR e metricas no limiar para uma tarefa binaria.

    Escores podem assumir qualquer valor real finito; quanto maior, mais
    anomala e a janela. O limiar e aplicado com ``>``, igual a
    :meth:`src.models.base.Detector.predict`.
    """

    true = _as_binary_vector(y_true, "y_true")
    score = _as_score_vector(y_score, "y_score")
    cutoff = _as_threshold(threshold)

    if true.shape != score.shape:
        raise ValueError("y_true e y_score devem ter o mesmo numero de exemplos.")
    if np.unique(true).size != 2:
        raise ValueError("AUC-PR exige ao menos um exemplo normal e um anomalo.")

    predicted = (score > cutoff).astype(np.int8)
    return {
        "auc_pr": float(average_precision_score(true, score)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
    }


def detection_latency(
    sample_scores: Sequence[float] | np.ndarray,
    fault_mask: Sequence[bool] | np.ndarray,
    threshold: float,
) -> int | None:
    """Retorna amostras entre o inicio real da falha e o primeiro alarme.

    A primeira posicao verdadeira da mascara e o inicio observado da falha. Um
    disparo anterior a ela nao conta. Se nao houver cruzamento do limiar dali
    ate o fim da janela, a latencia e indefinida e ``None`` e retornado
    explicitamente.
    """

    scores = _as_score_vector(sample_scores, "sample_scores")
    mask = np.asarray(fault_mask)
    if mask.ndim != 1:
        raise ValueError(
            f"fault_mask deve ser um vetor; recebido formato {mask.shape}."
        )
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("fault_mask deve conter valores booleanos.")
    if mask.shape != scores.shape:
        raise ValueError(
            "sample_scores e fault_mask devem ter o mesmo numero de amostras."
        )
    if not mask.any():
        raise ValueError("fault_mask deve marcar ao menos uma amostra alterada.")

    cutoff = _as_threshold(threshold)
    onset = int(np.flatnonzero(mask)[0])
    crossings = np.flatnonzero(scores[onset:] > cutoff)
    if crossings.size == 0:
        return None
    return int(crossings[0])


def latency_summary(
    sample_scores: np.ndarray,
    fault_masks: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Resume latencias de janelas anomalas sem esconder nao deteccoes.

    Media e mediana sao condicionais as janelas detectadas. As contagens de
    detectadas e nao detectadas permanecem no resultado para que essa media nao
    faca um detector com baixa cobertura parecer artificialmente rapido.
    """

    scores = _as_sample_scores(sample_scores, "sample_scores")
    masks = _as_masks(fault_masks, "fault_masks")
    cutoff = _as_threshold(threshold)
    if scores.shape != masks.shape:
        raise ValueError("sample_scores e fault_masks devem ter o mesmo formato.")

    latencies = [
        detection_latency(row_scores, row_mask, cutoff)
        for row_scores, row_mask in zip(scores, masks, strict=True)
    ]
    detected_latencies = [value for value in latencies if value is not None]
    summary: dict[str, Any] = {
        "defined": bool(detected_latencies),
        "detected": len(detected_latencies),
        "undetected": len(latencies) - len(detected_latencies),
    }
    if detected_latencies:
        summary.update(
            {
                "mean_detected_samples": float(np.mean(detected_latencies)),
                "median_detected_samples": float(np.median(detected_latencies)),
            }
        )
    return summary


def _as_fault_vector(
    values: Sequence[str] | np.ndarray,
    expected_size: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError(
            "fault deve ser um vetor com o mesmo numero de entradas de y_true."
        )
    if not all(isinstance(value, str) for value in array.tolist()):
        raise TypeError("fault deve conter nomes de classe em texto.")
    return array.astype(str, copy=False)


def detection_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    sample_scores: np.ndarray,
    threshold: float,
    fault: Sequence[str] | np.ndarray,
    fault_masks: np.ndarray,
    *,
    fault_names: Sequence[str] = FAULT_NAMES,
) -> dict[str, Any]:
    """Calcula as metricas globais e uma avaliacao contra normal por falha.

    Na desagregacao, cada classe e comparada somente com janelas normais. As
    demais falhas sao excluidas para que uma deteccao correta de outra anomalia
    nao seja contabilizada como falso positivo da classe em foco.
    """

    true = _as_binary_vector(y_true, "y_true")
    score = _as_score_vector(y_score, "y_score")
    scores_by_sample = _as_sample_scores(sample_scores, "sample_scores")
    masks = _as_masks(fault_masks, "fault_masks")
    cutoff = _as_threshold(threshold)
    faults = _as_fault_vector(fault, true.size)

    if score.size != true.size or scores_by_sample.shape[0] != true.size:
        raise ValueError("Todos os escores devem ter uma entrada por janela.")
    if masks.shape != scores_by_sample.shape:
        raise ValueError("sample_scores e fault_masks devem ter o mesmo formato.")

    names = tuple(fault_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("fault_names deve conter nomes unicos.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("fault_names deve conter nomes de classe nao vazios.")

    normal = true == 0
    anomalous = true == 1
    if np.any(faults[normal] != ""):
        raise ValueError("Janelas normais devem ter fault vazio.")
    if np.any(~np.isin(faults[anomalous], names)):
        raise ValueError("Toda janela anomala deve ter uma classe declarada.")
    if masks[normal].any():
        raise ValueError("Janelas normais nao podem ter amostras na fault_mask.")
    if anomalous.any() and np.any(~masks[anomalous].any(axis=1)):
        raise ValueError("Toda janela anomala deve ter ao menos uma amostra alterada.")

    result: dict[str, Any] = {
        "threshold": cutoff,
        **binary_detection_metrics(true, score, cutoff),
        "normal_support": int(normal.sum()),
        "anomaly_support": int(anomalous.sum()),
        "latency": latency_summary(
            scores_by_sample[anomalous], masks[anomalous], cutoff
        ),
    }

    per_fault: dict[str, Any] = {}
    for name in names:
        selected_fault = faults == name
        if not selected_fault.any():
            raise ValueError(f"A classe de falha {name!r} nao possui exemplos.")

        selected = normal | selected_fault
        class_true = selected_fault[selected].astype(np.int8)
        per_fault[name] = {
            **binary_detection_metrics(class_true, score[selected], cutoff),
            "normal_support": int(normal.sum()),
            "anomaly_support": int(selected_fault.sum()),
            "latency": latency_summary(
                scores_by_sample[selected_fault],
                masks[selected_fault],
                cutoff,
            ),
        }
    result["per_fault"] = per_fault
    return result
