"""Testes do Modelo B — PCA como autoencoder linear."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import REPRESENTATION_CONFIG, RepresentationConfig
from src.faults import EvaluationSet
from src.models.pca import PCADetector, build
from src.train import train_eval


WINDOW_SHAPE = (4, 3)
INPUT_DIM = WINDOW_SHAPE[0] * WINDOW_SHAPE[1]


def _subspace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(2, INPUT_DIM))
    offset = rng.normal(size=INPUT_DIM)
    return basis, offset


def _windows(
    n: int,
    basis: np.ndarray,
    offset: np.ndarray,
    seed: int,
) -> np.ndarray:
    coefficients = np.random.default_rng(seed).normal(size=(n, len(basis)))
    flattened = offset + coefficients @ basis
    return flattened.reshape(n, *WINDOW_SHAPE).astype(np.float32)


def _detector(latent_dim: int = 2) -> PCADetector:
    return PCADetector(
        representation_cfg=RepresentationConfig(latent_dim=latent_dim)
    )


def _evaluation(
    normal: np.ndarray,
    anomaly: np.ndarray,
) -> EvaluationSet:
    X = np.concatenate([normal, anomaly[None, ...]], axis=0)
    n = len(X)
    mask = np.zeros((n, WINDOW_SHAPE[0]), dtype=bool)
    mask[-1, 1:] = True
    return EvaluationSet(
        X=X,
        y=np.array([0] * len(normal) + [1], dtype=np.int8),
        fault=np.array([""] * len(normal) + ["gain"]),
        mask=mask,
        start=np.array([-1] * len(normal) + [1], dtype=np.int32),
    )


def test_dimensao_latente_vem_da_configuracao_compartilhada() -> None:
    detector = PCADetector()

    assert detector.latent_dim == REPRESENTATION_CONFIG.latent_dim == 16
    assert isinstance(build(), PCADetector)


def test_reconstroi_subespaco_linear_com_erro_quase_zero() -> None:
    basis, offset = _subspace()
    train = _windows(40, basis, offset, seed=1)
    normal = _windows(10, basis, offset, seed=2)
    detector = _detector().fit(train)

    scores = detector.score_samples(normal)

    assert scores.shape == (10, WINDOW_SHAPE[0])
    assert np.max(scores) < 1e-12


def test_anomalia_fora_do_subespaco_recebe_escore_maior() -> None:
    basis, offset = _subspace()
    detector = _detector().fit(_windows(40, basis, offset, seed=3))
    normal = _windows(1, basis, offset, seed=4)
    anomaly = normal.copy()
    anomaly[0, 2, 1] += 25.0

    normal_score = detector.score_windows(normal)[0]
    anomaly_score = detector.score_windows(anomaly)[0]

    assert anomaly_score > normal_score * 1_000


def test_contagem_de_parametros_e_variancia_explicada() -> None:
    basis, offset = _subspace()
    detector = _detector().fit(_windows(40, basis, offset, seed=5))

    details = detector.model_details()

    assert detector.n_parameters() == 2 * INPUT_DIM + INPUT_DIM
    assert details["latent_dim"] == 2
    assert details["input_dim"] == INPUT_DIM
    assert len(details["explained_variance_ratio"]) == 2
    assert 0.0 <= details["explained_variance_total"] <= 1.0 + 1e-12
    json.dumps(details, allow_nan=False)


def test_ajuste_e_pontuacao_sao_deterministicos() -> None:
    basis, offset = _subspace()
    train = _windows(40, basis, offset, seed=6)
    test = _windows(10, basis, offset, seed=7)

    first = _detector().fit(train)
    second = _detector().fit(train)

    assert np.array_equal(first.pca_.components_, second.pca_.components_)
    assert np.array_equal(first.score_samples(test), second.score_samples(test))
    assert first.model_details() == second.model_details()


def test_usar_antes_do_ajuste_falha_alto() -> None:
    basis, offset = _subspace()

    with pytest.raises(RuntimeError, match="ajustado"):
        _detector().score_samples(_windows(2, basis, offset, seed=8))


def test_formato_diferente_do_treino_e_rejeitado() -> None:
    basis, offset = _subspace()
    detector = _detector().fit(_windows(40, basis, offset, seed=9))

    with pytest.raises(ValueError, match="diverge"):
        detector.score_samples(np.zeros((2, 6, 2), dtype=np.float32))


def test_ajuste_rejeita_dimensao_inviavel_e_dados_constantes() -> None:
    with pytest.raises(ValueError, match="dimensão latente"):
        _detector(latent_dim=5).fit(
            np.ones((5, *WINDOW_SHAPE), dtype=np.float32)
        )

    with pytest.raises(ValueError, match="constantes"):
        _detector().fit(np.ones((10, *WINDOW_SHAPE), dtype=np.float32))


def test_train_eval_registra_variancia_explicada_no_json() -> None:
    basis, offset = _subspace()
    train = _windows(40, basis, offset, seed=10)
    normal_validation = _windows(5, basis, offset, seed=11)
    normal_test = _windows(5, basis, offset, seed=12)
    anomaly = normal_validation[0].copy()
    anomaly[1:, 2] += 20.0

    result = train_eval(
        _detector(),
        train,
        np.zeros(len(train), dtype=np.int8),
        _evaluation(normal_validation, anomaly),
        _evaluation(normal_test, anomaly),
        42,
        fault_names=("gain",),
    )

    assert result["model"] == "pca"
    assert result["training"]["method"] == "closed_form"
    assert result["parameters"] == {"total": 36, "trainable": 0}
    assert result["model_details"]["latent_dim"] == 2
    assert result["model_details"]["explained_variance_total"] > 0.0
    json.dumps(result, allow_nan=False)


def test_train_eval_impede_pca_de_ver_anomalia_no_treino() -> None:
    basis, offset = _subspace()
    train = _windows(40, basis, offset, seed=13)
    normal = _windows(5, basis, offset, seed=14)
    anomaly = normal[0].copy()
    anomaly[1:, 0] += 10.0
    detector = _detector()
    labels = np.zeros(len(train), dtype=np.int8)
    labels[-1] = 1

    with pytest.raises(AssertionError, match="anomalias no treino"):
        train_eval(
            detector,
            train,
            labels,
            _evaluation(normal, anomaly),
            _evaluation(normal, anomaly),
            42,
            fault_names=("gain",),
        )

    assert detector.pca_ is None
