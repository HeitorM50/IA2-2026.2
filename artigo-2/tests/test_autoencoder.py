"""Testes do Modelo C — autoencoder LSTM sequência-a-sequência."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import (  # noqa: E402
    REPRESENTATION_CONFIG,
    AutoencoderConfig,
    RepresentationConfig,
    TrainingConfig,
)
from src.faults import EvaluationSet  # noqa: E402
from src.models.autoencoder import LSTMAutoencoder, build  # noqa: E402
from src.models.pca import PCADetector  # noqa: E402
from src.train import train_eval  # noqa: E402


WINDOW_SAMPLES = 60
N_CHANNELS = 6


def _normal_windows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 2.0 * np.pi, WINDOW_SAMPLES, dtype=np.float32)
    template = np.stack(
        [
            np.sin(time),
            np.cos(time),
            0.5 * np.sin(time),
            0.25 * np.cos(time),
            np.sin(time + 0.5),
            0.75 * np.sin(time) + 0.25 * np.cos(time),
        ],
        axis=1,
    )
    scales = rng.uniform(0.9, 1.1, size=(n, 1, 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.01, size=(n, WINDOW_SAMPLES, N_CHANNELS))
    return (scales * template[None, ...] + noise).astype(np.float32)


def _faulty_window(normal: np.ndarray) -> np.ndarray:
    faulty = normal.copy()
    faulty[30:, 2] += 8.0
    return faulty


def _evaluation(normal: np.ndarray, faulty: np.ndarray) -> EvaluationSet:
    X = np.concatenate([normal, faulty[None, ...]], axis=0)
    n = len(X)
    mask = np.zeros((n, WINDOW_SAMPLES), dtype=bool)
    mask[-1, 30:] = True
    return EvaluationSet(
        X=X,
        y=np.array([0] * len(normal) + [1], dtype=np.int8),
        fault=np.array([""] * len(normal) + ["gain"]),
        mask=mask,
        start=np.array([-1] * len(normal) + [30], dtype=np.int32),
    )


def _tiny_model() -> LSTMAutoencoder:
    return LSTMAutoencoder(
        representation_cfg=RepresentationConfig(latent_dim=2),
        autoencoder_cfg=AutoencoderConfig(
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            score_batch_size=4,
        ),
    )


def _quick_training() -> TrainingConfig:
    return TrainingConfig(
        batch_size=4,
        learning_rate=1e-2,
        weight_decay=0.0,
        max_epochs=2,
        patience=2,
        min_delta=0.0,
    )


def test_dimensao_latente_e_compartilhada_com_pca() -> None:
    autoencoder = build()
    pca = PCADetector()

    assert autoencoder.latent_dim == pca.latent_dim == REPRESENTATION_CONFIG.latent_dim
    assert autoencoder.latent_dim == 16
    assert "score_windows" not in LSTMAutoencoder.__dict__
    assert "predict" not in LSTMAutoencoder.__dict__


def test_formato_do_escore_e_numero_de_parametros() -> None:
    model = build()
    model._mark_fitted()

    scores = model.score_samples(_normal_windows(3, seed=1))

    assert scores.shape == (3, WINDOW_SAMPLES)
    assert np.isfinite(scores).all()
    assert model.n_parameters() == 12_246


def test_fit_direto_rejeita_treino_sem_validacao() -> None:
    with pytest.raises(RuntimeError, match="validação normal"):
        _tiny_model().fit(_normal_windows(4, seed=2))


def test_falha_injetada_tem_erro_maior_e_json_registra_arquitetura() -> None:
    train = _normal_windows(12, seed=3)
    validation_normal = _normal_windows(3, seed=4)
    test_normal = _normal_windows(3, seed=5)
    faulty = _faulty_window(test_normal[0])
    model = _tiny_model()

    result = train_eval(
        model,
        train,
        np.zeros(len(train), dtype=np.int8),
        _evaluation(validation_normal, _faulty_window(validation_normal[0])),
        _evaluation(test_normal, faulty),
        42,
        training_config=_quick_training(),
        fault_names=("gain",),
    )

    normal_error = model.score_samples(test_normal[:1]).mean()
    faulty_error = model.score_samples(faulty[None, ...]).mean()

    assert faulty_error > normal_error
    assert result["parameters"] == {"total": 360, "trainable": 360}
    assert result["model_details"] == {
        "architecture": "lstm_sequence_autoencoder",
        "input_channels": 6,
        "window_samples": 60,
        "latent_dim": 2,
        "hidden_size": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "decoder_input": "repeated_latent",
    }


def test_mesma_seed_produz_mesmo_resultado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _normal_windows(8, seed=6)
    normal = _normal_windows(2, seed=7)
    evaluation = _evaluation(normal, _faulty_window(normal[0]))
    clock: Iterator[float] = iter([10.0, 11.0, 10.0, 11.0])
    monkeypatch.setattr("src.train.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("src.train._git_metadata", lambda: ("a" * 40, False))

    first = train_eval(
        _tiny_model(),
        train,
        np.zeros(len(train), dtype=np.int8),
        evaluation,
        evaluation,
        1337,
        training_config=_quick_training(),
        fault_names=("gain",),
    )
    second = train_eval(
        _tiny_model(),
        train,
        np.zeros(len(train), dtype=np.int8),
        evaluation,
        evaluation,
        1337,
        training_config=_quick_training(),
        fault_names=("gain",),
    )

    assert first == second


def test_formato_incorreto_e_valor_nao_finito_sao_rejeitados() -> None:
    model = _tiny_model()
    model._mark_fitted()

    with pytest.raises(ValueError, match="diverge"):
        model.score_samples(np.zeros((2, 30, N_CHANNELS), dtype=np.float32))

    invalid = _normal_windows(2, seed=8)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finitos"):
        model.score_samples(invalid)
