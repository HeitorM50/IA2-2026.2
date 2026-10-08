"""Testes do protocolo comum de treino, calibração e avaliação."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from src.config import DetectionConfig, TrainingConfig
from src.faults import EvaluationSet
from src.models.base import Detector
from src.train import calibrate_threshold, train_eval


def _evaluation(
    normal_scores: tuple[float, float] = (1.0, 3.0),
    anomaly_score: float = 10.0,
) -> EvaluationSet:
    X = np.zeros((3, 4, 1), dtype=np.float32)
    X[0, :, 0] = normal_scores[0]
    X[1, :, 0] = normal_scores[1]
    X[2, 1:, 0] = anomaly_score
    return EvaluationSet(
        X=X,
        y=np.array([0, 0, 1], dtype=np.int8),
        fault=np.array(["", "", "gain"]),
        mask=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, True, True, True],
            ]
        ),
        start=np.array([-1, -1, 1], dtype=np.int32),
    )


class ScoreDetector(Detector):
    name = "sintetico"

    def __init__(self) -> None:
        super().__init__()
        self.fit_calls = 0
        self.test_reference: np.ndarray | None = None
        self.test_score_calls = 0

    def fit(self, X: np.ndarray) -> "ScoreDetector":
        self.fit_calls += 1
        self._fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if X is self.test_reference:
            self.test_score_calls += 1
        return np.asarray(X, dtype=np.float32)[:, :, 0]

    def n_parameters(self) -> int:
        self._check_fitted()
        return 2


def _train_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((5, 4, 1), dtype=np.float32),
        np.zeros(5, dtype=np.int8),
    )


def test_calibracao_ignora_escores_anomalos_da_validacao() -> None:
    labels = np.array([0, 0, 1])

    first = calibrate_threshold([1.0, 3.0, 10.0], labels, 3.0)
    second = calibrate_threshold([1.0, 3.0, 1_000_000.0], labels, 3.0)

    assert first == second
    assert first == {
        "source": "validation_normal",
        "normal_windows": 2,
        "score_mean": 2.0,
        "score_std": 1.0,
        "sigma_multiplier": 3.0,
        "threshold": 5.0,
    }


def test_anomalia_no_treino_falha_antes_do_fit() -> None:
    model = ScoreDetector()
    train_X, _ = _train_arrays()

    with pytest.raises(AssertionError, match="anomalias no treino"):
        train_eval(
            model,
            train_X,
            [0, 0, 0, 0, 1],
            _evaluation(),
            _evaluation(),
            42,
            fault_names=("gain",),
        )

    assert model.fit_calls == 0


def test_rotulo_fracionario_na_validacao_nao_pode_virar_normal() -> None:
    train_X, train_y = _train_arrays()
    validation = _evaluation()
    validation = EvaluationSet(
        X=validation.X,
        y=np.array([0.0, 0.0, 0.5]),
        fault=validation.fault,
        mask=validation.mask,
        start=validation.start,
    )

    with pytest.raises(TypeError, match="validation.y"):
        train_eval(
            ScoreDetector(),
            train_X,
            train_y,
            validation,
            _evaluation(),
            42,
            fault_names=("gain",),
        )


def test_teste_e_pontuado_uma_unica_vez_e_nao_calibra_limiar() -> None:
    train_X, train_y = _train_arrays()
    validation = _evaluation()
    test = _evaluation((100.0, 200.0), anomaly_score=500.0)
    model = ScoreDetector()
    model.test_reference = test.X

    result = train_eval(
        model,
        train_X,
        train_y,
        validation,
        test,
        42,
        detection_config=DetectionConfig(calibration_sigmas=3.0),
        fault_names=("gain",),
    )

    assert model.test_score_calls == 1
    assert result["calibration"]["threshold"] == pytest.approx(5.0)
    assert result["test"]["normal_support"] == 2
    assert result["test"]["anomaly_support"] == 1


def test_mesma_semente_produz_mesmo_dicionario_com_relogio_congelado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_X, train_y = _train_arrays()
    clock: Iterator[float] = iter([10.0, 11.0, 10.0, 11.0])
    monkeypatch.setattr("src.train.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("src.train._git_metadata", lambda: ("a" * 40, False))

    first = train_eval(
        ScoreDetector(),
        train_X,
        train_y,
        _evaluation(),
        _evaluation(),
        42,
        fault_names=("gain",),
    )
    second = train_eval(
        ScoreDetector(),
        train_X,
        train_y,
        _evaluation(),
        _evaluation(),
        42,
        fault_names=("gain",),
    )

    assert first == second


def test_loop_mse_para_por_perda_de_validacao() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyAutoencoder(Detector, nn.Module):
        name = "autoencoder-tiny"
        fit_mode = "torch_mse"

        def __init__(self) -> None:
            nn.Module.__init__(self)
            Detector.__init__(self)
            self.linear = nn.Linear(1, 1, bias=False)

        def fit(self, X: np.ndarray) -> "TinyAutoencoder":
            raise AssertionError("O loop comum deve ajustar este modelo.")

        def forward(self, X):
            return self.linear(X)

        def score_samples(self, X: np.ndarray) -> np.ndarray:
            self._check_fitted()
            device = next(self.parameters()).device
            with torch.inference_mode():
                batch = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
                error = (self(batch) - batch).square().mean(dim=2)
            return error.cpu().numpy()

        def n_parameters(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

    train_X, train_y = _train_arrays()
    result = train_eval(
        TinyAutoencoder(),
        train_X,
        train_y,
        _evaluation((0.0, 0.0), anomaly_score=5.0),
        _evaluation((0.0, 0.0), anomaly_score=5.0),
        42,
        training_config=TrainingConfig(
            batch_size=2,
            learning_rate=1e-30,
            weight_decay=0.0,
            max_epochs=8,
            patience=2,
            min_delta=1.0,
        ),
        fault_names=("gain",),
    )

    assert result["training"]["method"] == "gradient_mse"
    assert result["training"]["epochs_ran"] == 3
    assert result["training"]["best_epoch"] == 1
    assert result["training"]["best_validation_loss"] == 0.0
    assert result["parameters"] == {"total": 1, "trainable": 1}


@pytest.mark.parametrize(
    ("scores", "labels", "multiplier"),
    [
        ([], [], 3.0),
        ([1.0], [1], 3.0),
        ([1.0, np.nan], [0, 1], 3.0),
        ([1.0, 2.0], [0], 3.0),
        ([1.0, 2.0], [0, 1], 0.0),
    ],
)
def test_calibracao_rejeita_entradas_invalidas(scores, labels, multiplier) -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(scores, labels, multiplier)
