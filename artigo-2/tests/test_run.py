"""Testes da orquestração e persistência dos resultados do Artigo 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import DATA_CONFIG, TrainingConfig
from src.data import Dataset, Split
from src.run import _execute_pair, _write_result_atomic, run_experiments


def _result(model: str, seed: int) -> dict:
    return {
        "schema_version": 1,
        "model": model,
        "seed": seed,
        "test": {"auc_pr": 0.75},
    }


def _synthetic_dataset() -> Dataset:
    rng = np.random.default_rng(7)

    def split(size: int, prefix: str) -> Split:
        X = rng.normal(
            0.0,
            1.0,
            size=(size, DATA_CONFIG.window_s, DATA_CONFIG.n_channels),
        ).astype(np.float32)
        sessions = np.array([f"{prefix}-{index // 10}" for index in range(size)])
        return Split(X=X, session=sessions)

    return Dataset(
        train=split(40, "train"),
        val=split(30, "val"),
        test=split(30, "test"),
        mean=np.zeros(DATA_CONFIG.n_channels, dtype=np.float32),
        std=np.ones(DATA_CONFIG.n_channels, dtype=np.float32),
    )


def test_baseline_percorre_pipeline_completo_em_dados_sinteticos() -> None:
    result = _execute_pair(
        "limiar",
        42,
        _synthetic_dataset(),
        TrainingConfig(),
    )

    assert result["model"] == "limiar"
    assert result["seed"] == 42
    assert result["training"]["method"] == "closed_form"
    assert result["calibration"]["source"] == "validation_normal"
    assert result["calibration"]["normal_windows"] == 24
    assert result["parameters"] == {"total": 12, "trainable": 0}
    assert set(result["test"]["per_fault"]) == {
        "gain",
        "stuck",
        "drift",
        "spike",
        "gap",
    }


def test_pca_percorre_pipeline_completo_e_registra_variancia() -> None:
    result = _execute_pair(
        "pca",
        42,
        _synthetic_dataset(),
        TrainingConfig(),
    )

    assert result["model"] == "pca"
    assert result["training"]["method"] == "closed_form"
    assert result["parameters"] == {"total": 6_120, "trainable": 0}
    assert result["model_details"]["latent_dim"] == 16
    assert result["model_details"]["input_dim"] == 360
    assert 0.0 < result["model_details"]["explained_variance_total"] < 1.0


def test_grava_um_json_por_modelo_e_semente_de_forma_deterministica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.run.load_dataset", lambda normalize: object())
    monkeypatch.setattr(
        "src.run._execute_pair",
        lambda model, seed, dataset, config, detection: _result(model, seed),
    )

    paths = run_experiments(
        ["limiar", "pca"],
        [42, 1337],
        tmp_path,
    )
    first_contents = {path.name: path.read_bytes() for path in paths}
    overwritten = run_experiments(
        ["limiar", "pca"],
        [42, 1337],
        tmp_path,
        overwrite=True,
    )

    assert [path.name for path in paths] == [
        "limiar-seed42.json",
        "limiar-seed1337.json",
        "pca-seed42.json",
        "pca-seed1337.json",
    ]
    assert {path.name: path.read_bytes() for path in overwritten} == first_contents
    assert not list(tmp_path.glob("*.tmp"))


def test_resultado_existente_exige_resume_ou_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.run.load_dataset", lambda normalize: object())
    monkeypatch.setattr(
        "src.run._execute_pair",
        lambda model, seed, dataset, config, detection: _result(model, seed),
    )
    paths = run_experiments(["limiar"], [42], tmp_path)

    with pytest.raises(FileExistsError):
        run_experiments(["limiar"], [42], tmp_path)

    assert run_experiments(["limiar"], [42], tmp_path, resume=True) == paths


def test_escrita_rejeita_nan_e_nao_deixa_temporario(tmp_path: Path) -> None:
    output = tmp_path / "resultado.json"

    with pytest.raises(ValueError, match="não finito"):
        _write_result_atomic({"metric": float("nan")}, output)

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_json_persistido_e_estrito(tmp_path: Path) -> None:
    output = tmp_path / "resultado.json"
    result = _result("limiar", 42)

    _write_result_atomic(result, output)

    assert json.loads(output.read_text(encoding="utf-8")) == result
