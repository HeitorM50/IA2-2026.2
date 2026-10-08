"""Executa pares detector × semente e persiste um JSON por execução."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import (
    DATA_CONFIG,
    MODEL_NAMES,
    TRAINING_CONFIG,
    TrainingConfig,
    set_seed,
)
from src.data import Dataset, apply_normalizer, load_dataset
from src.faults import EvaluationSet, build_evaluation_set
from src.models.base import Detector
from src.train import train_eval

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_MODULES = {
    "limiar": "src.models.baseline",
    "pca": "src.models.pca",
    "autoencoder": "src.models.autoencoder",
}


def _load_model(model_name: str) -> Detector:
    try:
        module_name = MODEL_MODULES[model_name]
    except KeyError as error:
        raise ValueError(f"Modelo desconhecido: {model_name!r}.") from error
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise RuntimeError(
                f"O modelo {model_name!r} ainda não foi implementado."
            ) from error
        raise
    build = getattr(module, "build", None)
    if not callable(build):
        raise RuntimeError(f"{module_name!r} deve expor uma função build().")
    model = build()
    if not isinstance(model, Detector):
        raise TypeError(f"{module_name}.build() deve devolver um Detector.")
    return model


def _normalized_evaluation(
    windows: np.ndarray,
    channel_std: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
) -> EvaluationSet:
    physical = build_evaluation_set(windows, channel_std, rng)
    normalized = apply_normalizer(physical.X, mean, std)
    return replace(physical, X=normalized)


def _execute_pair(
    model_name: str,
    seed: int,
    dataset: Dataset,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    rng = set_seed(seed)
    model = _load_model(model_name)

    train_X = apply_normalizer(dataset.train.X, dataset.mean, dataset.std)
    validation = _normalized_evaluation(
        dataset.val.X,
        dataset.std,
        dataset.mean,
        dataset.std,
        rng,
    )
    test = _normalized_evaluation(
        dataset.test.X,
        dataset.std,
        dataset.mean,
        dataset.std,
        rng,
    )
    train_y = np.zeros(len(train_X), dtype=np.int8)
    return train_eval(
        model,
        train_X,
        train_y,
        validation,
        test,
        seed,
        training_config=training_config,
    )


def _validate_json_value(value: Any, path: str = "result") -> None:
    if value is None:
        raise ValueError(f"{path} não pode ser null.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contém valor não finito.")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_json_value(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")


def _write_result_atomic(result: dict[str, Any], path: Path) -> None:
    _validate_json_value(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_existing_result(path: Path, model_name: str, seed: int) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        _validate_json_value(result)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == 1
        and result.get("model") == model_name
        and result.get("seed") == seed
    )


def run_experiments(
    model_names: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    resume: bool = False,
    overwrite: bool = False,
    quick: bool = False,
) -> list[Path]:
    """Executa a grade pedida e grava cada resultado assim que ele termina."""

    if resume and overwrite:
        raise ValueError("resume e overwrite são mutuamente exclusivos.")
    if not model_names or not seeds:
        raise ValueError("Informe ao menos um modelo e uma semente.")
    unknown = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {unknown}.")

    training_config = (
        TRAINING_CONFIG.for_quick_run() if quick else TRAINING_CONFIG
    )

    paths: list[Path] = []
    dataset: Dataset | None = None
    for model_name in model_names:
        for seed in seeds:
            output_path = output_dir / f"{model_name}-seed{seed}.json"
            if output_path.exists():
                if resume and _valid_existing_result(output_path, model_name, seed):
                    paths.append(output_path)
                    print(f"resultado válido já existe, pulando: {output_path}")
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"O resultado já existe: {output_path}. "
                        "Use --resume ou --overwrite."
                    )

            # Carrega sob demanda para que --resume não baixe o dataset quando
            # toda a grade solicitada já estiver completa e válida. O loader
            # devolve unidades físicas; a injeção precede a normalização.
            if dataset is None:
                dataset = load_dataset(normalize=False)
            print(f"iniciando execução: {model_name} seed={seed}", flush=True)
            result = _execute_pair(
                model_name,
                seed,
                dataset,
                training_config,
            )
            _write_result_atomic(result, output_path)
            paths.append(output_path)
            print(f"resultado gravado: {output_path}")
    return paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DATA_CONFIG.canonical_seeds),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_experiments(
        args.models,
        args.seeds,
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        quick=args.quick,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
