"""Orquestra os pares modelo × seed e persiste um resultado por execução."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DATA_CONFIG, MODEL_SPECS, TRAINING_CONFIGS
from .data import get_dataloaders, set_seed
from .train import Loaders, train_eval

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _toy_split(repeats: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(DATA_CONFIG.num_classes).repeat_interleave(repeats)
    features = torch.eye(DATA_CONFIG.num_classes).repeat_interleave(repeats, dim=0)
    noise = torch.randn(features.shape, generator=generator) * 0.03
    return TensorDataset(features + noise, targets)


def _toy_experiment(seed: int) -> tuple[nn.Module, Loaders]:
    set_seed(seed)
    model = nn.Linear(DATA_CONFIG.num_classes, DATA_CONFIG.num_classes)
    loader_generator = torch.Generator().manual_seed(seed)
    loaders: Loaders = (
        DataLoader(
            _toy_split(8, seed),
            batch_size=16,
            shuffle=True,
            generator=loader_generator,
        ),
        DataLoader(_toy_split(4, seed + 1), batch_size=16, shuffle=False),
        DataLoader(_toy_split(4, seed + 2), batch_size=16, shuffle=False),
    )
    return model, loaders


def _load_model(model_name: str, seed: int) -> tuple[nn.Module, Loaders]:
    if model_name == "toy":
        return _toy_experiment(seed)

    try:
        spec = MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"Modelo desconhecido: {model_name!r}.") from error

    set_seed(seed)
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if not (
            missing_module == spec.module
            or spec.module.startswith(f"{missing_module}.")
        ):
            raise
        raise RuntimeError(
            f"O modelo {model_name!r} ainda não foi implementado em {spec.module!r}."
        ) from error
    build = getattr(module, "build", None)
    if not callable(build):
        raise RuntimeError(f"{spec.module!r} deve expor uma função build().")
    model = build()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{spec.module}.build() deve devolver um nn.Module.")
    loaders = get_dataloaders(seed)
    return model, loaders


def _training_config_name(model_name: str) -> str:
    if model_name == "toy":
        return "toy"
    try:
        return MODEL_SPECS[model_name].training_config
    except KeyError as error:
        raise ValueError(f"Modelo desconhecido: {model_name!r}.") from error


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
) -> list[Path]:
    """Executa a grade solicitada e grava cada resultado assim que termina."""

    if resume and overwrite:
        raise ValueError("resume e overwrite são mutuamente exclusivos.")
    if not model_names or not seeds:
        raise ValueError("Informe ao menos um modelo e uma seed.")

    output_paths: list[Path] = []
    for model_name in model_names:
        config_name = _training_config_name(model_name)
        try:
            training_config = TRAINING_CONFIGS[config_name]
        except KeyError as error:
            raise ValueError(
                f"Configuração desconhecida para {model_name!r}: {config_name!r}."
            ) from error
        for seed in seeds:
            output_path = output_dir / f"{model_name}-seed{seed}.json"
            if output_path.exists():
                if resume and _valid_existing_result(output_path, model_name, seed):
                    output_paths.append(output_path)
                    print(f"resultado válido já existe, pulando: {output_path}")
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"O resultado já existe: {output_path}. "
                        "Use --resume ou --overwrite."
                    )

            model, loaders = _load_model(model_name, seed)
            result = train_eval(
                model,
                loaders,
                seed,
                model_name=model_name,
                config=training_config,
            )
            _write_result_atomic(result, output_path)
            output_paths.append(output_path)
            print(f"resultado gravado: {output_path}")
    return output_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--smoke",
        action="store_true",
        help="executa somente o modelo linear sintético",
    )
    selection.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        help="modelos reais a executar",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true")
    behavior.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_names = ["toy"] if args.smoke else args.models
    seeds = args.seeds or (
        [DATA_CONFIG.canonical_seeds[0]]
        if args.smoke
        else list(DATA_CONFIG.canonical_seeds)
    )
    run_experiments(
        model_names,
        seeds,
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
