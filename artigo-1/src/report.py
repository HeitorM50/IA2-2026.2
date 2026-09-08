"""Valida e consolida os resultados canônicos do Artigo 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.ticker import PercentFormatter

from .config import DATA_CONFIG, MODEL_SPECS, TRAINING_CONFIGS

MODEL_ORDER = tuple(MODEL_SPECS)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "resumo.csv"
DEFAULT_FIGURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "figs"
    / "confusao-melhor-modelo.pdf"
)
DEFAULT_HISTORY_FIGURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "figs"
    / "acuracia-validacao-epocas.pdf"
)
DEFAULT_TEX_PATH = (
    Path(__file__).resolve().parents[1] / "paper" / "results-generated.tex"
)

MODEL_TEX_PREFIXES = {
    "logreg": "Logreg",
    "cnn": "Cnn",
    "resnet18": "Resnet",
}

MODEL_DISPLAY_NAMES = {
    "logreg": "Regressão logística",
    "cnn": "CNN compacta",
    "resnet18": "ResNet18",
}

SUMMARY_FIELDS = (
    "model",
    "runs",
    "test_f1_macro_mean",
    "test_f1_macro_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_balanced_accuracy_mean",
    "test_balanced_accuracy_std",
    "training_elapsed_seconds_mean",
    "training_elapsed_seconds_std",
    "parameters_total",
    "parameters_trainable",
    "device_name",
)


class ResultValidationError(ValueError):
    """Indica que uma grade não satisfaz o protocolo canônico."""


def _expected_pairs() -> tuple[tuple[str, int], ...]:
    return tuple(
        (model, seed)
        for model in MODEL_ORDER
        for seed in DATA_CONFIG.canonical_seeds
    )


def _validate_finite(value: Any, path: str = "result") -> None:
    if value is None:
        raise ResultValidationError(f"{path} não pode ser null.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ResultValidationError(f"{path} contém valor não finito.")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{path} deve ser um objeto JSON.")
    return value


def _require_metric(section: dict[str, Any], name: str, path: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{path}.{name} deve ser numérico.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ResultValidationError(f"{path}.{name} deve estar no intervalo [0, 1].")
    return number


def _validate_confusion_matrix(
    value: Any,
    expected_size: int,
    path: str,
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != DATA_CONFIG.num_classes:
        raise ResultValidationError(
            f"{path} deve possuir {DATA_CONFIG.num_classes} linhas."
        )
    matrix: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != DATA_CONFIG.num_classes:
            raise ResultValidationError(
                f"{path}[{row_index}] deve possuir {DATA_CONFIG.num_classes} colunas."
            )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in row
        ):
            raise ResultValidationError(
                f"{path}[{row_index}] deve conter inteiros não negativos."
            )
        matrix.append(row)
    if sum(sum(row) for row in matrix) != expected_size:
        raise ResultValidationError(
            f"{path} não representa o split completo de "
            f"{expected_size} exemplos."
        )
    return matrix


def _validate_result(
    result: dict[str, Any],
    model: str,
    seed: int,
    path: Path,
) -> None:
    _validate_finite(result)
    if result.get("schema_version") != 1:
        raise ResultValidationError(f"{path.name}: schema_version deve ser 1.")
    if result.get("model") != model or result.get("seed") != seed:
        raise ResultValidationError(
            f"{path.name}: modelo ou seed não corresponde ao nome do arquivo."
        )

    dataset = _require_mapping(result.get("dataset"), f"{path.name}.dataset")
    if (
        dataset.get("name") != DATA_CONFIG.dataset_flag
        or dataset.get("image_size") != DATA_CONFIG.dataset_size
        or dataset.get("num_classes") != DATA_CONFIG.num_classes
        or dataset.get("split_sizes") != DATA_CONFIG.expected_split_sizes
    ):
        raise ResultValidationError(f"{path.name}: protocolo do dataset diverge.")

    hyperparameters = _require_mapping(
        result.get("hyperparameters"), f"{path.name}.hyperparameters"
    )
    if hyperparameters.get("max_train_batches") != 0 or hyperparameters.get(
        "max_eval_batches"
    ) != 0:
        raise ResultValidationError(f"{path.name}: execução rápida não é canônica.")
    expected_config = TRAINING_CONFIGS[model].as_dict()
    for name, expected in expected_config.items():
        if hyperparameters.get(name) != expected:
            raise ResultValidationError(
                f"{path.name}: hiperparâmetro canônico {name!r} diverge."
            )

    training = _require_mapping(result.get("training"), f"{path.name}.training")
    elapsed = training.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed <= 0
    ):
        raise ResultValidationError(
            f"{path.name}.training.elapsed_seconds deve ser positivo."
        )
    epochs_ran = training.get("epochs_ran")
    if (
        isinstance(epochs_ran, bool)
        or not isinstance(epochs_ran, int)
        or not 1 <= epochs_ran <= expected_config["max_epochs"]
    ):
        raise ResultValidationError(f"{path.name}: número de épocas inválido.")
    history = training.get("history")
    if not isinstance(history, list) or len(history) != epochs_ran:
        raise ResultValidationError(f"{path.name}: histórico de épocas incompleto.")
    for epoch, entry in enumerate(history, start=1):
        entry_path = f"{path.name}.training.history[{epoch - 1}]"
        epoch_result = _require_mapping(entry, entry_path)
        if epoch_result.get("epoch") != epoch:
            raise ResultValidationError(
                f"{entry_path}.epoch deve ser uma sequência iniciada em 1."
            )
        _require_metric(epoch_result, "validation_accuracy", entry_path)

    validation = _require_mapping(
        result.get("validation"), f"{path.name}.validation"
    )
    for metric in ("f1_macro", "accuracy", "balanced_accuracy"):
        _require_metric(validation, metric, f"{path.name}.validation")
    _validate_confusion_matrix(
        validation.get("confusion_matrix"),
        DATA_CONFIG.val_size,
        f"{path.name}.validation.confusion_matrix",
    )
    test = _require_mapping(result.get("test"), f"{path.name}.test")
    for metric in ("f1_macro", "accuracy", "balanced_accuracy"):
        _require_metric(test, metric, f"{path.name}.test")
    _validate_confusion_matrix(
        test.get("confusion_matrix"),
        DATA_CONFIG.test_size,
        f"{path.name}.test.confusion_matrix",
    )

    parameters = _require_mapping(
        result.get("parameters"), f"{path.name}.parameters"
    )
    for name in ("total", "trainable"):
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResultValidationError(
                f"{path.name}.parameters.{name} deve ser inteiro positivo."
            )
    if parameters["trainable"] != parameters["total"]:
        raise ResultValidationError(
            f"{path.name}: todos os parâmetros devem permanecer treináveis."
        )

    environment = _require_mapping(
        result.get("environment"), f"{path.name}.environment"
    )
    if environment.get("device_type") != "cuda":
        raise ResultValidationError(f"{path.name}: execução canônica exige CUDA.")
    if not environment.get("device_name"):
        raise ResultValidationError(f"{path.name}: modelo da GPU não registrado.")
    if environment.get("git_dirty") is not False:
        raise ResultValidationError(f"{path.name}: árvore Git deve estar limpa.")
    if environment.get("git_commit") in (None, "", "unavailable"):
        raise ResultValidationError(f"{path.name}: commit Git não registrado.")


def load_canonical_results(results_dir: Path) -> list[dict[str, Any]]:
    """Carrega a grade 3 × 3 e rejeita qualquer desvio do protocolo."""

    expected_pairs = _expected_pairs()
    expected_names = {f"{model}-seed{seed}.json" for model, seed in expected_pairs}
    actual_names = {path.name for path in results_dir.glob("*.json")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ResultValidationError(
            "Grade canônica incompleta ou inesperada; "
            f"ausentes={missing}, inesperados={unexpected}."
        )

    results: list[dict[str, Any]] = []
    for model, seed in expected_pairs:
        path = results_dir / f"{model}-seed{seed}.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResultValidationError(f"Não foi possível ler {path.name}.") from error
        result = _require_mapping(result, path.name)
        _validate_result(result, model, seed, path)
        results.append(result)

    environment_fields = (
        "git_commit",
        "python_version",
        "python_implementation",
        "torch_version",
        "torchvision_version",
        "device_type",
        "device_name",
        "cuda_version",
        "platform",
    )
    signatures = {
        tuple(result["environment"].get(field) for field in environment_fields)
        for result in results
    }
    if len(signatures) != 1:
        raise ResultValidationError(
            "Os nove resultados devem vir do mesmo commit e ambiente de execução."
        )

    for model in MODEL_ORDER:
        parameter_signatures = {
            (
                result["parameters"]["total"],
                result["parameters"]["trainable"],
            )
            for result in results
            if result["model"] == model
        }
        if len(parameter_signatures) != 1:
            raise ResultValidationError(
                f"As contagens de parâmetros de {model!r} variam entre seeds."
            )
    return results


def summarize_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calcula médias e desvios amostrais na ordem canônica dos modelos."""

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        runs = [result for result in results if result["model"] == model]
        if len(runs) != len(DATA_CONFIG.canonical_seeds):
            raise ResultValidationError(f"{model!r} não possui três execuções.")

        def stats(section: str, metric: str) -> tuple[float, float]:
            values = [float(run[section][metric]) for run in runs]
            return statistics.mean(values), statistics.stdev(values)

        f1_mean, f1_std = stats("test", "f1_macro")
        accuracy_mean, accuracy_std = stats("test", "accuracy")
        balanced_mean, balanced_std = stats("test", "balanced_accuracy")
        elapsed_mean, elapsed_std = stats("training", "elapsed_seconds")
        rows.append(
            {
                "model": model,
                "runs": len(runs),
                "test_f1_macro_mean": f1_mean,
                "test_f1_macro_std": f1_std,
                "test_accuracy_mean": accuracy_mean,
                "test_accuracy_std": accuracy_std,
                "test_balanced_accuracy_mean": balanced_mean,
                "test_balanced_accuracy_std": balanced_std,
                "training_elapsed_seconds_mean": elapsed_mean,
                "training_elapsed_seconds_std": elapsed_std,
                "parameters_total": runs[0]["parameters"]["total"],
                "parameters_trainable": runs[0]["parameters"]["trainable"],
                "device_name": runs[0]["environment"]["device_name"],
            }
        )
    return rows


def _atomic_write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=SUMMARY_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _latex_decimal(value: float, decimal_places: int) -> str:
    return f"{value:.{decimal_places}f}".replace(".", "{,}")


def _latex_mean_std(mean: float, standard_deviation: float, decimal_places: int) -> str:
    return (
        f"{_latex_decimal(mean, decimal_places)} "
        f"\\pm {_latex_decimal(standard_deviation, decimal_places)}"
    )


def _latex_integer(value: int) -> str:
    return f"{value:,}".replace(",", "\\,")


def _latex_text(value: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _atomic_write_latex_macros(
    rows: Sequence[dict[str, Any]],
    best_model: str,
    path: Path,
) -> None:
    runs = {int(row["runs"]) for row in rows}
    devices = {str(row["device_name"]) for row in rows}
    if len(runs) != 1 or len(devices) != 1:
        raise ResultValidationError(
            "As macros LaTeX exigem o mesmo número de execuções e a mesma GPU."
        )

    lines = [
        "% Gerado automaticamente por python -m src.report.",
        "% Fonte: os nove arquivos JSON canônicos em src/results/.",
        "% Não editar manualmente.",
        _latex_command("ResultRuns", str(runs.pop())),
        _latex_command("ResultDevice", _latex_text(devices.pop())),
    ]
    for row in rows:
        model = str(row["model"])
        prefix = MODEL_TEX_PREFIXES[model]
        lines.extend(
            [
                _latex_command(
                    f"Result{prefix}FOne",
                    _latex_mean_std(
                        float(row["test_f1_macro_mean"]),
                        float(row["test_f1_macro_std"]),
                        3,
                    ),
                ),
                _latex_command(
                    f"Result{prefix}BalancedAccuracy",
                    _latex_mean_std(
                        float(row["test_balanced_accuracy_mean"]),
                        float(row["test_balanced_accuracy_std"]),
                        3,
                    ),
                ),
                _latex_command(
                    f"Result{prefix}Accuracy",
                    _latex_mean_std(
                        float(row["test_accuracy_mean"]),
                        float(row["test_accuracy_std"]),
                        3,
                    ),
                ),
                _latex_command(
                    f"Result{prefix}ParametersTotal",
                    _latex_integer(int(row["parameters_total"])),
                ),
                _latex_command(
                    f"Result{prefix}ParametersTrainable",
                    _latex_integer(int(row["parameters_trainable"])),
                ),
                _latex_command(
                    f"Result{prefix}Time",
                    _latex_mean_std(
                        float(row["training_elapsed_seconds_mean"]),
                        float(row["training_elapsed_seconds_std"]),
                        1,
                    ),
                ),
            ]
        )

    best_prefix = MODEL_TEX_PREFIXES[best_model]
    lines.extend(
        [
            _latex_command(
                "ResultBestModel",
                _latex_text(MODEL_DISPLAY_NAMES[best_model]),
            ),
            _latex_command("ResultBestFOne", f"\\Result{best_prefix}FOne"),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _best_model(rows: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["test_f1_macro_mean"]),
            int(row["parameters_trainable"]),
            MODEL_ORDER.index(str(row["model"])),
        ),
    )
    return str(ordered[0]["model"])


def _aggregated_normalized_confusion(
    results: Sequence[dict[str, Any]], model: str
) -> np.ndarray:
    matrices = [
        np.asarray(result["test"]["confusion_matrix"], dtype=np.float64)
        for result in results
        if result["model"] == model
    ]
    aggregate = np.sum(matrices, axis=0)
    row_totals = aggregate.sum(axis=1, keepdims=True)
    if (row_totals <= 0).any():
        raise ResultValidationError("A matriz agregada contém classe verdadeira vazia.")
    return aggregate / row_totals


def _write_confusion_figure(matrix: np.ndarray, path: Path) -> None:
    labels = [
        name.replace("granulócito imaturo", "granulócito\nimaturo")
        for name in DATA_CONFIG.class_names_pt
    ]
    figure, axis = plt.subplots(figsize=(3.5, 3.45), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    axis.set_xlabel("Classe predita", fontsize=9)
    axis.set_ylabel("Classe verdadeira", fontsize=9)
    axis.set_xticks(range(DATA_CONFIG.num_classes), labels, rotation=55, ha="right")
    axis.set_yticks(range(DATA_CONFIG.num_classes), labels)
    axis.tick_params(axis="both", labelsize=8)

    for row in range(DATA_CONFIG.num_classes):
        for column in range(DATA_CONFIG.num_classes):
            value = matrix[row, column]
            if value < 0.02:
                continue
            color = "white" if value >= 0.5 else "black"
            axis.text(
                column,
                row,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, format=path.suffix.lstrip("."), bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _write_accuracy_history_figure(
    results: Sequence[dict[str, Any]], path: Path
) -> None:
    styles = {
        42: ("#0072B2", "-"),
        1337: ("#D55E00", "--"),
        2026: ("#009E73", ":"),
    }
    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(7.15, 2.2))

    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        model_results = sorted(
            (result for result in results if result["model"] == model),
            key=lambda result: int(result["seed"]),
        )
        model_accuracies: list[float] = []
        max_epoch = 1
        for result in model_results:
            history = result["training"]["history"]
            epochs = [int(entry["epoch"]) for entry in history]
            accuracies = [float(entry["validation_accuracy"]) for entry in history]
            seed = int(result["seed"])
            color, linestyle = styles[seed]
            axis.plot(
                epochs,
                accuracies,
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                marker="o",
                markevery=[len(epochs) - 1],
                markersize=3,
                label=f"Semente {seed}",
            )
            model_accuracies.extend(accuracies)
            max_epoch = max(max_epoch, epochs[-1])

        lower = max(0.0, math.floor((min(model_accuracies) - 0.01) / 0.05) * 0.05)
        upper = min(1.0, math.ceil((max(model_accuracies) + 0.01) / 0.05) * 0.05)
        axis.set_ylim(lower, upper)
        axis.set_xlim(0.5, max_epoch + 0.5)
        axis.set_xticks([1, *range(5, max_epoch + 1, 5)])
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_title(MODEL_DISPLAY_NAMES[model], fontsize=9)
        axis.set_xlabel("Época", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Acurácia de validação", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.76, wspace=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, format=path.suffix.lstrip("."), bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def generate_report(
    results_dir: Path,
    summary_path: Path,
    figure_path: Path,
    tex_path: Path | None = None,
    history_figure_path: Path | None = None,
) -> str:
    """Valida resultados e grava os artefatos reproduzíveis do artigo."""

    results = load_canonical_results(results_dir)
    rows = summarize_results(results)
    best_model = _best_model(rows)
    _atomic_write_csv(rows, summary_path)
    if tex_path is not None:
        _atomic_write_latex_macros(rows, best_model, tex_path)
    matrix = _aggregated_normalized_confusion(results, best_model)
    _write_confusion_figure(matrix, figure_path)
    if history_figure_path is not None:
        _write_accuracy_history_figure(results, history_figure_path)
    return best_model


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument(
        "--history-figure", type=Path, default=DEFAULT_HISTORY_FIGURE_PATH
    )
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    best_model = generate_report(
        args.results_dir,
        args.summary,
        args.figure,
        args.tex,
        args.history_figure,
    )
    print(f"grade canônica validada; melhor modelo: {best_model}")
    print(f"resumo gravado: {args.summary}")
    print(f"figura gravada: {args.figure}")
    print(f"curvas de acurácia gravadas: {args.history_figure}")
    print(f"macros LaTeX gravadas: {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
