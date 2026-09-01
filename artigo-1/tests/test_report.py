from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.config import DATA_CONFIG, TRAINING_CONFIGS
from src.report import ResultValidationError, generate_report, load_canonical_results

PARAMETERS = {
    "logreg": 98_312,
    "cnn": 288_488,
    "resnet18": 11_180_616,
}
BASE_F1 = {"logreg": 0.70, "cnn": 0.80, "resnet18": 0.90}


def _confusion_matrix(
    total: int,
    mistakes_per_class: int,
) -> list[list[int]]:
    quotient, remainder = divmod(total, DATA_CONFIG.num_classes)
    class_sizes = [
        quotient + int(index < remainder)
        for index in range(DATA_CONFIG.num_classes)
    ]
    matrix = [[0] * DATA_CONFIG.num_classes for _ in range(DATA_CONFIG.num_classes)]
    for index, class_size in enumerate(class_sizes):
        matrix[index][index] = class_size - mistakes_per_class
        matrix[index][(index + 1) % DATA_CONFIG.num_classes] = mistakes_per_class
    return matrix


def _result(model: str, seed: int, seed_index: int) -> dict:
    f1 = BASE_F1[model] + seed_index * 0.01
    config = TRAINING_CONFIGS[model]
    return {
        "schema_version": 1,
        "model": model,
        "seed": seed,
        "dataset": {
            "name": DATA_CONFIG.dataset_flag,
            "image_size": DATA_CONFIG.dataset_size,
            "num_classes": DATA_CONFIG.num_classes,
            "split_sizes": DATA_CONFIG.expected_split_sizes,
        },
        "hyperparameters": {
            **config.as_dict(),
            "loss": "CrossEntropyLoss",
            "batch_size": DATA_CONFIG.batch_size,
        },
        "training": {
            "epochs_ran": 3,
            "best_epoch": 2,
            "best_validation_f1_macro": f1,
            "elapsed_seconds": 100.0 + seed_index,
            "history": [{"epoch": epoch} for epoch in range(1, 4)],
        },
        "validation": {
            "loss": 0.5,
            "accuracy": f1,
            "f1_macro": f1,
            "balanced_accuracy": f1,
            "confusion_matrix": _confusion_matrix(DATA_CONFIG.val_size, 12),
        },
        "test": {
            "loss": 0.5,
            "accuracy": f1,
            "f1_macro": f1,
            "balanced_accuracy": f1,
            "confusion_matrix": _confusion_matrix(
                DATA_CONFIG.test_size,
                10 - seed_index,
            ),
        },
        "parameters": {
            "total": PARAMETERS[model],
            "trainable": PARAMETERS[model],
        },
        "environment": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.12.12",
            "python_implementation": "CPython",
            "torch_version": "2.8.0+cu126",
            "torchvision_version": "0.23.0+cu126",
            "device_type": "cuda",
            "device_name": "NVIDIA T4",
            "cuda_version": "12.6",
            "platform": "linux",
        },
    }


def _write_grid(results_dir: Path) -> None:
    results_dir.mkdir()
    for model in PARAMETERS:
        for seed_index, seed in enumerate(DATA_CONFIG.canonical_seeds):
            path = results_dir / f"{model}-seed{seed}.json"
            path.write_text(
                json.dumps(_result(model, seed, seed_index)),
                encoding="utf-8",
            )


def test_generate_report_writes_deterministic_summary_and_pdf(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    summary_path = tmp_path / "resumo.csv"
    figure_path = tmp_path / "confusao.pdf"
    _write_grid(results_dir)

    best_model = generate_report(results_dir, summary_path, figure_path)

    assert best_model == "resnet18"
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model"] for row in rows] == ["logreg", "cnn", "resnet18"]
    assert all(row["runs"] == "3" for row in rows)
    assert float(rows[2]["test_f1_macro_mean"]) == pytest.approx(0.91)
    assert float(rows[2]["test_f1_macro_std"]) == pytest.approx(0.01)
    assert rows[2]["device_name"] == "NVIDIA T4"
    assert figure_path.read_bytes().startswith(b"%PDF")


def test_report_rejects_incomplete_grid(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_grid(results_dir)
    (results_dir / "cnn-seed1337.json").unlink()

    with pytest.raises(ResultValidationError, match="Grade canônica incompleta"):
        load_canonical_results(results_dir)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("hyperparameters", "max_train_batches"), 10, "execução rápida"),
        (("environment", "git_dirty"), True, "árvore Git"),
        (("environment", "device_type"), "cpu", "exige CUDA"),
        (("test", "confusion_matrix"), [[1] * 8 for _ in range(8)], "split completo"),
    ],
)
def test_report_rejects_noncanonical_result(
    tmp_path: Path,
    path: tuple[str, str],
    value,
    message: str,
) -> None:
    results_dir = tmp_path / "results"
    _write_grid(results_dir)
    result_path = results_dir / "logreg-seed42.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[path[0]][path[1]] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ResultValidationError, match=message):
        load_canonical_results(results_dir)


def test_report_rejects_mixed_execution_environments(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_grid(results_dir)
    result_path = results_dir / "resnet18-seed2026.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["environment"]["device_name"] = "NVIDIA L4"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ResultValidationError, match="mesmo commit e ambiente"):
        load_canonical_results(results_dir)
