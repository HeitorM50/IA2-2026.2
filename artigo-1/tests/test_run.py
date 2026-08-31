from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_CONFIG, TRAINING_CONFIGS
from src.run import DEFAULT_RESULTS_DIR, main, run_experiments


def _image_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    targets = torch.arange(DATA_CONFIG.num_classes).repeat(2)
    images = torch.zeros(
        len(targets),
        DATA_CONFIG.channels,
        DATA_CONFIG.dataset_size,
        DATA_CONFIG.dataset_size,
    )
    for index, target in enumerate(targets.tolist()):
        images[
            index,
            target % DATA_CONFIG.channels,
            target * 8 : (target + 1) * 8,
        ] = 1
    dataset = TensorDataset(images, targets)
    loaders = tuple(
        DataLoader(dataset, batch_size=8, shuffle=False) for _ in range(3)
    )
    return loaders  # type: ignore[return-value]


def test_smoke_cli_writes_complete_json_without_weights(tmp_path: Path) -> None:
    assert main(["--smoke", "--output-dir", str(tmp_path)]) == 0

    result_path = tmp_path / "toy-seed42.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["model"] == "toy"
    assert result["seed"] == 42
    assert result["environment"]["git_commit"]
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_result_requires_resume_or_overwrite(tmp_path: Path) -> None:
    paths = run_experiments(["toy"], [42], tmp_path)
    original = paths[0].read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_experiments(["toy"], [42], tmp_path)

    resumed = run_experiments(["toy"], [42], tmp_path, resume=True)
    assert resumed == paths
    assert paths[0].read_text(encoding="utf-8") == original

    overwritten = run_experiments(["toy"], [42], tmp_path, overwrite=True)
    assert overwritten == paths


def test_resume_rejects_corrupt_result(tmp_path: Path) -> None:
    result_path = tmp_path / "toy-seed42.json"
    result_path.write_text('{"schema_version": 1, "model": "toy"}', encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_experiments(["toy"], [42], tmp_path, resume=True)


def test_unimplemented_model_reports_clear_message(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ainda não foi implementado"):
        run_experiments(["cnn"], [42], tmp_path)


def test_quick_config_preserves_model_hyperparameters() -> None:
    canonical = TRAINING_CONFIGS["resnet18"]

    quick = canonical.for_quick_run()

    assert quick.learning_rate == canonical.learning_rate
    assert quick.head_learning_rate == canonical.head_learning_rate
    assert quick.weight_decay == canonical.weight_decay
    assert quick.head_parameter_prefix == canonical.head_parameter_prefix
    assert quick.max_epochs == 2
    assert quick.patience == 1
    assert quick.max_train_batches == 10
    assert quick.max_eval_batches == 5
    assert canonical.max_epochs == 30
    assert canonical.max_train_batches == 0
    assert canonical.max_eval_batches == 0


def test_quick_cli_loads_baseline_and_writes_noncanonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.run.get_dataloaders", lambda seed: _image_loaders())

    assert (
        main(
            [
                "--models",
                "logreg",
                "--seeds",
                "42",
                "--quick",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    result = json.loads((tmp_path / "logreg-seed42.json").read_text("utf-8"))
    assert result["model"] == "logreg"
    assert result["seed"] == 42
    assert result["parameters"] == {"total": 98_312, "trainable": 98_312}
    assert result["hyperparameters"]["max_epochs"] == 2
    assert result["hyperparameters"]["max_train_batches"] == 10
    assert result["hyperparameters"]["max_eval_batches"] == 5
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.tmp"))


def test_quick_run_rejects_canonical_results_directory() -> None:
    with pytest.raises(ValueError, match="diretório canônico"):
        run_experiments(
            ["logreg"],
            [42],
            DEFAULT_RESULTS_DIR,
            quick=True,
        )


def test_quick_cli_rejects_synthetic_smoke(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--smoke", "--quick", "--output-dir", str(tmp_path)])


def test_quick_cli_requires_noncanonical_output_directory() -> None:
    with pytest.raises(SystemExit):
        main(["--models", "logreg", "--quick"])
