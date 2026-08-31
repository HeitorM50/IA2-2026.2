from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.run import main, run_experiments


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


def test_real_model_reports_clear_message_until_builder_exists(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ainda não foi implementado"):
        run_experiments(["logreg"], [42], tmp_path)
