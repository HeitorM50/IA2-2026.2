"""Testes do validador somente leitura do pacote canônico."""

from __future__ import annotations

import json

import pytest

from src.config import CONFIG
from src.validar_artefatos import ArtefatosError, SHA_CANONICO, sha256, validar


def _gravar_json(caminho, valor):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(valor), encoding="utf-8")


def _pacote(tmp_path):
    configs = list(CONFIG.chunking.ids)
    perguntas = [f"q{i:03d}" for i in range(45)]
    selecao = tmp_path / "retrieval/dev/top-k-selection.json"
    _gravar_json(selecao, {"top_k_selecionado": 5, "configs": configs})
    comum = {
        "git_sha": SHA_CANONICO,
        "configs": configs,
        "perguntas": perguntas,
        "corpus_manifesto_sha256": "a",
        "dataset_sha256": "b",
        "protocolo_sha256": "c",
        "selecao_top_k_sha256": sha256(selecao),
    }
    _gravar_json(tmp_path / "raw/test/run-manifest.json", comum | {"top_k": 5})
    _gravar_json(tmp_path / "retrieval/test/run-manifest.json", comum | {"top_k_principal": 5})
    rankings = tmp_path / "retrieval/test/rankings.jsonl"
    rankings.parent.mkdir(parents=True, exist_ok=True)
    rankings.write_text("\n".join(json.dumps({"pergunta_id": q, "config_id": c})
                                   for q in perguntas for c in configs) + "\n", encoding="utf-8")
    for q in perguntas:
        for c in configs:
            _gravar_json(tmp_path / f"raw/test/{c}/{q}.json", {"pergunta_id": q, "config_id": c})
    return tmp_path


def test_valida_grade_hashes_commit_e_top_k(tmp_path):
    resumo = validar(_pacote(tmp_path))
    assert resumo["pares"] == 315
    assert resumo["perguntas"] == 45
    assert resumo["top_k"] == 5


def test_rejeita_grade_incompleta(tmp_path):
    raiz = _pacote(tmp_path)
    (raiz / "raw/test/slide/q000.json").unlink()
    with pytest.raises(ArtefatosError, match="grade de respostas incompleta"):
        validar(raiz)
