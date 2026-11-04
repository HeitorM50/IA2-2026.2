"""Testes do orquestrador (#53): retomada, manifesto e guarda da execução canônica."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import GeradorFalso, RecuperadorFalso

from src.config import CONFIG, ExperimentoConfig
from src.run import ExecucaoError, executar


def _item(i: int, split: str, categoria: str = "factual") -> dict:
    respondivel = categoria != "nao_respondivel"
    return {
        "id": f"q{i:02d}", "split": split, "parte": "V", "aula": 14, "pagina": 1,
        "categoria": categoria, "pergunta": f"Pergunta {i}?", "resposta_referencia": "r",
        "evidencias_validas": [{"aula": 14, "pagina": 1}] if respondivel else [],
        "respondivel": respondivel,
    }


def _dataset_completo() -> list[dict]:
    cats = ["factual"] * 30 + ["contextual"] * 15 + ["multitrecho"] * 10 + ["nao_respondivel"] * 5
    return [_item(i, "dev" if i < 15 else "test", c) for i, c in enumerate(cats)]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def ambiente(tmp_path) -> tuple[ExperimentoConfig, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "codigo.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "inicial")

    dados = tmp_path / "dados"
    extraido = dados / "extraido"
    extraido.mkdir(parents=True)
    texto = b'{"aula": 14, "pagina": 1}\n'
    (extraido / "aula-14.jsonl").write_bytes(texto)
    manifesto = dados / "manifesto.json"
    manifesto.write_text(json.dumps({"arquivos": [
        {"extraido": "aula-14.jsonl", "sha256_extraido": hashlib.sha256(texto).hexdigest()}
    ]}))
    perguntas = dados / "perguntas.jsonl"
    perguntas.write_text("".join(json.dumps(i) + "\n" for i in _dataset_completo()))

    cfg = replace(
        CONFIG,
        corpus=replace(CONFIG.corpus, extraido_dir=extraido, manifesto_path=manifesto,
                       catalogo={a: ("V", "t") for a in CONFIG.corpus.aulas}),
        chunking=replace(CONFIG.chunking, tokenizador="tok"),
        recuperacao=replace(CONFIG.recuperacao, modelo_embedding="m", revisao_embedding="r"),
        geracao=replace(CONFIG.geracao, modelo_gerador="falso", revisao_gerador="v0"),
        dataset=replace(CONFIG.dataset, dataset_path=perguntas, doi="10.0/x", licenca="CC BY 4.0"),
        results_dir=repo / "results",  # dentro do repo, mas fora da checagem de sujeira
    )
    return cfg, repo


def _rodar(cfg, repo, split="dev", **kw):
    return executar(split, fabrica_recuperador=RecuperadorFalso, gerador=GeradorFalso(), cfg=cfg, repo=repo, **kw)


def test_dev_de_ponta_a_ponta_grava_um_json_por_par(ambiente):
    cfg, repo = ambiente
    r = _rodar(cfg, repo, limite=3, configs=("c128-o0", "c512-o20"))
    assert (r.executados, r.pulados) == (6, 0)
    arquivos = sorted(p.relative_to(r.saida).as_posix() for p in r.saida.glob("*/*.json"))
    assert arquivos == [f"{c}/q{i:02d}.json" for c in ("c128-o0", "c512-o20") for i in range(3)]
    dado = json.loads((r.saida / "c128-o0" / "q00.json").read_text())
    assert dado["config_id"] == "c128-o0" and dado["respondivel"] is True
    for nome in ("environment.json", "pip-freeze.txt", "run.log", "run-manifest.json"):
        assert (r.saida / nome).is_file()


def test_retomada_nao_duplica_e_refaz_json_corrompido(ambiente):
    cfg, repo = ambiente
    r1 = _rodar(cfg, repo, limite=3, configs=("c128-o0",))
    (r1.saida / "c128-o0" / "q01.json").write_text("{corrompido")
    (r1.saida / "c128-o0" / "q02.json").unlink()
    r2 = _rodar(cfg, repo, limite=3, configs=("c128-o0",), resume=True)
    assert (r2.executados, r2.pulados) == (2, 1)
    assert len(list(r2.saida.glob("*/*.json"))) == 3
    json.loads((r2.saida / "c128-o0" / "q01.json").read_text())


def test_saida_existente_sem_resume_e_recusada(ambiente):
    cfg, repo = ambiente
    _rodar(cfg, repo, limite=1, configs=("c128-o0",))
    with pytest.raises(ExecucaoError, match="--resume"):
        _rodar(cfg, repo, limite=1, configs=("c128-o0",))


def test_manifesto_incompativel_bloqueia_retomada(ambiente):
    cfg, repo = ambiente
    _rodar(cfg, repo, limite=1, configs=("c128-o0",))
    outro = replace(cfg, recuperacao=replace(cfg.recuperacao, top_k=3))
    with pytest.raises(ExecucaoError, match="incompatível.*protocolo_sha256"):
        _rodar(outro, repo, limite=1, configs=("c128-o0",), resume=True)


def test_corpus_diferente_do_manifesto_aborta(ambiente):
    cfg, repo = ambiente
    (cfg.corpus.extraido_dir / "aula-14.jsonl").write_text("alterado\n")
    with pytest.raises(ExecucaoError, match="não corresponde ao manifesto"):
        _rodar(cfg, repo, limite=1)


def test_canonica_exige_git_limpo_e_sha_destacado(ambiente):
    cfg, repo = ambiente
    (repo / "codigo.py").write_text("x = 2\n")
    with pytest.raises(ExecucaoError, match="Git sujo"):
        _rodar(cfg, repo, split="test")
    _git(repo, "checkout", "-q", "codigo.py")
    with pytest.raises(ExecucaoError, match="SHA destacado"):
        _rodar(cfg, repo, split="test")
    _git(repo, "checkout", "-q", "--detach")
    r = _rodar(cfg, repo, split="test", configs=("c128-o0",))
    assert r.executados == 45


def test_canonica_recusa_protocolo_pendente_e_limite(ambiente):
    cfg, repo = ambiente
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(ExecucaoError, match="--limite"):
        _rodar(cfg, repo, split="test", limite=2)
    pendente = replace(cfg, geracao=replace(cfg.geracao, modelo_gerador="PENDENTE"))
    with pytest.raises(RuntimeError, match="não congelado"):
        _rodar(pendente, repo, split="test")


def test_canonica_recusa_gerador_diferente_do_congelado(ambiente):
    cfg, repo = ambiente
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(ExecucaoError, match="difere do congelado"):
        executar("test", fabrica_recuperador=RecuperadorFalso, gerador=GeradorFalso(revisao="v1"),
                 cfg=cfg, repo=repo, configs=("c128-o0",))
