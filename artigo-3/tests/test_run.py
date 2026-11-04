"""Testes do orquestrador (#53): retomada, manifesto e guarda da execução canônica."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import GeradorFalso, RecuperadorFalso

from src import dataset as ds
from src.avaliacao_recuperacao import (
    REGRA_SELECAO,
    coletar,
    selecionar,
    sha256_itens,
    sha256_protocolo,
)
from src.config import CONFIG, ExperimentoConfig
from src.run import ExecucaoError, executar


def _item(
    i: int,
    split: str,
    categoria: str = "factual",
    *,
    aula: int = 14,
    parte: str = "V",
) -> dict:
    respondivel = categoria != "nao_respondivel"
    evidencias = [{"aula": aula, "pagina": 1}] if respondivel else []
    if categoria == "multitrecho":
        evidencias.append({"aula": aula, "pagina": 2})
    return {
        "id": f"q{i:02d}", "split": split, "parte": parte, "aula": aula, "pagina": 1,
        "categoria": categoria, "pergunta": f"Pergunta {i}?",
        "resposta_referencia": "r" if respondivel else None,
        "evidencias_validas": evidencias,
        "respondivel": respondivel,
    }


def _dataset_completo() -> list[dict]:
    categorias_por_aula = {
        14: ("factual", "factual", "factual", "contextual", "multitrecho", "nao_respondivel"),
        15: ("factual", "factual", "factual", "contextual", "contextual", "multitrecho"),
        16: ("factual", "factual", "factual", "contextual", "multitrecho", "nao_respondivel"),
        17: ("factual", "factual", "factual", "contextual", "contextual", "multitrecho"),
        18: ("factual", "factual", "factual", "contextual", "multitrecho", "nao_respondivel"),
        19: ("factual", "factual", "factual", "contextual", "contextual", "multitrecho"),
        20: ("factual", "factual", "factual", "contextual", "contextual", "multitrecho"),
        21: ("factual", "factual", "factual", "contextual", "multitrecho", "nao_respondivel"),
        22: ("factual", "factual", "factual", "contextual", "contextual", "multitrecho"),
        23: ("factual", "factual", "factual", "contextual", "multitrecho", "nao_respondivel"),
    }
    # Índices locais escolhidos antes dos experimentos: 8/4/2/1 itens de
    # factual/contextual/multitrecho/não respondível no desenvolvimento.
    dev_por_aula = {
        14: {0, 5}, 15: {0, 3}, 16: {0}, 17: {5}, 18: {0},
        19: {0, 3}, 20: {0, 3}, 21: {0}, 22: {5}, 23: {0, 3},
    }
    partes = {a: "V" if a <= 17 else "VI" if a <= 20 else "VII" for a in range(14, 24)}
    itens = []
    i = 0
    for aula, categorias in categorias_por_aula.items():
        for posicao, categoria in enumerate(categorias):
            split = "dev" if posicao in dev_por_aula[aula] else "test"
            itens.append(_item(i, split, categoria, aula=aula, parte=partes[aula]))
            i += 1
    return itens


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
    arquivos = []
    for aula in range(14, 24):
        nome = f"aula-{aula:02d}.jsonl"
        texto = (f'{{"aula": {aula}, "pagina": 1}}\n' f'{{"aula": {aula}, "pagina": 2}}\n').encode()
        (extraido / nome).write_bytes(texto)
        arquivos.append(
            {"aula": aula, "extraido": nome, "n_paginas": 2,
             "sha256_extraido": hashlib.sha256(texto).hexdigest()}
        )
    manifesto = dados / "manifesto.json"
    manifesto.write_text(json.dumps({"arquivos": arquivos}))
    perguntas = dados / "perguntas.jsonl"
    perguntas.write_text("".join(json.dumps(i) + "\n" for i in _dataset_completo()))

    cfg = replace(
        CONFIG,
        corpus=replace(CONFIG.corpus, extraido_dir=extraido, manifesto_path=manifesto,
                       catalogo={
                           a: ("V" if a <= 17 else "VI" if a <= 20 else "VII", "t")
                           for a in CONFIG.corpus.aulas
                       }),
        chunking=replace(CONFIG.chunking, tokenizador="tok"),
        recuperacao=replace(CONFIG.recuperacao, modelo_embedding="m", revisao_embedding="r"),
        geracao=replace(CONFIG.geracao, modelo_gerador="falso", revisao_gerador="v0"),
        dataset=replace(CONFIG.dataset, dataset_path=perguntas, doi="10.0/x", licenca="CC BY 4.0"),
        results_dir=repo / "results",  # dentro do repo, mas fora da checagem de sujeira
    )
    return cfg, repo


def _rodar(cfg, repo, split="dev", **kw):
    if split == "test" and "selecao_top_k" not in kw:
        kw["selecao_top_k"] = _selecao(cfg)
    return executar(split, fabrica_recuperador=RecuperadorFalso, gerador=GeradorFalso(), cfg=cfg, repo=repo, **kw)


def _selecao(cfg: ExperimentoConfig) -> Path:
    caminho = cfg.results_dir / "top-k-selection.json"
    dev = ds.carregar(
        cfg.dataset.dataset_path,
        split="dev",
        estrito=True,
        manifesto_path=cfg.corpus.manifesto_path,
        cfg=cfg.dataset,
        corpus=cfg.corpus,
    )
    corpus_sha = hashlib.sha256(cfg.corpus.manifesto_path.read_bytes()).hexdigest()
    artefato = {
        "schema_version": 1,
        "regra": REGRA_SELECAO,
        "candidatos": list(cfg.recuperacao.ks_metricas),
        "top_k_selecionado": 3,
        "recall_medio_dev": {"1": 0.5, "3": 1.0, "5": 1.0},
        "configs": list(cfg.chunking.ids),
        "perguntas_dev": [item.id for item in dev],
        "dev_dataset_sha256": sha256_itens(dev),
        "protocolo_sha256": sha256_protocolo(cfg),
        "corpus_manifesto_sha256": corpus_sha,
        "avaliacao_manifesto_sha256": "a" * 64,
        "rankings_sha256": "b" * 64,
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(artefato))
    return caminho


def test_dev_de_ponta_a_ponta_grava_um_json_por_par(ambiente):
    cfg, repo = ambiente
    r = _rodar(cfg, repo, limite=3, configs=("c128-o0", "c512-o20"))
    assert (r.executados, r.pulados) == (6, 0)
    arquivos = sorted(p.relative_to(r.saida).as_posix() for p in r.saida.glob("*/*.json"))
    assert arquivos == [
        f"{c}/{pergunta_id}.json"
        for c in ("c128-o0", "c512-o20")
        for pergunta_id in ("q00", "q05", "q06")
    ]
    dado = json.loads((r.saida / "c128-o0" / "q00.json").read_text())
    assert dado["config_id"] == "c128-o0" and dado["respondivel"] is True
    for nome in ("environment.json", "pip-freeze.txt", "run.log", "run-manifest.json"):
        assert (r.saida / nome).is_file()


def test_retomada_nao_duplica_e_refaz_json_corrompido(ambiente):
    cfg, repo = ambiente
    r1 = _rodar(cfg, repo, limite=3, configs=("c128-o0",))
    (r1.saida / "c128-o0" / "q05.json").write_text("{corrompido")
    (r1.saida / "c128-o0" / "q06.json").unlink()
    r2 = _rodar(cfg, repo, limite=3, configs=("c128-o0",), resume=True)
    assert (r2.executados, r2.pulados) == (2, 1)
    assert len(list(r2.saida.glob("*/*.json"))) == 3
    json.loads((r2.saida / "c128-o0" / "q05.json").read_text())


def test_saida_existente_sem_resume_e_recusada(ambiente):
    cfg, repo = ambiente
    _rodar(cfg, repo, limite=1, configs=("c128-o0",))
    with pytest.raises(ExecucaoError, match="--resume"):
        _rodar(cfg, repo, limite=1, configs=("c128-o0",))


def test_manifesto_incompativel_bloqueia_retomada(ambiente):
    cfg, repo = ambiente
    _rodar(cfg, repo, limite=1, configs=("c128-o0",))
    outro = replace(cfg, recuperacao=replace(cfg.recuperacao, ks_metricas=(1, 3)))
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
                 cfg=cfg, repo=repo, configs=("c128-o0",), selecao_top_k=_selecao(cfg))


def test_canonica_exige_selecao_dev_e_recusa_selecao_adulterada(ambiente):
    cfg, repo = ambiente
    with pytest.raises(ExecucaoError, match="exige --selecao-top-k"):
        executar(
            "test", fabrica_recuperador=RecuperadorFalso, gerador=GeradorFalso(),
            cfg=cfg, repo=repo,
        )
    caminho = _selecao(cfg)
    artefato = json.loads(caminho.read_text())
    artefato["top_k_selecionado"] = 5
    caminho.write_text(json.dumps(artefato))
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(ExecucaoError, match="não corresponde à regra"):
        executar(
            "test", fabrica_recuperador=RecuperadorFalso, gerador=GeradorFalso(),
            cfg=cfg, repo=repo, selecao_top_k=caminho,
        )


def test_avaliacao_recuperacao_ponta_a_ponta_gera_artefatos_e_selecao(ambiente):
    cfg, _ = ambiente
    saida = coletar(
        "dev",
        fabrica_recuperador=RecuperadorFalso,
        cfg=cfg,
        saida=cfg.results_dir / "retrieval" / "dev",
    )
    for nome in ("rankings.jsonl", "summary.json", "summary.csv", "run-manifest.json"):
        assert (saida / nome).is_file()
    registros = [
        json.loads(linha)
        for linha in (saida / "rankings.jsonl").read_text().splitlines()
    ]
    assert len(registros) == len(cfg.chunking.ids) * cfg.dataset.n_dev
    primeiro = registros[0]
    assert primeiro["ranking_id"] and primeiro["manifesto_avaliacao_sha256"]
    assert [r["posicao"] for r in primeiro["ranking"]] == [1, 2, 3, 4, 5]
    resumo = json.loads((saida / "summary.json").read_text())
    assert set(resumo["por_configuracao"]) == set(cfg.chunking.ids)
    assert set(resumo["por_configuracao"]["slide"]) == {"geral", "V", "VI", "VII"}

    selecao = selecionar(saida)
    artefato = json.loads(selecao.read_text())
    # O recuperador falso só devolve aula 14; todos os candidatos empatam na
    # grade completa e a regra escolhe o menor k.
    assert artefato["top_k_selecionado"] == 1
    assert artefato["rankings_sha256"]
