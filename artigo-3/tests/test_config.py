"""Testes do protocolo único (#47)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.config import CONFIG, PENDENTE, como_dict, pendencias, validar_congelamento


def test_grade_tem_as_seis_configuracoes_da_issue():
    assert CONFIG.chunking.ids == (
        "c128-o0",
        "c128-o20",
        "c256-o0",
        "c256-o20",
        "c512-o0",
        "c512-o20",
    )


def test_split_soma_sessenta_itens_e_categorias_batem():
    ds = CONFIG.dataset
    assert ds.n_dev + ds.n_test == 60
    assert sum(ds.categorias.values()) == 60


def test_execucao_canonica_bloqueada_enquanto_ha_pendencia():
    assert "recuperacao.modelo_embedding" in pendencias()
    with pytest.raises(RuntimeError, match="não congelado"):
        validar_congelamento()


def test_protocolo_congelado_passa():
    cfg = replace(
        CONFIG,
        corpus=replace(CONFIG.corpus, catalogo={a: ("V", "t") for a in CONFIG.corpus.aulas}),
        chunking=replace(CONFIG.chunking, tokenizador="tok"),
        recuperacao=replace(CONFIG.recuperacao, modelo_embedding="m", revisao_embedding="r"),
        geracao=replace(CONFIG.geracao, modelo_gerador="g", revisao_gerador="r"),
        dataset=replace(CONFIG.dataset, doi="10.0/x", licenca="CC BY 4.0"),
    )
    assert pendencias(cfg) == []
    validar_congelamento(cfg)


def test_protocolo_e_serializavel():
    texto = json.dumps(como_dict())
    assert PENDENTE in texto
