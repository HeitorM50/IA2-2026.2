"""Testes do protocolo único (#47)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.config import CONFIG, PENDENTE, como_dict, pendencias, validar_congelamento


def test_grade_tem_linha_de_base_por_slide_e_as_seis_janelas():
    assert CONFIG.chunking.ids == (
        "slide",
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


def test_catalogo_cobre_as_dez_aulas_nas_tres_partes():
    cat = CONFIG.corpus.catalogo
    assert sorted(cat) == list(CONFIG.corpus.aulas)
    assert {parte for parte, _ in cat.values()} == {"V", "VI", "VII"}


def test_execucao_canonica_bloqueada_enquanto_ha_pendencia():
    assert "recuperacao.modelo_embedding" not in pendencias()
    assert "geracao.modelo_gerador" in pendencias()
    assert "corpus.catalogo" not in pendencias()
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


def test_modelo_tokenizador_e_revisao_de_embedding_estao_congelados():
    assert CONFIG.chunking.tokenizador == "intfloat/multilingual-e5-base"
    assert CONFIG.recuperacao.modelo_embedding == CONFIG.chunking.tokenizador
    assert CONFIG.recuperacao.revisao_embedding == "d128750597153bb5987e10b1c3493a34e5a4502a"
    assert CONFIG.recuperacao.indice_faiss == "IndexFlatIP"
    assert CONFIG.recuperacao.ks_metricas == (1, 3, 5)
    assert not hasattr(CONFIG.recuperacao, "top_k")
