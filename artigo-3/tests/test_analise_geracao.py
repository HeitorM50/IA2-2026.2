"""Testes da análise da geração sobre resultados brutos sintéticos (#52)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from src.analise_geracao import (
    AnaliseError,
    amostrar_perguntas,
    carregar_brutos,
    cobertura_evidencias,
    desfechos,
    gerar_amostra,
    ler_chave,
    metricas_automaticas,
    recuperou_evidencia,
)

CONFIGS = ("slide", "c256-o0")
CATEGORIAS = {"factual": 6, "contextual": 3, "multitrecho": 2, "nao_respondivel": 1}


def _bruto(qid, config, categoria, *, abstencao=False, citacoes=(1,), invalidas=(), paginas=(3, 3),
           evidencias=((15, 3),)):
    respondivel = categoria != "nao_respondivel"
    return {
        "pergunta_id": qid,
        "config_id": config,
        "pergunta": f"pergunta {qid}",
        "categoria": categoria,
        "respondivel": respondivel,
        "evidencias_validas": [list(e) for e in evidencias] if respondivel else [],
        "trechos": [
            {"aula": 15, "pagina_inicio": paginas[0], "pagina_fim": paginas[1], "texto": "t"}
        ],
        "resposta": "resposta",
        "abstencao": abstencao,
        "citacoes": list(citacoes),
        "citacoes_invalidas": list(invalidas),
        "tokens_entrada": 100,
        "tokens_saida": 10,
        "latencia_geracao_s": 1.0,
    }


def _grade():
    brutos, i = [], 0
    for cat, n in CATEGORIAS.items():
        for _ in range(n):
            i += 1
            for c in CONFIGS:
                brutos.append(_bruto(f"t{i:02d}", c, cat))
    return brutos


def test_evidencia_dentro_do_intervalo_de_paginas():
    assert recuperou_evidencia(_bruto("q", "c", "factual", paginas=(2, 4)))
    assert not recuperou_evidencia(_bruto("q", "c", "factual", paginas=(4, 6)))
    assert not recuperou_evidencia(_bruto("q", "c", "nao_respondivel"))


def test_cobertura_completa_parcial_ausente_e_evidencias_alternativas():
    # Em factual/contextual, uma das páginas anotadas é alternativa suficiente.
    factual = _bruto("q", "c", "factual", paginas=(3, 3), evidencias=((15, 3), (15, 8)))
    assert cobertura_evidencias(factual) == "completa"
    contextual = factual | {"categoria": "contextual"}
    assert cobertura_evidencias(contextual) == "completa"
    # Em multitrecho, todas as páginas são obrigatórias.
    multi = factual | {"categoria": "multitrecho"}
    assert cobertura_evidencias(multi) == "parcial"
    multi["trechos"] = [
        {"aula": 15, "pagina_inicio": 3, "pagina_fim": 3, "texto": "a"},
        {"aula": 15, "pagina_inicio": 8, "pagina_fim": 8, "texto": "b"},
    ]
    assert cobertura_evidencias(multi) == "completa"
    assert cobertura_evidencias(_bruto("q", "c", "multitrecho", paginas=(9, 9),
                                       evidencias=((15, 3), (15, 8)))) == "ausente"
    assert cobertura_evidencias(_bruto("q", "c", "nao_respondivel")) == "ausente"


def test_amostra_estratificada_deterministica_e_com_todas_as_categorias():
    brutos = _grade()
    a1 = amostrar_perguntas(brutos, 6, seed=42)
    assert a1 == amostrar_perguntas(brutos, 6, seed=42)
    assert len(a1) == len(set(a1)) == 6
    cat = {b["pergunta_id"]: b["categoria"] for b in brutos}
    contagem = Counter(cat[q] for q in a1)
    assert set(contagem) == set(CATEGORIAS)
    assert contagem["factual"] == 3  # 6 × 6/12


@pytest.mark.parametrize("n", [0, 3, 13])
def test_amostra_com_n_invalido(n):
    with pytest.raises(AnaliseError):
        amostrar_perguntas(_grade(), n, seed=1)


def test_planilha_da_amostra_leva_todas_as_configuracoes(tmp_path):
    brutos = _grade()
    refs = {b["pergunta_id"]: "ref" for b in brutos}
    perguntas = gerar_amostra(brutos, refs, 4, tmp_path, seed=7)
    chave = ler_chave(tmp_path / "chave.csv")
    assert len(chave) == 4 * len(CONFIGS)
    assert {q for q, _ in chave.values()} == set(perguntas)
    with (tmp_path / "planilha-cega.csv").open(encoding="utf-8") as f:
        cabecalho = next(csv.reader(f))
    assert not any("config" in c for c in cabecalho)


def test_amostra_recusa_grade_incompleta(tmp_path):
    brutos = [b for b in _grade() if not (b["pergunta_id"] == "t01" and b["config_id"] == "slide")]
    with pytest.raises(AnaliseError, match="incompleta"):
        gerar_amostra(brutos, {b["pergunta_id"]: "r" for b in brutos}, 12, tmp_path, seed=1)


def test_metricas_automaticas_por_configuracao():
    brutos = [
        _bruto("a", "slide", "factual"),
        _bruto("b", "slide", "factual", abstencao=True, citacoes=()),
        _bruto("c", "slide", "factual", citacoes=(), invalidas=(7,)),
        _bruto("d", "slide", "nao_respondivel", abstencao=True, citacoes=()),
    ]
    m = metricas_automaticas(brutos)["slide"]
    assert m["n"] == 4
    assert m["abstencao_correta"] == 1.0
    assert m["abstencao_indevida"] == pytest.approx(1 / 3)
    assert m["abstencao_indevida_com_evidencia"] == pytest.approx(1 / 3)
    assert m["respostas_com_citacao"] == pytest.approx(1 / 2)
    assert m["respostas_com_citacao_invalida"] == pytest.approx(1 / 2)
    assert m["n_respostas_substantivas"] == 2
    assert m["n_respostas_com_citacao"] == 1
    assert m["tokens_entrada_medio"] == 100


def test_desfechos_descrevem_correcao_e_cobertura():
    brutos = [
        _bruto("a", "slide", "factual"),  # com evidência
        _bruto("b", "slide", "factual", paginas=(9, 9)),  # sem evidência
        _bruto("c", "slide", "nao_respondivel"),  # respondeu algo
    ]
    chave = {"R1": ("a", "slide"), "R2": ("b", "slide"), "R3": ("c", "slide")}
    nota = lambda c: {"correcao": c, "completude": 0, "fidelidade": 0, "afirmacoes_nao_sustentadas": 0}
    d = desfechos(brutos, {"R1": nota(1), "R2": nota(0), "R3": nota(0)}, chave)["slide"]
    assert d["incorreta_cobertura_completa"] == 1
    assert d["incorreta_cobertura_ausente"] == 1
    assert d["resposta_inventada"] == 1
    assert sum(d.values()) == 3


def test_desfechos_exigem_consenso_completo():
    with pytest.raises(AnaliseError, match="consenso"):
        desfechos([_bruto("a", "slide", "factual")], {}, {"R1": ("a", "slide")})


def test_carrega_brutos_ignorando_manifestos(tmp_path):
    (tmp_path / "slide").mkdir()
    (tmp_path / "slide" / "t01.json").write_text(json.dumps(_bruto("t01", "slide", "factual")))
    (tmp_path / "run-manifest.json").write_text("{}")
    (tmp_path / "slide" / "environment.json").write_text("{}")
    assert [b["pergunta_id"] for b in carregar_brutos(tmp_path)] == ["t01"]
    with pytest.raises(AnaliseError):
        carregar_brutos(tmp_path / "nada")
