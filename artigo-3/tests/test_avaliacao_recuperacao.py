"""Testes das métricas de recuperação com rankings de resultado conhecido."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.avaliacao_recuperacao import (
    AvaliacaoRecuperacaoError,
    agregar,
    avaliar_ranking,
    escolher_top_k,
    evidencias_do_trecho,
    redundancia_tokens,
    validar_selecao,
)
from src.config import CONFIG
from src.contratos import Trecho


def _trecho(
    i: int,
    *,
    aula: int = 14,
    pagina_inicio: int | None = None,
    pagina_fim: int | None = None,
    token_inicio: int | None = None,
    token_fim: int | None = None,
) -> Trecho:
    pagina_inicio = pagina_inicio if pagina_inicio is not None else i
    pagina_fim = pagina_fim if pagina_fim is not None else pagina_inicio
    token_inicio = token_inicio if token_inicio is not None else (i - 1) * 10
    token_fim = token_fim if token_fim is not None else token_inicio + 10
    return Trecho(
        chunk_id=f"c{i}",
        texto=f"trecho {i}",
        score=1 - i / 10,
        aula=aula,
        pagina_inicio=pagina_inicio,
        pagina_fim=pagina_fim,
        token_inicio=token_inicio,
        token_fim=token_fim,
        n_tokens=token_fim - token_inicio,
        origem=f"aula-{aula}.pdf",
    )


def test_metricas_sinteticas_tem_valores_conhecidos():
    ranking = [
        _trecho(1, pagina_inicio=1),
        _trecho(2, pagina_inicio=9),
        _trecho(3, pagina_inicio=2),
        _trecho(4, pagina_inicio=1),
        _trecho(5, pagina_inicio=3),
    ]
    resultado = avaliar_ranking(
        ranking, ((14, 1), (14, 2), (14, 3)), respondivel=True
    )
    metricas = resultado["metricas_evidencia"]
    assert metricas["recall_at_k"] == {"1": 1 / 3, "3": 2 / 3, "5": 1.0}
    assert metricas["precision_at_k"] == {"1": 1.0, "3": 2 / 3, "5": 4 / 5}
    assert metricas["todas_evidencias_at_k"] == {"1": False, "3": False, "5": True}
    assert metricas["mrr_at_5"] == 1.0
    assert [r["posicao"] for r in resultado["ranking"]] == [1, 2, 3, 4, 5]


def test_multitrecho_e_diagnostico_distinguem_posicao_baixa_de_ausencia():
    ranking = [
        _trecho(1, pagina_inicio=7),
        _trecho(2, pagina_inicio=8),
        _trecho(3, pagina_inicio=9),
        _trecho(4, pagina_inicio=10),
        _trecho(5, pagina_inicio=1, pagina_fim=2),
    ]
    resultado = avaliar_ranking(
        ranking, ((14, 1), (14, 2), (14, 3)), respondivel=True
    )
    por_id = {e["id"]: e for e in resultado["evidencias"]}
    assert por_id["a14-p0001"]["primeira_posicao"] == 5
    assert por_id["a14-p0002"]["primeira_posicao"] == 5
    assert por_id["a14-p0003"]["primeira_posicao"] is None
    assert por_id["a14-p0003"]["estado"] == "ausente_no_top_5"
    assert resultado["metricas_evidencia"]["recall_at_k"]["5"] == 2 / 3
    assert resultado["ranking"][4]["evidencias_correspondentes"] == [
        "a14-p0001", "a14-p0002"
    ]


def test_mapeamento_de_pagina_e_inclusivo_e_respeita_aula():
    trecho = _trecho(1, pagina_inicio=3, pagina_fim=5)
    assert evidencias_do_trecho(
        trecho, ((14, 3), (14, 4), (14, 5), (14, 6), (15, 4))
    ) == ((14, 3), (14, 4), (14, 5))


def test_redundancia_por_iou_de_tokens():
    a = _trecho(1, token_inicio=0, token_fim=10)
    b = _trecho(2, token_inicio=5, token_fim=15)
    c = _trecho(3, aula=15, token_inicio=0, token_fim=10)
    igual = _trecho(4, token_inicio=0, token_fim=10)
    assert redundancia_tokens([a], 1) == 0.0
    assert redundancia_tokens([a, igual], 2) == 1.0
    assert redundancia_tokens([a, b], 2) == pytest.approx(1 / 3)
    assert redundancia_tokens([a, b, c], 3) == pytest.approx(1 / 9)


def _registro(
    config: str,
    pergunta: str,
    *,
    parte: str,
    respondivel: bool,
    recalls: tuple[float, float, float] = (0.0, 0.0, 0.0),
    latencia: float = 0.1,
) -> dict:
    metricas = None
    if respondivel:
        metricas = {
            "recall_at_k": dict(zip(("1", "3", "5"), recalls)),
            "precision_at_k": {"1": 1.0, "3": 1 / 3, "5": 0.2},
            "todas_evidencias_at_k": {
                "1": recalls[0] == 1,
                "3": recalls[1] == 1,
                "5": recalls[2] == 1,
            },
            "mrr_at_5": 1.0 if recalls[0] else 0.0,
        }
    return {
        "split": "dev",
        "config_id": config,
        "pergunta_id": pergunta,
        "parte": parte,
        "respondivel": respondivel,
        "metricas_evidencia": metricas,
        "redundancia_at_k": {"1": 0.0, "3": 0.1, "5": 0.2},
        "latencia_recuperacao_s": latencia,
    }


def test_agregacao_exclui_nao_respondiveis_apenas_das_metricas_de_evidencia():
    registros = [
        _registro("c", "q1", parte="V", respondivel=True, recalls=(1, 1, 1), latencia=0.1),
        _registro("c", "q2", parte="V", respondivel=False, latencia=0.3),
        _registro("c", "q3", parte="VI", respondivel=True, recalls=(0, 1, 1), latencia=0.2),
    ]
    resumo = agregar(registros)
    geral = resumo["por_configuracao"]["c"]["geral"]
    assert (geral["n_total"], geral["n_respondiveis"], geral["n_nao_respondiveis"]) == (3, 2, 1)
    assert geral["recall_at_k"]["1"] == 0.5
    assert geral["latencia_recuperacao_s"]["media"] == pytest.approx(0.2)
    assert geral["redundancia_at_k"]["5"] == pytest.approx(0.2)
    assert resumo["por_configuracao"]["c"]["VII"]["n_total"] == 0


def test_selecao_global_desempata_pelo_menor_k_e_exige_grade_completa():
    registros = [
        _registro(config, pergunta, parte="V", respondivel=True, recalls=(0.5, 1.0, 1.0))
        for config in ("a", "b")
        for pergunta in ("q1", "q2")
    ]
    escolhido, resultados = escolher_top_k(
        registros,
        configs_esperadas=("a", "b"),
        perguntas_esperadas=("q1", "q2"),
    )
    assert escolhido == 3
    assert resultados == {"1": 0.5, "3": 1.0, "5": 1.0}
    with pytest.raises(AvaliacaoRecuperacaoError, match="grade dev incompleta"):
        escolher_top_k(
            registros[:-1],
            configs_esperadas=("a", "b"),
            perguntas_esperadas=("q1", "q2"),
        )


def test_selecao_rejeita_test_e_valida_regra_contra_adulteracao():
    registro = _registro("a", "q", parte="V", respondivel=True, recalls=(0, 1, 1))
    registro_test = dict(registro, split="test")
    with pytest.raises(AvaliacaoRecuperacaoError, match="somente rankings de dev"):
        escolher_top_k(
            [registro_test], configs_esperadas=("a",), perguntas_esperadas=("q",)
        )

    cfg = replace(
        CONFIG,
        chunking=replace(CONFIG.chunking, estruturais=("a",), tamanhos_tokens=(), sobreposicoes=()),
    )
    artefato = {
        "regra": "maior_recall_medio_dev_desempate_menor_k",
        "candidatos": [1, 3, 5],
        "top_k_selecionado": 5,
        "recall_medio_dev": {"1": 0.0, "3": 1.0, "5": 1.0},
        "configs": ["a"],
        "dev_dataset_sha256": "dev",
        "protocolo_sha256": "invalido",
        "corpus_manifesto_sha256": "corpus",
        "avaliacao_manifesto_sha256": "m",
        "rankings_sha256": "r",
    }
    # Corrige o protocolo para isolar a adulteração do k.
    from src.avaliacao_recuperacao import sha256_protocolo
    artefato["protocolo_sha256"] = sha256_protocolo(cfg)
    with pytest.raises(AvaliacaoRecuperacaoError, match="não corresponde à regra"):
        validar_selecao(
            artefato, cfg=cfg, corpus_sha256="corpus", dev_dataset_sha256="dev"
        )
