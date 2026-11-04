"""Testes da geração restrita ao contexto (#52)."""

from __future__ import annotations

from conftest import GeradorFalso, RecuperadorFalso

from src.config import CONFIG
from src.geracao import eh_abstencao, extrair_citacoes, montar_prompt, responder


def test_prompt_numera_trechos_restringe_e_pede_abstencao():
    trechos = RecuperadorFalso("c128-o0").buscar("atenção", 2)
    prompt = montar_prompt("O que é atenção?", trechos)
    assert "[1] (aula 14, p. 1)" in prompt and "[2] (aula 14, p. 2)" in prompt
    assert "SOMENTE" in prompt
    assert CONFIG.geracao.texto_abstencao in prompt
    assert prompt.rstrip().endswith("Pergunta: O que é atenção?\nResposta:")


def test_prompt_sem_trechos_e_explicito():
    assert "(nenhum trecho recuperado)" in montar_prompt("x", [])


def test_abstencao_tolera_caixa_acento_e_pontuacao():
    assert eh_abstencao("nao ha informacao suficiente nas aulas para responder")
    assert eh_abstencao("Resposta: Não há informação suficiente nas aulas para responder!")
    assert not eh_abstencao("A informação está na aula 14 [1].")


def test_citacoes_validas_invalidas_e_agrupadas():
    validas, invalidas = extrair_citacoes("A [1]. B [2, 3]. C [7]. D [1].", n_trechos=5)
    assert validas == [1, 2, 3]
    assert invalidas == [7]


def test_responder_registra_contexto_parametros_e_tokens():
    rec = RecuperadorFalso("c256-o20")
    ger = GeradorFalso()
    r = responder("q01", "O que é atenção?", rec, ger, top_k=3)
    assert r.config_id == "c256-o20" and r.top_k == 3
    assert len(r.trechos) == 3 and r.trechos[0]["origem"] == "Aula14.pdf"
    assert r.citacoes == [1] and r.abstencao is False
    assert r.temperatura == CONFIG.geracao.temperatura
    assert r.tokens_saida == 5 and r.latencia_geracao_s >= 0
    assert ger.prompts == [r.prompt]


def test_mesmo_contexto_mesmo_prompt():
    rec = RecuperadorFalso("c128-o0")
    a = responder("q", "p?", rec, GeradorFalso(), top_k=2)
    b = responder("q", "p?", rec, GeradorFalso(), top_k=2)
    assert a.prompt == b.prompt and a.resposta == b.resposta
