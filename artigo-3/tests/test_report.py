"""Testes do consolidador de resultados (#54) com dados sintéticos."""

from __future__ import annotations

import re

import pytest

from src.report import (
    ReportError,
    bootstrap_diferenca,
    bootstrap_ic,
    escrever,
    fmt,
    gerar,
    macro,
    resumo_juizes,
)

CONFIGS = ("slide", "c128-o0")


def _ranking(config, q, r1, r5, mrr, respondivel=True):
    return {
        "split": "test", "config_id": config, "pergunta_id": q, "respondivel": respondivel,
        "categoria": "factual" if respondivel else "nao_respondivel",
        "metricas_evidencia": {
            "recall_at_k": {"1": r1, "3": r5, "5": r5}, "mrr_at_5": mrr,
            "precision_at_k": {"1": r1, "3": r5 / 3, "5": r5 / 5},
            "todas_evidencias_at_k": {"1": bool(r1), "3": bool(r5), "5": bool(r5)},
        },
    }


def _bruto(config, q, *, respondivel=True, abstencao=False, citacoes=(1,)):
    return {
        "config_id": config, "pergunta_id": q, "respondivel": respondivel, "abstencao": abstencao,
        "citacoes": list(citacoes), "citacoes_invalidas": [], "tokens_entrada": 1000 if config == "slide" else 500,
        "latencia_geracao_s": 2.0,
    }


def _dados():
    rankings, brutos = [], []
    for c in CONFIGS:
        for i in range(4):
            bom = c == "slide" or i < 2
            rankings.append(_ranking(c, f"q{i}", 1.0 if bom else 0.0, 1.0, 1.0 if bom else 0.5))
            brutos.append(_bruto(c, f"q{i}"))
        rankings.append(_ranking(c, "q9", 0.0, 0.0, 0.0, respondivel=False))
        brutos.append(_bruto(c, "q9", respondivel=False, abstencao=True, citacoes=()))
    return rankings, brutos


def _macros(lista):
    return dict(re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", m).groups() for m in lista)


def test_macros_correspondem_aos_resultados_brutos():
    linhas, macros, tabela = gerar(*_dados(), n_bootstrap=200)
    m = _macros(macros)
    assert m["NPerguntasTeste"] == "5" and m["NRespondiveisTeste"] == "4" and m["NPares"] == "10"
    assert m["MRRSlide"] == "1{,}00"
    assert m["MRRCPeqSem"] == "0{,}75"  # (1 + 1 + 0,5 + 0,5) / 4
    assert m["RecallUmCPeqSem"] == "0{,}50"
    assert m["SucessoCincoCPeqSem"] == "1{,}00"
    assert m["DifMRRCPeqSem"] == "\\ensuremath{-}0{,}25"
    assert m["AbstencaoCorretaSlide"] == "100\\%"
    assert m["TokensEntradaCPeqSem"] == "500"
    assert m["MelhorMRRConfig"] == "Slide"
    assert "R@1" not in tabela and "P@5" not in tabela and "F1" not in tabela
    assert "Suc.@5" in tabela and "\\textbf{1{,}00}" in tabela
    assert {l["config_id"] for l in linhas} == set(CONFIGS)


def test_nao_respondiveis_ficam_fora_da_recuperacao():
    linhas, _, _ = gerar(*_dados(), n_bootstrap=50)
    slide = next(l for l in linhas if l["config_id"] == "slide")
    assert slide["recall5"] == 1.0  # a não respondível com recall 0 não entra


def test_ic_contem_a_media_e_diferenca_pareada():
    vals = [0.0, 1.0, 1.0, 1.0]
    lo, hi = bootstrap_ic(vals, n=500)
    assert lo <= 0.75 <= hi
    d, dlo, dhi = bootstrap_diferenca({"a": 1.0, "b": 1.0}, {"a": 0.5, "b": 0.5}, n=200)
    assert d == dlo == dhi == 0.5  # diferença constante: IC degenerado


def test_diferencas_de_sobreposicao_sao_pareadas_por_tamanho():
    from src.config import CONFIG

    rankings, brutos = [], []
    for c in CONFIG.chunking.ids:
        for i in range(4):
            valor = 1.0 if c.endswith("o20") else 0.5
            if c == "slide":
                valor = 0.75
            rankings.append(_ranking(c, f"q{i}", valor, valor, valor))
            brutos.append(_bruto(c, f"q{i}"))
    _, macros, _ = gerar(rankings, brutos, n_bootstrap=100)
    m = _macros(macros)
    assert m["DifOverlapRecallCincoCPeq"] == "0{,}50"
    assert m["DifOverlapMRRCMedIC"] == "\\mbox{[0{,}50; 0{,}50]}"
    assert m["NOverlapSucessoCincoCGrd"] == "4"


def test_configuracoes_divergentes_entre_recuperacao_e_geracao():
    rankings, brutos = _dados()
    with pytest.raises(ReportError, match="configurações diferentes"):
        gerar(rankings, [b for b in brutos if b["config_id"] == "slide"], n_bootstrap=10)


def test_juizes_agregam_media_e_kappa():
    notas = {"correcao": 2, "completude": 2, "fidelidade": 2, "afirmacoes_nao_sustentadas": 0}
    ruim = {"correcao": 0, "completude": 0, "fidelidade": 1, "afirmacoes_nao_sustentadas": 2}
    chave = {"R1": ("q0", "slide"), "R2": ("q0", "c128-o0")}
    juizes = {"juiz-a": {"R1": notas, "R2": ruim}, "juiz-b": {"R1": notas, "R2": ruim}}
    agregado, kappas = resumo_juizes(juizes, chave)
    assert agregado["slide"]["correcao"] == 1.0 and agregado["c128-o0"]["fidelidade"] == 0.5
    assert agregado["slide"]["completude"] == 1.0
    assert kappas["correcao"] == 1.0
    with pytest.raises(ReportError, match="dois juízes"):
        resumo_juizes({"juiz-a": juizes["juiz-a"]}, chave)


def test_formatacao_e_nome_de_macro():
    assert fmt(0.876) == "0{,}88" and fmt(6.06, 1) == "6{,}1"
    with pytest.raises(ReportError):
        macro("R5Slide", "x")


def test_escreve_arquivos(tmp_path):
    linhas, macros, tabela = gerar(*_dados(), n_bootstrap=20)
    escrever(linhas, macros, tabela, resultados=tmp_path, paper=tmp_path)
    assert (tmp_path / "resumo.csv").read_text().startswith("config_id,")
    assert "\\newcommand{\\MRRSlide}" in (tmp_path / "results-generated.tex").read_text()
    assert (tmp_path / "tabela-principal.tex").read_text().count("\\toprule") == 1


def test_f1_de_tokens_ignora_citacao_acento_e_palavras_vazias():
    from src.report import f1_tokens, f1_tokens_v1

    assert f1_tokens("É a similaridade do cosseno [1].", "A similaridade do cosseno.") == 1.0
    assert f1_tokens("Não há informação.", "BLEU.") == 0.0
    assert 0 < f1_tokens("BLEU e ROUGE", "BLEU") < 1
    # A variante corrigida distingue a negação; a v1 a removia.
    assert f1_tokens("não supervisionado", "supervisionado") < f1_tokens_v1(
        "não supervisionado", "supervisionado"
    )


def test_precisao_de_citacao_conta_so_trechos_com_evidencia():
    from src.report import por_config_geracao

    base = {
        "config_id": "slide", "pergunta_id": "q1", "respondivel": True, "abstencao": False,
        "resposta": "x [1] [2]", "citacoes": [1, 2], "evidencias_validas": [[15, 3]],
        "trechos": [
            {"aula": 15, "pagina_inicio": 2, "pagina_fim": 4},  # cobre a evidência
            {"aula": 15, "pagina_inicio": 9, "pagina_fim": 9},
        ],
    }
    abst = base | {"pergunta_id": "q2", "abstencao": True, "citacoes": []}
    g = por_config_geracao([base, abst], {"q1": "x", "q2": "y"})["slide"]
    assert g["precisao_citacao"] == {"q1": 0.5}
    assert set(g["f1_tokens_v1"]) == {"q1", "q2"}
    assert set(g["f1_lexical"]) == {"q1", "q2"}
