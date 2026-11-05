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
        "metricas_evidencia": {
            "recall_at_k": {"1": r1, "3": r5, "5": r5}, "mrr_at_5": mrr,
            "precision_at_k": {"1": r1, "3": r5 / 3, "5": r5 / 5},
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
    assert m["DifMRRCPeqSem"] == "\\ensuremath{-}0{,}25"
    assert m["AbstencaoCorretaSlide"] == "100\\%"
    assert m["TokensEntradaCPeqSem"] == "500"
    assert m["MelhorMRRConfig"] == "Slide"
    assert "\\textbf{1{,}00}" in tabela and tabela.count("\\\\") == 3  # cabeçalho + 2 linhas
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
