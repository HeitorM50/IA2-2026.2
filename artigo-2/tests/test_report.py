"""Testes do relatório do Artigo 2.

A regra que estes testes protegem está no `CLAUDE.md` §9: nenhum número do artigo
pode ser digitado à mão. Se o relatório aceitar resultado incompleto ou misturar
agregações, a tabela sai silenciosamente errada — e ninguém percebe até a correção.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import report
from src.report import Aggregate, ResultValidationError


def _resultado(modelo: str, seed: int, agregacao: str, auc: float) -> dict:
    por_classe = {
        c: {"auc_pr": auc, "f1": 0.3, "precision": 0.4, "recall": 0.2,
            "latency": {"defined": True, "detected": 5, "undetected": 1,
                        "mean_detected_samples": 7.0, "median_detected_samples": 7.0}}
        for c in report.FAULTS
    }
    return {
        "schema_version": 1,
        "model": modelo,
        "seed": seed,
        "window_aggregation": agregacao,
        "parameters": {"total": 10, "trainable": 0},
        "training": {"elapsed_seconds": 1.5},
        "test": {
            "auc_pr": auc, "f1": 0.3, "precision": 0.4, "recall": 0.2,
            "latency": {"defined": True, "detected": 5, "undetected": 1,
                        "mean_detected_samples": 7.0, "median_detected_samples": 7.0},
            "per_fault": por_classe,
        },
    }


def _grade(destino: Path, agregacao: str, auc: float = 0.5) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    for modelo in report.MODELS:
        for seed in report.EXPECTED_SEEDS:
            caminho = destino / f"{modelo}-seed{seed}.json"
            caminho.write_text(
                json.dumps(_resultado(modelo, seed, agregacao, auc)), encoding="utf-8"
            )
    return destino


def test_carrega_a_grade_completa(tmp_path):
    por_modelo = report.load_results(_grade(tmp_path / "r", "max"), "max")

    assert set(por_modelo) == set(report.MODELS)
    assert all(len(v) == len(report.EXPECTED_SEEDS) for v in por_modelo.values())


def test_resultado_faltando_interrompe(tmp_path):
    pasta = _grade(tmp_path / "r", "max")
    (pasta / "pca-seed1337.json").unlink()

    with pytest.raises(ResultValidationError, match="ausente"):
        report.load_results(pasta, "max")


def test_agregacao_trocada_interrompe(tmp_path):
    """Misturar agregações na mesma tabela inviabiliza a comparação."""

    pasta = _grade(tmp_path / "r", "mean")

    with pytest.raises(ResultValidationError, match="agregação"):
        report.load_results(pasta, "max")


def test_valor_nao_finito_interrompe(tmp_path):
    pasta = _grade(tmp_path / "r", "max")
    alvo = pasta / "limiar-seed42.json"
    dados = json.loads(alvo.read_text())
    dados["test"]["auc_pr"] = float("nan")
    alvo.write_text(json.dumps(dados))

    with pytest.raises(ResultValidationError, match="finito"):
        report.summarize(report.load_results(pasta, "max"))


def test_virgula_decimal_e_obrigatoria():
    """O artigo é em português: separador decimal é vírgula."""

    assert report._decimal(0.5009, 4) == "0,5009"
    assert "," in report._mean_std(Aggregate(0.5, 0.01, 3), 3)
    assert "." not in report._mean_std(Aggregate(0.5, 0.01, 3), 3)


def test_execucao_unica_nao_reporta_desvio():
    assert "$\\pm$" not in report._mean_std(Aggregate(0.5, 0.0, 1), 3)


def test_apenas_a_melhor_linha_vai_em_negrito(tmp_path):
    resumos = {
        "max": report.summarize(report.load_results(_grade(tmp_path / "a", "max"), "max")),
        "mean": report.summarize(
            report.load_results(_grade(tmp_path / "b", "mean"), "mean")
        ),
    }
    # Desempata para que exista um único melhor por agregação.
    resumos["max"]["autoencoder"]["auc_pr"] = Aggregate(0.9, 0.01, 3)
    resumos["mean"]["pca"]["auc_pr"] = Aggregate(0.9, 0.01, 3)

    tabela = report.build_table(resumos)

    # Uma linha por modelo e uma coluna por agregação: o destaque vai no VALOR
    # da melhor célula de cada coluna, nunca no nome do modelo.
    assert tabela.count("\\textbf{0,9000 $\\pm$ 0,0100}") == 2
    for rotulo in report.MODEL_LABELS.values():
        assert f"\\textbf{{{rotulo}}}" not in tabela


def test_macros_cobrem_todos_os_modelos_e_classes(tmp_path):
    resumos = {
        "max": report.summarize(report.load_results(_grade(tmp_path / "a", "max"), "max")),
        "mean": report.summarize(
            report.load_results(_grade(tmp_path / "b", "mean"), "mean")
        ),
    }

    macros = report.build_macros(resumos)

    for sufixo in ("Max", "Mean"):
        for chave in report.MODEL_KEYS.values():
            assert f"\\Result{chave}AucPr{sufixo}" in macros
            for classe in report.FAULT_KEYS.values():
                assert f"\\Result{chave}{classe}{sufixo}" in macros


def test_relatorio_completo_grava_os_tres_artefatos(tmp_path):
    resumo = tmp_path / "resumo.csv"
    macros = tmp_path / "results-generated.tex"
    figura = tmp_path / "figs" / "fig.pdf"

    saidas = report.generate_report(
        _grade(tmp_path / "max", "max"),
        _grade(tmp_path / "mean", "mean"),
        summary_path=resumo,
        macros_path=macros,
        figure_path=figura,
    )

    assert all(p.exists() for p in saidas.values())
    assert "agregacao,modelo" in resumo.read_text(encoding="utf-8")
    assert "\\newcommand{\\ResultTable}" in macros.read_text(encoding="utf-8")
    assert figura.stat().st_size > 0
