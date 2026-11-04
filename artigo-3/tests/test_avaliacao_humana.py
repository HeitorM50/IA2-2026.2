"""Testes da avaliação cega e da concordância (#52)."""

from __future__ import annotations

import csv

import pytest
from conftest import GeradorFalso, RecuperadorFalso

from src.avaliacao_humana import (
    AvaliacaoError,
    classificar_falha,
    concordancia,
    divergencias,
    gerar_planilha_cega,
    kappa_cohen,
    ler_avaliacao,
)
from src.geracao import responder


def _respostas():
    return [
        responder(q, f"pergunta {q}", RecuperadorFalso(c), GeradorFalso(), top_k=2).como_dict()
        for c in ("c128-o0", "c512-o20")
        for q in ("q1", "q2", "q3")
    ]


def test_planilha_nao_expoe_configuracao_e_chave_desfaz(tmp_path):
    destino, chave = tmp_path / "planilha.csv", tmp_path / "chave.csv"
    refs = {"q1": "r1", "q2": "r2", "q3": "r3"}
    gerar_planilha_cega(_respostas(), refs, destino, chave, seed=7)

    texto = destino.read_text(encoding="utf-8")
    assert "c128" not in texto and "c512" not in texto and "config" not in texto
    linhas = list(csv.DictReader(destino.open(encoding="utf-8")))
    assert len(linhas) == 6 and linhas[0]["correcao"] == ""
    mapa = {l["item"]: (l["pergunta_id"], l["config_id"]) for l in csv.DictReader(chave.open())}
    assert sorted(mapa.values()) == sorted((q, c) for c in ("c128-o0", "c512-o20") for q in ("q1", "q2", "q3"))


def test_planilha_e_deterministica_pela_seed(tmp_path):
    refs = {"q1": "r1", "q2": "r2", "q3": "r3"}
    for nome in ("a", "b"):
        gerar_planilha_cega(_respostas(), refs, tmp_path / f"{nome}.csv", tmp_path / f"k{nome}.csv", seed=7)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_kappa_casos_conhecidos():
    assert kappa_cohen([0, 1, 2, 2], [0, 1, 2, 2], (0, 1, 2)) == pytest.approx(1.0)
    # Exemplo clássico não ponderado: concordância observada 0,7; esperada 0,5 → κ = 0,4
    # 50 itens: sim/sim 20, sim/não 5, não/sim 10, não/não 15
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert kappa_cohen(a, b, (0, 1), ponderado=False) == pytest.approx(0.4)
    # Discordar por 1 ponto pesa menos que discordar por 2 no ponderado
    perto = kappa_cohen([0, 1, 2, 0, 2], [1, 1, 2, 0, 2], (0, 1, 2))
    longe = kappa_cohen([0, 1, 2, 0, 2], [2, 1, 2, 0, 2], (0, 1, 2))
    assert perto > longe


def _planilha(tmp_path, nome, notas):
    caminho = tmp_path / nome
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "correcao", "completude", "fidelidade", "afirmacoes_nao_sustentadas", "observacao"])
        for item, valores in notas.items():
            w.writerow([item, *valores, ""])
    return caminho


def test_concordancia_e_divergencias(tmp_path):
    av1 = ler_avaliacao(_planilha(tmp_path, "a.csv", {"R1": (2, 2, 2, 0), "R2": (0, 1, 1, 2)}))
    av2 = ler_avaliacao(_planilha(tmp_path, "b.csv", {"R1": (2, 2, 2, 0), "R2": (1, 1, 1, 2)}))
    c = concordancia(av1, av2)
    assert c["completude"]["concordancia_exata"] == 1.0
    assert c["correcao"]["concordancia_exata"] == 0.5
    assert divergencias(av1, av2) == [("R2", "correcao", 0, 1)]


def test_nota_fora_da_rubrica_e_rejeitada(tmp_path):
    with pytest.raises(AvaliacaoError, match="fora de"):
        ler_avaliacao(_planilha(tmp_path, "x.csv", {"R1": (3, 2, 2, 0)}))


@pytest.mark.parametrize(
    "respondivel, recuperou, abstencao, correcao, esperado",
    [
        (True, True, False, 2, "acerto"),
        (True, True, False, 1, "falha_geracao"),
        (True, True, True, 0, "falha_geracao"),
        (True, False, False, 0, "falha_recuperacao"),
        (True, False, False, 2, "acerto_sem_evidencia"),
        (False, False, True, 0, "abstencao_correta"),
        (False, False, False, 0, "resposta_inventada"),
    ],
)
def test_falha_de_recuperacao_separada_da_geracao(respondivel, recuperou, abstencao, correcao, esperado):
    assert (
        classificar_falha(respondivel=respondivel, recuperou_evidencia=recuperou, abstencao=abstencao, correcao=correcao)
        == esperado
    )
