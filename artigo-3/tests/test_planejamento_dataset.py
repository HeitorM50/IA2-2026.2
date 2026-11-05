"""Testes do congelamento prévio dos IDs e splits (#49)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.planejamento_dataset import (
    PlanejamentoError,
    carregar_plano,
    validar_contra_dataset,
    validar_plano,
)


def test_planejamento_versionado_cumpre_protocolo():
    itens = carregar_plano()
    validar_plano(itens)
    assert len(itens) == 60
    assert len({i.id for i in itens}) == 60


def test_revisao_cruzada_exige_pessoas_diferentes():
    itens = carregar_plano()
    itens[0] = replace(itens[0], revisor=itens[0].autor)
    with pytest.raises(PlanejamentoError, match="pessoas diferentes"):
        validar_plano(itens)


def test_dataset_nao_pode_mudar_split_planejado():
    from src.dataset import Item

    plano = carregar_plano()
    p = plano[0]
    item = Item(
        id=p.id,
        split="test",
        parte=p.parte,
        aula=p.aula,
        pagina=1,
        categoria=p.categoria,
        pergunta="Pergunta?",
        resposta_referencia="Resposta.",
        evidencias_validas=((p.aula, 1),),
        respondivel=True,
    )
    with pytest.raises(PlanejamentoError, match="metadados alterados"):
        validar_contra_dataset([p], [item])
