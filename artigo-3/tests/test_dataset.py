"""Testes do validador do dataset (#49)."""

from __future__ import annotations

import json

import pytest
from test_run import _dataset_completo, _item

from src.dataset import DatasetError, carregar


def _gravar(tmp_path, itens):
    p = tmp_path / "perguntas.jsonl"
    p.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in itens))
    return p


def test_dataset_do_protocolo_carrega_e_filtra_split(tmp_path):
    p = _gravar(tmp_path, _dataset_completo())
    assert len(carregar(p)) == 60
    assert len(carregar(p, split="dev")) == 15
    assert {it.split for it in carregar(p, split="test")} == {"test"}


@pytest.mark.parametrize(
    "mutacao, trecho",
    [
        (lambda d: d.__setitem__(1, d[0] | {"split": d[1]["split"]}), "IDs repetidos"),
        (lambda d: d[0].__setitem__("evidencias_validas", []), "sem evidência"),
        (lambda d: d[-1].__setitem__("evidencias_validas", [{"aula": 14, "pagina": 1}]), "não respondível com evidência"),
        (lambda d: d[0].__setitem__("split", "treino"), "split"),
        (lambda d: d[20].__setitem__("pergunta", d[0]["pergunta"]), "perguntas repetidas"),
        (lambda d: d.pop(), "split esperado"),
    ],
)
def test_erros_do_dataset(tmp_path, mutacao, trecho):
    itens = _dataset_completo()
    mutacao(itens)
    with pytest.raises(DatasetError, match=trecho):
        carregar(_gravar(tmp_path, itens))


def test_modo_nao_estrito_aceita_dataset_parcial(tmp_path):
    p = _gravar(tmp_path, [_item(0, "dev"), _item(1, "dev")])
    assert len(carregar(p, estrito=False)) == 2
