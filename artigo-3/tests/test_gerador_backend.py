"""Testes do backend do gerador sem baixar o modelo (#52)."""

from __future__ import annotations

import pytest

from src.config import CONFIG
from src.gerador_backend import GeradorHF


class _Tensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))

    def to(self, _device):
        return self

    def __getitem__(self, i):
        return self.ids if i == 0 else None


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.mensagens = None

    def apply_chat_template(self, mensagens, add_generation_prompt, return_tensors, return_dict):
        self.mensagens = mensagens
        assert add_generation_prompt and return_tensors == "pt" and return_dict
        return {"input_ids": _Tensor([11, 12, 13]), "attention_mask": _Tensor([1, 1, 1])}

    def decode(self, ids, skip_special_tokens):
        return " resposta " + "-".join(map(str, ids)) + " "


class _Modelo:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, entrada, **kwargs):
        self.kwargs = kwargs
        return [_Tensor(entrada.ids + [7, 8]).ids]


def _gerador():
    return GeradorHF(nome="m", revisao="r", tokenizer=_Tokenizer(), model=_Modelo())


def test_gera_so_os_tokens_novos_de_forma_gulosa():
    g = _gerador()
    s = g.gerar("prompt", temperatura=0.0, max_tokens=32)
    assert s.texto == "resposta 7-8"
    assert (s.tokens_entrada, s.tokens_saida) == (3, 2)
    assert g.model.kwargs["do_sample"] is False and g.model.kwargs["max_new_tokens"] == 32
    assert g.tokenizer.mensagens == [{"role": "user", "content": "prompt"}]


def test_temperatura_diferente_de_zero_e_recusada():
    with pytest.raises(ValueError, match="gulosa"):
        _gerador().gerar("p", temperatura=0.7, max_tokens=8)


def test_gerador_congelado_no_config():
    assert CONFIG.geracao.modelo_gerador == "Qwen/Qwen2.5-3B-Instruct"
    assert len(CONFIG.geracao.revisao_gerador) == 40  # hash completo do commit no Hub
