"""Testes de tamanho, sobreposição, origem e determinismo dos chunks (#50)."""

from __future__ import annotations

import re

import pytest

from src.chunking import gerar_chunks, hash_entrada, parametros_janela
from src.corpus import Pagina


class TokenizadorEspaco:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverso: dict[int, str] = {}

    def encode(self, texto, *, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        for token in re.findall(r"\S+", texto):
            if token not in self.vocab:
                indice = len(self.vocab) + 1
                self.vocab[token] = indice
                self.inverso[indice] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(self.inverso[t] for t in tokens)


def _pagina(aula: int, pagina: int, n_tokens: int) -> Pagina:
    parte = "V" if aula <= 17 else "VI" if aula <= 20 else "VII"
    return Pagina(
        aula=aula,
        parte=parte,
        titulo=f"Aula {aula}",
        pagina=pagina,
        origem=f"Aula{aula}.pdf",
        sha256_pdf=f"sha-{aula}",
        texto=" ".join(f"a{aula}p{pagina}t{i}" for i in range(n_tokens)),
        n_caracteres=n_tokens * 5,
        suspeita=None,
    )


@pytest.mark.parametrize(
    "config_id,tamanho,sobreposicao,passo",
    [
        ("c128-o0", 128, 0, 128),
        ("c128-o20", 128, 26, 102),
        ("c256-o0", 256, 0, 256),
        ("c256-o20", 256, 51, 205),
        ("c512-o0", 512, 0, 512),
        ("c512-o20", 512, 102, 410),
    ],
)
def test_limite_e_sobreposicao_em_tokens(config_id, tamanho, sobreposicao, passo):
    parametros = parametros_janela(config_id)
    assert (parametros.tamanho_tokens, parametros.sobreposicao_tokens, parametros.passo_tokens) == (
        tamanho, sobreposicao, passo
    )
    chunks = gerar_chunks([_pagina(14, 1, 1400)], config_id, TokenizadorEspaco())
    assert all(0 < c.n_tokens <= tamanho for c in chunks)
    for anterior, atual in zip(chunks, chunks[1:]):
        assert anterior.token_fim - atual.token_inicio == sobreposicao


def test_chunks_nunca_atravessam_aulas_e_preservam_origem():
    paginas = [_pagina(14, 1, 90), _pagina(14, 2, 90), _pagina(15, 1, 90)]
    chunks = gerar_chunks(paginas, "c128-o20", TokenizadorEspaco())
    assert {c.aula for c in chunks} == {14, 15}
    assert all(c.origem == f"Aula{c.aula}.pdf" for c in chunks)
    assert all(c.parte == "V" and c.titulo == f"Aula {c.aula}" for c in chunks)
    assert any(c.pagina_inicio == 1 and c.pagina_fim == 2 for c in chunks if c.aula == 14)
    assert all(not (c.aula == 14 and c.pagina_fim > 2) for c in chunks)


def test_todas_as_configuracoes_recebem_o_mesmo_texto_de_entrada():
    paginas = [_pagina(14, 1, 20), _pagina(15, 1, 20)]
    digest = hash_entrada(paginas)
    for config_id in ("slide", "c128-o0", "c128-o20", "c256-o0", "c256-o20", "c512-o0", "c512-o20"):
        gerar_chunks(paginas, config_id, TokenizadorEspaco())
        assert hash_entrada(paginas) == digest


def test_mesma_entrada_gera_mesmos_ids_textos_e_intervalos():
    paginas = [_pagina(18, 1, 300), _pagina(18, 2, 80)]
    a = gerar_chunks(paginas, "c128-o20", TokenizadorEspaco())
    b = gerar_chunks(paginas, "c128-o20", TokenizadorEspaco())
    assert a == b


def test_slide_gera_um_chunk_por_pagina_nao_vazia():
    tokenizador = TokenizadorEspaco()
    paginas = [_pagina(21, 1, 20), _pagina(21, 2, 30), _pagina(21, 3, 0)]
    chunks = gerar_chunks(paginas, "slide", tokenizador)
    assert [(c.pagina_inicio, c.pagina_fim, c.n_tokens) for c in chunks] == [
        (1, 1, 20),
        (2, 2, 30),
    ]
