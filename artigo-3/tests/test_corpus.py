"""Testes da extração do corpus (#48).

Os PDFs reais não estão no repositório, então os testes geram PDFs sintéticos
com o PyMuPDF. O que se protege aqui é o vínculo página → origem e o
determinismo do manifesto, que a execução canônica usa como verificação.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.config import CorpusConfig
from src.corpus import (
    CorpusError,
    carregar_paginas,
    descobrir_pdfs,
    extrair_corpus,
    normalizar,
    validar_catalogo,
)

AULAS = tuple(range(14, 24))
TEXTO_LONGO = "Transformers usam atenção multi-cabeça para modelar dependências."


def _pdf(caminho: Path, paginas: list[str]) -> None:
    doc = fitz.open()
    for texto in paginas:
        page = doc.new_page()
        if texto:
            page.insert_text((72, 72), texto)
    doc.save(caminho)
    doc.close()


def _cfg(tmp_path: Path, **kw) -> CorpusConfig:
    base = CorpusConfig(
        corpus_dir=tmp_path / "corpus",
        extraido_dir=tmp_path / "extraido",
        manifesto_path=tmp_path / "manifesto.json",
        aulas=AULAS,
        catalogo={a: ("V", f"Aula {a}") for a in AULAS},
    )
    return replace(base, **kw)


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusConfig:
    cfg = _cfg(tmp_path)
    cfg.corpus_dir.mkdir()
    for a in AULAS:
        # página 2 vazia na aula 14 para exercitar a marcação de suspeita
        paginas = [f"{TEXTO_LONGO} Aula {a}.", "" if a == 14 else f"{TEXTO_LONGO} p2"]
        _pdf(cfg.corpus_dir / f"Aula{a}-slides.pdf", paginas)
    return cfg


def test_normalizar_preserva_quebras_e_colapsa_espacos():
    bruto = "  Atenção\t\tmulti   cabeça \n\n\n\nSoftmax  escalado  "
    assert normalizar(bruto) == "Atenção multi cabeça\n\nSoftmax escalado"


def test_diretorio_ausente_gera_erro_explicito(tmp_path):
    with pytest.raises(CorpusError, match="não encontrado"):
        descobrir_pdfs(_cfg(tmp_path))


def test_aula_faltando_e_nome_sem_numero(corpus):
    (corpus.corpus_dir / "Aula23-slides.pdf").unlink()
    _pdf(corpus.corpus_dir / "apostila.pdf", [TEXTO_LONGO])
    with pytest.raises(CorpusError) as e:
        descobrir_pdfs(corpus)
    assert "[23]" in str(e.value)
    assert "apostila.pdf" in str(e.value)


def test_aula_duplicada(corpus):
    _pdf(corpus.corpus_dir / "aula_14_v2.pdf", [TEXTO_LONGO])
    with pytest.raises(CorpusError, match="mais de um PDF"):
        descobrir_pdfs(corpus)


@pytest.mark.parametrize(
    "catalogo, trecho",
    [
        ({14: ("VIII", "x")}, "parte"),
        ({30: ("V", "x")}, "fora"),
        ({14: ("V", "  ")}, "título vazio"),
    ],
)
def test_metadado_invalido(tmp_path, catalogo, trecho):
    with pytest.raises(CorpusError, match=trecho):
        validar_catalogo(_cfg(tmp_path, catalogo=catalogo))


def test_extracao_mantem_origem_e_marca_pagina_vazia(corpus):
    manifesto = extrair_corpus(corpus)
    assert [a["aula"] for a in manifesto["arquivos"]] == list(AULAS)
    assert manifesto["catalogo_completo"] is True

    paginas = carregar_paginas(corpus)
    assert len(paginas) == 2 * len(AULAS)
    p = next(p for p in paginas if p.aula == 15 and p.pagina == 1)
    assert p.origem == "Aula15-slides.pdf"
    assert p.parte == "V" and p.titulo == "Aula 15"
    assert "Aula 15" in p.texto and p.suspeita is None

    aula14 = next(a for a in manifesto["arquivos"] if a["aula"] == 14)
    assert aula14["paginas_suspeitas"] == [{"pagina": 2, "motivo": "sem_texto"}]


def test_mesma_entrada_mesmo_manifesto_e_textos(corpus, tmp_path):
    m1 = extrair_corpus(corpus)
    bytes1 = corpus.manifesto_path.read_bytes()
    textos1 = {p.name: p.read_bytes() for p in corpus.extraido_dir.iterdir()}

    outro = replace(
        corpus,
        extraido_dir=tmp_path / "extraido2",
        manifesto_path=tmp_path / "manifesto2.json",
    )
    m2 = extrair_corpus(outro)
    assert m1 == m2
    assert outro.manifesto_path.read_bytes() == bytes1
    assert {p.name: p.read_bytes() for p in outro.extraido_dir.iterdir()} == textos1


def test_catalogo_incompleto_fica_registrado(corpus):
    cfg = replace(corpus, catalogo={})
    manifesto = extrair_corpus(cfg)
    assert manifesto["catalogo_completo"] is False
    assert json.loads(cfg.manifesto_path.read_text())["catalogo_completo"] is False
