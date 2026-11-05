"""Testes do índice, busca determinística e cache por conteúdo (#50)."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from src.config import CONFIG
from src.corpus import Pagina
from src.indice import IndiceError, construir_recuperador


class TokenizadorEspaco:
    def __init__(self):
        self.vocab = {}
        self.inverso = {}

    def encode(self, texto, *, add_special_tokens=False):
        ids = []
        for token in re.findall(r"\S+", texto):
            if token not in self.vocab:
                i = len(self.vocab) + 1
                self.vocab[token] = i
                self.inverso[i] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(self.inverso[t] for t in tokens)


class TokenizadorComEspeciais(TokenizadorEspaco):
    def encode(self, texto, *, add_special_tokens=False):
        ids = super().encode(texto, add_special_tokens=False)
        return [-1, *ids, -2] if add_special_tokens else ids


class CodificadorFalso:
    nome = "modelo-teste"
    revisao = "rev-1"

    @staticmethod
    def _vetor(texto):
        texto = texto.casefold()
        return [
            5.0 if "atenção" in texto else 1.0,
            5.0 if "reforço" in texto else 1.0,
            5.0 if "fairness" in texto else 1.0,
            float(len(texto) % 7 + 1),
        ]

    def codificar_documentos(self, textos):
        return np.asarray([self._vetor(t) for t in textos], dtype=np.float64)

    def codificar_consulta(self, texto):
        return np.asarray(self._vetor(texto), dtype=np.float64)


class IndexFlatIPFalso:
    def __init__(self, dimensao):
        self.d = dimensao
        self.vetores = np.empty((0, dimensao), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vetores)

    def add(self, vetores):
        self.vetores = np.asarray(vetores, dtype=np.float32).copy()

    def search(self, consultas, k):
        produtos = np.asarray(consultas, dtype=np.float32) @ self.vetores.T
        scores = np.empty((len(consultas), k), dtype=np.float32)
        indices = np.empty((len(consultas), k), dtype=np.int64)
        for linha, valores in enumerate(produtos):
            ordem = np.lexsort((np.arange(len(valores)), -valores))[:k]
            scores[linha] = valores[ordem]
            indices[linha] = ordem
        return scores, indices


class FaissFalso:
    IndexFlatIP = IndexFlatIPFalso

    @staticmethod
    def write_index(indice, caminho):
        with open(caminho, "wb") as arquivo:
            np.save(arquivo, indice.vetores, allow_pickle=False)

    @staticmethod
    def read_index(caminho):
        with open(caminho, "rb") as arquivo:
            vetores = np.load(arquivo, allow_pickle=False)
        indice = IndexFlatIPFalso(vetores.shape[1])
        indice.add(vetores)
        return indice


def _pagina(aula, pagina, texto):
    return Pagina(
        aula=aula,
        parte="V",
        titulo=f"Aula {aula}",
        pagina=pagina,
        origem=f"Aula{aula}.pdf",
        sha256_pdf=f"sha-{aula}",
        texto=texto,
        n_caracteres=len(texto),
        suspeita=None,
    )


@pytest.fixture
def ambiente(tmp_path):
    cfg = replace(
        CONFIG,
        corpus=replace(CONFIG.corpus, manifesto_path=tmp_path / "manifesto-ausente.json"),
        chunking=replace(CONFIG.chunking, tokenizador="tokenizador-teste"),
        recuperacao=replace(
            CONFIG.recuperacao,
            modelo_embedding="modelo-teste",
            revisao_embedding="rev-1",
            cache_dir=tmp_path / "cache",
        ),
    )
    paginas = [
        _pagina(14, 1, "atenção transforma consultas em representações"),
        _pagina(14, 2, "aprendizado por reforço usa recompensas"),
        _pagina(15, 1, "fairness compara resultados entre grupos"),
    ]
    return cfg, paginas


def _construir(cfg, paginas, **kw):
    return construir_recuperador(
        "slide",
        cfg=cfg,
        paginas=paginas,
        tokenizador=TokenizadorEspaco(),
        codificador=CodificadorFalso(),
        faiss_mod=FaissFalso,
        **kw,
    )


def test_vetores_normalizados_e_busca_retorna_metadados(ambiente):
    cfg, paginas = ambiente
    recuperador = _construir(cfg, paginas)
    assert np.allclose(np.linalg.norm(recuperador.indice.vetores, axis=1), 1.0)
    resultados = recuperador.buscar("atenção", 2)
    assert resultados[0].texto.startswith("atenção")
    assert resultados[0].score >= resultados[1].score
    assert resultados[0].aula == 14
    assert resultados[0].origem == "Aula14.pdf"
    assert resultados[0].pagina_inicio == resultados[0].pagina_fim == 1
    assert resultados[0].parte == "V" and resultados[0].titulo == "Aula 14"


def test_manifesto_registra_contagem_tamanho_e_tempos(ambiente):
    cfg, paginas = ambiente
    recuperador = _construir(cfg, paginas)
    manifesto = json.loads((recuperador.cache_dir / "manifest.json").read_text())
    assert manifesto["n_chunks"] == 3
    assert manifesto["tamanho_indice_bytes"] > 0
    assert manifesto["dimensao_embedding"] == 4
    for campo in (
        "tempo_chunking_segundos", "tempo_embeddings_segundos",
        "tempo_faiss_segundos", "tempo_indexacao_segundos",
    ):
        assert manifesto[campo] >= 0


def test_manifesto_conta_prefixo_e_tokens_especiais_no_limite(tmp_path):
    cfg = replace(
        CONFIG,
        corpus=replace(CONFIG.corpus, manifesto_path=tmp_path / "manifesto-ausente.json"),
        chunking=replace(CONFIG.chunking, tokenizador="tokenizador-teste"),
        recuperacao=replace(
            CONFIG.recuperacao,
            modelo_embedding="modelo-teste",
            revisao_embedding="rev-1",
            cache_dir=tmp_path / "cache",
            max_tokens_modelo=512,
        ),
    )
    pagina = _pagina(14, 1, " ".join(f"t{i}" for i in range(512)))

    recuperador = construir_recuperador(
        "c512-o0",
        cfg=cfg,
        paginas=[pagina],
        tokenizador=TokenizadorComEspeciais(),
        codificador=CodificadorFalso(),
        faiss_mod=FaissFalso,
    )

    manifesto = json.loads((recuperador.cache_dir / "manifest.json").read_text())
    assert manifesto["n_chunks"] == 1
    assert manifesto["max_tokens_modelo"] == 512
    assert manifesto["contagem_limite_inclui_prefixo_e_tokens_especiais"] is True
    assert manifesto["chunks_acima_limite_modelo"] == 1


def test_segunda_construcao_reutiliza_cache_e_preserva_busca(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    esperado = [(r.chunk_id, r.score) for r in primeiro.buscar("fairness", 3)]
    segundo = _construir(cfg, paginas)
    observado = [(r.chunk_id, r.score) for r in segundo.buscar("fairness", 3)]
    assert primeiro.usou_cache is False and segundo.usou_cache is True
    assert observado == esperado


def test_reconstrucao_forcada_produz_mesmo_resultado(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    esperado = [(r.chunk_id, r.score) for r in primeiro.buscar("reforço", 3)]
    reconstruido = _construir(cfg, paginas, forcar=True)
    assert reconstruido.usou_cache is False
    assert [(r.chunk_id, r.score) for r in reconstruido.buscar("reforço", 3)] == esperado


def test_cache_corrompido_e_reconstruido(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    (primeiro.cache_dir / "index.faiss").write_bytes(b"corrompido")
    segundo = _construir(cfg, paginas)
    assert segundo.usou_cache is False
    assert segundo.buscar("atenção", 1)[0].pagina_inicio == 1


def test_mudanca_do_corpus_gera_outra_chave(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    alteradas = list(paginas)
    alteradas[0] = replace(alteradas[0], texto=alteradas[0].texto + " alterado")
    segundo = _construir(cfg, alteradas)
    assert primeiro.cache_dir != segundo.cache_dir


def test_mudanca_da_configuracao_gera_outra_chave(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    segundo = construir_recuperador(
        "c128-o20", cfg=cfg, paginas=paginas, tokenizador=TokenizadorEspaco(),
        codificador=CodificadorFalso(), faiss_mod=FaissFalso,
    )
    assert primeiro.cache_dir != segundo.cache_dir


def test_mudanca_de_modelo_gera_outra_chave(ambiente):
    cfg, paginas = ambiente
    primeiro = _construir(cfg, paginas)
    outro_cfg = replace(
        cfg,
        recuperacao=replace(cfg.recuperacao, modelo_embedding="outro-modelo"),
    )
    outro_codificador = CodificadorFalso()
    outro_codificador.nome = "outro-modelo"
    segundo = construir_recuperador(
        "slide", cfg=outro_cfg, paginas=paginas, tokenizador=TokenizadorEspaco(),
        codificador=outro_codificador, faiss_mod=FaissFalso,
    )
    assert primeiro.cache_dir != segundo.cache_dir


def test_backend_diferente_do_protocolo_e_rejeitado(ambiente):
    cfg, paginas = ambiente
    codificador = CodificadorFalso()
    codificador.revisao = "outra"
    with pytest.raises(IndiceError, match="difere do protocolo"):
        construir_recuperador(
            "slide", cfg=cfg, paginas=paginas, tokenizador=TokenizadorEspaco(),
            codificador=codificador, faiss_mod=FaissFalso,
        )


def test_faiss_real_persiste_e_reconstroi_mesmo_ranking(ambiente):
    faiss = pytest.importorskip("faiss")
    cfg, paginas = ambiente
    primeiro = construir_recuperador(
        "slide", cfg=cfg, paginas=paginas, tokenizador=TokenizadorEspaco(),
        codificador=CodificadorFalso(), faiss_mod=faiss,
    )
    esperado = [(r.chunk_id, r.score) for r in primeiro.buscar("atenção", 3)]
    segundo = construir_recuperador(
        "slide", cfg=cfg, paginas=paginas, tokenizador=TokenizadorEspaco(),
        codificador=CodificadorFalso(), faiss_mod=faiss,
    )
    assert type(segundo.indice).__name__ == "IndexFlatIP"
    assert segundo.usou_cache is True
    assert [(r.chunk_id, r.score) for r in segundo.buscar("atenção", 3)] == esperado


def test_codificador_padrao_carrega_o_modelo_uma_vez(monkeypatch):
    import src.indice as indice

    criados = []
    monkeypatch.setattr(indice, "CodificadorSentenceTransformers", lambda cfg: criados.append(cfg) or object())
    indice.codificador_padrao.cache_clear()
    try:
        a = indice.codificador_padrao(CONFIG.recuperacao)
        b = indice.codificador_padrao(CONFIG.recuperacao)
    finally:
        indice.codificador_padrao.cache_clear()
    assert a is b and len(criados) == 1
