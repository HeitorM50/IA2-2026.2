"""Embeddings normalizados, índice FAISS exato e cache auditável (#50)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.chunking import Chunk, Tokenizador, gerar_chunks, hash_entrada, parametros_janela
from src.config import CONFIG, PENDENTE, ExperimentoConfig
from src.contratos import Trecho
from src.corpus import Pagina, carregar_paginas

ARQUIVOS_CACHE = ("chunks.jsonl", "embeddings.npy", "index.faiss", "manifest.json")


class IndiceError(RuntimeError):
    pass


class Codificador(Protocol):
    nome: str
    revisao: str

    def codificar_documentos(self, textos: Sequence[str]) -> np.ndarray: ...

    def codificar_consulta(self, texto: str) -> np.ndarray: ...


class CodificadorSentenceTransformers:
    """Backend real, carregado apenas quando a execução precisa do modelo."""

    def __init__(self, cfg=CONFIG.recuperacao):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise IndiceError(
                "sentence-transformers não instalado; execute `pip install -r requirements.txt`"
            ) from e
        self.nome = cfg.modelo_embedding
        self.revisao = cfg.revisao_embedding
        self.prefixo_documento = cfg.prefixo_documento
        self.prefixo_consulta = cfg.prefixo_consulta
        self.batch_size = cfg.batch_size
        self.modelo = SentenceTransformer(self.nome, revision=self.revisao)
        self.tokenizador = self.modelo.tokenizer

    def _codificar(self, textos: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.modelo.encode(
                list(textos),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                precision="float32",
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def codificar_documentos(self, textos: Sequence[str]) -> np.ndarray:
        return self._codificar([self.prefixo_documento + t for t in textos])

    def codificar_consulta(self, texto: str) -> np.ndarray:
        return self._codificar([self.prefixo_consulta + texto])[0]


def _modulo_faiss():
    try:
        import faiss
    except ImportError as e:
        raise IndiceError("faiss-cpu não instalado; execute `pip install -r requirements.txt`") from e
    return faiss


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def normalizar_l2(vetores: np.ndarray) -> np.ndarray:
    """Converte para matriz float32 e normaliza cada linha para norma unitária."""
    matriz = np.asarray(vetores, dtype=np.float32)
    if matriz.ndim == 1:
        matriz = matriz.reshape(1, -1)
    if matriz.ndim != 2 or matriz.shape[0] == 0 or matriz.shape[1] == 0:
        raise IndiceError(f"matriz de embeddings inválida: shape={matriz.shape}")
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    if np.any(normas == 0) or not np.all(np.isfinite(normas)):
        raise IndiceError("embedding nulo ou não finito não pode ser normalizado")
    return np.ascontiguousarray(matriz / normas, dtype=np.float32)


def _json_deterministico(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _hash_manifesto_corpus(cfg: ExperimentoConfig) -> str | None:
    caminho = cfg.corpus.manifesto_path
    return sha256_arquivo(caminho) if caminho.is_file() else None


def chave_cache(config_id: str, paginas: Sequence[Pagina], cfg: ExperimentoConfig = CONFIG) -> str:
    parametros = parametros_janela(config_id, cfg.chunking)
    payload = {
        "config_id": config_id,
        "parametros": None if parametros is None else {
            "tamanho_tokens": parametros.tamanho_tokens,
            "sobreposicao_tokens": parametros.sobreposicao_tokens,
            "passo_tokens": parametros.passo_tokens,
        },
        "entrada_sha256": hash_entrada(paginas),
        "corpus_manifesto_sha256": _hash_manifesto_corpus(cfg),
        "modelo_embedding": cfg.recuperacao.modelo_embedding,
        "revisao_embedding": cfg.recuperacao.revisao_embedding,
        "tokenizador": cfg.chunking.tokenizador,
        "versao_chunker": cfg.chunking.versao_chunker,
        "prefixo_documento": cfg.recuperacao.prefixo_documento,
        "prefixo_consulta": cfg.recuperacao.prefixo_consulta,
        "indice_faiss": cfg.recuperacao.indice_faiss,
    }
    return hashlib.sha256(_json_deterministico(payload)).hexdigest()


def _chunks_jsonl(chunks: Sequence[Chunk]) -> str:
    return "".join(
        json.dumps(c.como_dict(), ensure_ascii=False, sort_keys=True) + "\n" for c in chunks
    )


def _carregar_chunks(caminho: Path) -> list[Chunk]:
    return [
        Chunk(**json.loads(linha))
        for linha in caminho.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]


def _cache_valido(destino: Path, cache_key: str) -> bool:
    if not all((destino / nome).is_file() for nome in ARQUIVOS_CACHE):
        return False
    try:
        manifesto = json.loads((destino / "manifest.json").read_text(encoding="utf-8"))
        if manifesto.get("cache_key") != cache_key:
            return False
        hashes = manifesto["sha256_artefatos"]
        return all(
            sha256_arquivo(destino / nome) == hashes[nome]
            for nome in ("chunks.jsonl", "embeddings.npy", "index.faiss")
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


@dataclass
class RecuperadorFaiss:
    config_id: str
    indice: Any
    chunks: list[Chunk]
    codificador: Codificador
    cache_dir: Path
    usou_cache: bool

    def buscar(self, pergunta: str, k: int) -> list[Trecho]:
        if k <= 0 or not self.chunks:
            return []
        consulta = normalizar_l2(self.codificador.codificar_consulta(pergunta))
        limite = min(k, len(self.chunks))
        scores, posicoes = self.indice.search(consulta, limite)
        candidatos = []
        for score, posicao in zip(scores[0], posicoes[0]):
            indice = int(posicao)
            if indice < 0:
                continue
            candidatos.append((float(score), indice, self.chunks[indice]))
        candidatos.sort(key=lambda x: (-x[0], x[1]))
        return [
            Trecho(
                chunk_id=chunk.chunk_id,
                texto=chunk.texto,
                score=score,
                aula=chunk.aula,
                pagina_inicio=chunk.pagina_inicio,
                pagina_fim=chunk.pagina_fim,
                token_inicio=chunk.token_inicio,
                token_fim=chunk.token_fim,
                n_tokens=chunk.n_tokens,
                origem=chunk.origem,
                parte=chunk.parte,
                titulo=chunk.titulo,
            )
            for score, _, chunk in candidatos
        ]


def _validar_backend(codificador: Codificador, cfg: ExperimentoConfig) -> None:
    esperado = (cfg.recuperacao.modelo_embedding, cfg.recuperacao.revisao_embedding)
    observado = (codificador.nome, codificador.revisao)
    if PENDENTE in esperado:
        raise IndiceError("modelo e revisão de embeddings precisam estar congelados")
    if observado != esperado:
        raise IndiceError(
            f"codificador {observado[0]}@{observado[1]} difere do protocolo "
            f"{esperado[0]}@{esperado[1]}"
        )


def _carregar_cache(
    destino: Path,
    config_id: str,
    codificador: Codificador,
    faiss_mod: Any,
) -> RecuperadorFaiss:
    chunks = _carregar_chunks(destino / "chunks.jsonl")
    embeddings = np.load(destino / "embeddings.npy", allow_pickle=False)
    indice = faiss_mod.read_index(str(destino / "index.faiss"))
    if embeddings.shape[0] != len(chunks) or int(indice.ntotal) != len(chunks):
        raise IndiceError("cache inconsistente: quantidades de chunks, vetores e índice divergem")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise IndiceError("cache inconsistente: embeddings não estão normalizados")
    return RecuperadorFaiss(config_id, indice, chunks, codificador, destino, True)


def _publicar_cache(temporario: Path, destino: Path) -> None:
    if destino.exists():
        antigo = destino.with_name(destino.name + f".substituido-{uuid.uuid4().hex}")
        os.replace(destino, antigo)
        try:
            os.replace(temporario, destino)
        except BaseException:
            os.replace(antigo, destino)
            raise
        shutil.rmtree(antigo)
    else:
        os.replace(temporario, destino)


def construir_recuperador(
    config_id: str,
    *,
    cfg: ExperimentoConfig = CONFIG,
    paginas: Sequence[Pagina] | None = None,
    tokenizador: Tokenizador | None = None,
    codificador: Codificador | None = None,
    faiss_mod: Any | None = None,
    forcar: bool = False,
) -> RecuperadorFaiss:
    """Constrói ou carrega um índice compatível com ``src.run``."""
    if cfg.recuperacao.indice_faiss != "IndexFlatIP":
        raise IndiceError("esta implementação exige indice_faiss='IndexFlatIP'")
    paginas = list(carregar_paginas(cfg.corpus) if paginas is None else paginas)
    if not paginas:
        raise IndiceError("corpus extraído vazio; execute `python -m src.corpus`")
    codificador = codificador or CodificadorSentenceTransformers(cfg.recuperacao)
    _validar_backend(codificador, cfg)
    faiss_mod = faiss_mod or _modulo_faiss()
    cache_key = chave_cache(config_id, paginas, cfg)
    destino = cfg.recuperacao.cache_dir / config_id / cache_key
    if not forcar and _cache_valido(destino, cache_key):
        return _carregar_cache(destino, config_id, codificador, faiss_mod)

    if tokenizador is None:
        tokenizador = getattr(codificador, "tokenizador", None)
    if tokenizador is None:
        raise IndiceError("tokenizador é obrigatório para construir um cache novo")

    inicio_total = time.perf_counter()
    inicio = time.perf_counter()
    chunks = gerar_chunks(paginas, config_id, tokenizador, cfg.chunking)
    tempo_chunking = time.perf_counter() - inicio
    if not chunks:
        raise IndiceError(f"{config_id}: nenhum chunk não vazio foi produzido")

    inicio = time.perf_counter()
    embeddings = normalizar_l2(
        codificador.codificar_documentos([chunk.texto for chunk in chunks])
    )
    tempo_embeddings = time.perf_counter() - inicio
    if embeddings.shape[0] != len(chunks):
        raise IndiceError("codificador devolveu quantidade diferente de embeddings")

    inicio = time.perf_counter()
    indice = faiss_mod.IndexFlatIP(int(embeddings.shape[1]))
    indice.add(embeddings)
    tempo_faiss = time.perf_counter() - inicio

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = Path(tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=destino.parent))
    try:
        (temporario / "chunks.jsonl").write_text(_chunks_jsonl(chunks), encoding="utf-8")
        with (temporario / "embeddings.npy").open("wb") as arquivo:
            np.save(arquivo, embeddings, allow_pickle=False)
        faiss_mod.write_index(indice, str(temporario / "index.faiss"))
        parametros = parametros_janela(config_id, cfg.chunking)
        manifesto = {
            "cache_key": cache_key,
            "config_id": config_id,
            "entrada_sha256": hash_entrada(paginas),
            "corpus_manifesto_sha256": _hash_manifesto_corpus(cfg),
            "modelo_embedding": cfg.recuperacao.modelo_embedding,
            "revisao_embedding": cfg.recuperacao.revisao_embedding,
            "tokenizador": cfg.chunking.tokenizador,
            "versao_chunker": cfg.chunking.versao_chunker,
            "indice_faiss": cfg.recuperacao.indice_faiss,
            "tamanho_tokens": parametros.tamanho_tokens if parametros else None,
            "sobreposicao_tokens": parametros.sobreposicao_tokens if parametros else None,
            "passo_tokens": parametros.passo_tokens if parametros else None,
            "n_chunks": len(chunks),
            "n_tokens_total": sum(c.n_tokens for c in chunks),
            "dimensao_embedding": int(embeddings.shape[1]),
            "chunks_acima_limite_modelo": sum(
                c.n_tokens > cfg.recuperacao.max_tokens_modelo for c in chunks
            ),
            "tamanho_indice_bytes": (temporario / "index.faiss").stat().st_size,
            "tempo_chunking_segundos": tempo_chunking,
            "tempo_embeddings_segundos": tempo_embeddings,
            "tempo_faiss_segundos": tempo_faiss,
            "tempo_indexacao_segundos": time.perf_counter() - inicio_total,
            "sha256_artefatos": {
                nome: sha256_arquivo(temporario / nome)
                for nome in ("chunks.jsonl", "embeddings.npy", "index.faiss")
            },
        }
        (temporario / "manifest.json").write_text(
            json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publicar_cache(temporario, destino)
    except BaseException:
        if temporario.exists():
            shutil.rmtree(temporario)
        raise
    return RecuperadorFaiss(config_id, indice, chunks, codificador, destino, False)
