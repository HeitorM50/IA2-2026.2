"""Configuração única do protocolo do Artigo 3.

O artigo compara seis configurações de *chunking* em um sistema de geração
aumentada por recuperação (RAG) sobre as aulas 14–23 de Inteligência Artificial II.
A comparação só vale se as seis passarem pelo mesmo corpus, pelo mesmo modelo de
embedding, pelo mesmo gerador e pelo mesmo conjunto de perguntas. Por isso todo
valor experimental mora aqui, e nenhum outro módulo ou notebook pode redefini-lo.

Valores marcados como ``PENDENTE`` dependem das decisões da issue #46 e precisam
ser congelados antes da execução canônica; `validar_congelamento()` falha enquanto
algum deles estiver em aberto.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ARTIGO = Path(__file__).resolve().parent.parent
SRC = ARTIGO / "src"

PENDENTE = "PENDENTE"


@dataclass(frozen=True)
class CorpusConfig:
    """Origem e identificação dos dez PDFs das aulas."""

    # Os PDFs não são versionados (direitos autorais em aberto, #46).
    # A variável de ambiente permite apontar para a cópia local de cada autor.
    corpus_dir: Path = Path(
        os.environ.get("IA2_CORPUS_DIR", str(ARTIGO / "data" / "corpus"))
    )
    extraido_dir: Path = ARTIGO / "data" / "extraido"
    manifesto_path: Path = ARTIGO / "data" / "manifesto-corpus.json"

    aulas: tuple[int, ...] = tuple(range(14, 24))
    # Número da aula extraído do nome do arquivo, ex.: "Aula14-Transformers.pdf".
    padrao_nome_arquivo: str = r"(?i)aula[\s_-]*0*(\d{1,2})"

    # aula -> (parte, título). Preencher com os dados dos PDFs (#48).
    # Enquanto vazio, a extração roda, mas o manifesto marca o catálogo como pendente.
    catalogo: dict[int, tuple[str, str]] = field(default_factory=dict)

    # Página com menos caracteres que isso é registrada como suspeita
    # (slide só com figura, fórmula renderizada como imagem, etc.).
    min_caracteres_pagina: int = 40


@dataclass(frozen=True)
class ChunkingConfig:
    """Grade de configurações comparadas (#50). Tamanhos em tokens, não caracteres."""

    tamanhos_tokens: tuple[int, ...] = (128, 256, 512)
    sobreposicoes: tuple[float, ...] = (0.0, 0.2)
    # O tokenizador é o do próprio modelo de embedding, para que "128 tokens"
    # signifique o que o modelo de fato enxerga.
    tokenizador: str = PENDENTE

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(
            f"c{t}-o{round(s * 100)}"
            for t in self.tamanhos_tokens
            for s in self.sobreposicoes
        )


@dataclass(frozen=True)
class RecuperacaoConfig:
    """Embedding, índice e métricas de recuperação (#50, #51)."""

    modelo_embedding: str = PENDENTE
    revisao_embedding: str = PENDENTE
    # Vetores normalizados + produto interno = similaridade de cosseno.
    indice_faiss: str = "IndexFlatIP"
    top_k: int = 5
    ks_metricas: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class GeracaoConfig:
    """Gerador congelado para a avaliação de fidelidade e alucinação (#52)."""

    modelo_gerador: str = PENDENTE
    revisao_gerador: str = PENDENTE
    temperatura: float = 0.0
    max_tokens_resposta: int = 256
    texto_abstencao: str = "Não há informação suficiente nas aulas para responder."


@dataclass(frozen=True)
class DatasetConfig:
    """Conjunto de perguntas, respostas e evidências (#49)."""

    dataset_path: Path = ARTIGO / "data" / "perguntas.jsonl"
    n_dev: int = 15
    n_test: int = 45
    categorias: dict[str, int] = field(
        default_factory=lambda: {
            "factual": 30,
            "contextual": 15,
            "multitrecho": 10,
            "nao_respondivel": 5,
        }
    )
    doi: str = PENDENTE
    licenca: str = PENDENTE


@dataclass(frozen=True)
class ExperimentoConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    recuperacao: RecuperacaoConfig = field(default_factory=RecuperacaoConfig)
    geracao: GeracaoConfig = field(default_factory=GeracaoConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 42
    results_dir: Path = SRC / "results"


CONFIG = ExperimentoConfig()


def pendencias(cfg: ExperimentoConfig = CONFIG) -> list[str]:
    """Lista os campos ainda não congelados, no formato ``secao.campo``."""
    faltando: list[str] = []
    for secao, valores in asdict(cfg).items():
        if not isinstance(valores, dict):
            continue
        for nome, valor in valores.items():
            if valor == PENDENTE:
                faltando.append(f"{secao}.{nome}")
    if not cfg.corpus.catalogo:
        faltando.append("corpus.catalogo")
    return faltando


def validar_congelamento(cfg: ExperimentoConfig = CONFIG) -> None:
    """Impede a execução canônica com protocolo em aberto."""
    faltando = pendencias(cfg)
    if faltando:
        raise RuntimeError(
            "Protocolo não congelado; preencher em src/config.py: "
            + ", ".join(faltando)
        )


def set_seed(seed: int) -> None:
    """Fixa as fontes de aleatoriedade usadas pelo pipeline."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:  # numpy só entra a partir do chunking
        return
    np.random.seed(seed)


def como_dict(cfg: ExperimentoConfig = CONFIG) -> dict[str, Any]:
    """Versão serializável do protocolo, gravada junto de cada resultado."""

    def _conv(valor: Any) -> Any:
        if isinstance(valor, Path):
            return str(valor)
        if isinstance(valor, dict):
            return {str(k): _conv(v) for k, v in valor.items()}
        if isinstance(valor, (list, tuple)):
            return [_conv(v) for v in valor]
        return valor

    return _conv(asdict(cfg))
