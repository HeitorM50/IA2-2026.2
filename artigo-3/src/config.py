"""Configuração única do protocolo do Artigo 3.

O artigo compara sete configurações de *chunking* em um sistema de geração
aumentada por recuperação (RAG) sobre as aulas 14–23 de Inteligência Artificial II.
A comparação só vale se as sete passarem pelo mesmo corpus, pelo mesmo modelo de
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

    # aula -> (parte, título). Títulos copiados da capa de cada PDF.
    # Os slides não numeram as partes; a divisão segue o bloco de conteúdo
    # (V: PLN e LLMs; VI: aprendizado por reforço; VII: XAI e fairness).
    catalogo: dict[int, tuple[str, str]] = field(
        default_factory=lambda: {
            14: ("V", "Fundamentos de Processamento de Linguagem Natural"),
            15: ("V", "Arquitetura Transformer e Mecanismos de Atenção"),
            16: ("V", "LLMs, Embeddings e Fine-tuning"),
            17: ("V", "Prompt Engineering e Aplicações Práticas"),
            18: ("VI", "Fundamentos de Aprendizado por Reforço"),
            19: ("VI", "Deep RL: Deep Q-Networks e Variantes"),
            20: ("VI", "Métodos de Política, PPO e Fechamento"),
            21: ("VII", "Introdução à IA Explicável (XAI)"),
            22: ("VII", "Ferramentas Práticas de XAI"),
            23: ("VII", "Fairness, Viés e Accountability em Sistemas de IA"),
        }
    )

    # Página com menos caracteres que isso é registrada como suspeita
    # (slide só com figura, fórmula renderizada como imagem, etc.).
    min_caracteres_pagina: int = 40


@dataclass(frozen=True)
class ChunkingConfig:
    """Estratégias comparadas (#50). Tamanhos em tokens, não caracteres.

    Duas famílias, para que a comparação seja entre estratégias e não só entre
    valores de um parâmetro:
    - janela fixa de tokens, com e sem sobreposição (``cT-oS``);
    - estrutural: um chunk por slide (``slide``), a segmentação natural do
      material, que funciona como linha de base sem nenhum parâmetro de tamanho.
    """

    tamanhos_tokens: tuple[int, ...] = (128, 256, 512)
    sobreposicoes: tuple[float, ...] = (0.0, 0.2)
    estruturais: tuple[str, ...] = ("slide",)
    versao_chunker: int = 1
    # Um slide maior que o limite de entrada do modelo de embedding é truncado
    # pelo próprio modelo; o chunker deve registrar quantos slides passaram do limite.
    # O tokenizador é o do próprio modelo de embedding, para que "128 tokens"
    # signifique o que o modelo de fato enxerga.
    tokenizador: str = "intfloat/multilingual-e5-base"

    @property
    def ids(self) -> tuple[str, ...]:
        janelas = tuple(
            f"c{t}-o{round(s * 100)}"
            for t in self.tamanhos_tokens
            for s in self.sobreposicoes
        )
        return self.estruturais + janelas


@dataclass(frozen=True)
class RecuperacaoConfig:
    """Embedding, índice e métricas de recuperação (#50, #51)."""

    modelo_embedding: str = "intfloat/multilingual-e5-base"
    revisao_embedding: str = "d128750597153bb5987e10b1c3493a34e5a4502a"
    prefixo_documento: str = "passage: "
    prefixo_consulta: str = "query: "
    batch_size: int = 32
    max_tokens_modelo: int = 512
    # Vetores normalizados + produto interno = similaridade de cosseno.
    indice_faiss: str = "IndexFlatIP"
    cache_dir: Path = ARTIGO / "cache" / "indices"
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
    categorias_por_split: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "dev": {
                "factual": 8,
                "contextual": 4,
                "multitrecho": 2,
                "nao_respondivel": 1,
            },
            "test": {
                "factual": 22,
                "contextual": 11,
                "multitrecho": 8,
                "nao_respondivel": 4,
            },
        }
    )
    doi: str = PENDENTE
    # A licença cobre as anotações autorais, não os slides de terceiros.
    licenca: str = "CC BY 4.0"


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


def como_dict(cfg: ExperimentoConfig = CONFIG, *, sem_caminhos: bool = False) -> dict[str, Any]:
    """Versão serializável do protocolo, gravada junto de cada resultado.

    Com ``sem_caminhos``, remove os caminhos locais: o hash do protocolo precisa
    ser o mesmo na máquina de cada autor e no Colab.
    """

    def _conv(valor: Any) -> Any:
        if isinstance(valor, Path):
            return str(valor)
        if isinstance(valor, dict):
            return {
                str(k): _conv(v)
                for k, v in valor.items()
                if not (sem_caminhos and isinstance(v, Path))
            }
        if isinstance(valor, (list, tuple)):
            return [_conv(v) for v in valor]
        return valor

    return _conv(asdict(cfg))
