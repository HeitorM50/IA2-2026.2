"""Contratos entre as etapas do pipeline.

A recuperação (#50) e a geração (#52) são implementadas em paralelo pelos dois
autores. Este módulo fixa o que uma etapa entrega à outra, para que o `run.py`
(#53) e os testes não dependam de qual índice ou modelo está por trás.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class Trecho:
    """Um chunk recuperado, com a origem necessária para conferir a evidência."""

    chunk_id: str
    texto: str
    score: float
    aula: int
    pagina_inicio: int
    pagina_fim: int
    origem: str
    parte: str | None = None
    titulo: str | None = None

    def como_dict(self) -> dict:
        return asdict(self)


class Recuperador(Protocol):
    """Implementado pelo índice de uma configuração de chunking (#50)."""

    config_id: str

    def buscar(self, pergunta: str, k: int) -> list[Trecho]:
        """Devolve até ``k`` trechos em ordem decrescente de score."""
        ...


@dataclass(frozen=True)
class SaidaGerador:
    texto: str
    tokens_entrada: int | None
    tokens_saida: int | None


class Gerador(Protocol):
    """Modelo de linguagem congelado (#52)."""

    nome: str
    revisao: str

    def gerar(self, prompt: str, *, temperatura: float, max_tokens: int) -> SaidaGerador:
        ...
