"""Dublês compartilhados: recuperador e gerador determinísticos, sem modelo real."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.contratos import SaidaGerador, Trecho


@dataclass
class RecuperadorFalso:
    config_id: str
    chamadas: list[str] = field(default_factory=list)

    def buscar(self, pergunta: str, k: int) -> list[Trecho]:
        self.chamadas.append(pergunta)
        return [
            Trecho(
                chunk_id=f"{self.config_id}-{i}",
                texto=f"Trecho {i} sobre {pergunta}",
                score=1.0 - i / 10,
                aula=14,
                pagina_inicio=i + 1,
                pagina_fim=i + 1,
                token_inicio=i * 10,
                token_fim=(i + 1) * 10,
                n_tokens=10,
                origem="Aula14.pdf",
            )
            for i in range(k)
        ]


@dataclass
class GeradorFalso:
    resposta: str = "A atenção pondera os tokens [1]."
    nome: str = "falso"
    revisao: str = "v0"
    prompts: list[str] = field(default_factory=list)

    def gerar(self, prompt: str, *, temperatura: float, max_tokens: int) -> SaidaGerador:
        self.prompts.append(prompt)
        return SaidaGerador(texto=self.resposta, tokens_entrada=len(prompt.split()), tokens_saida=5)


@pytest.fixture
def gerador() -> GeradorFalso:
    return GeradorFalso()
