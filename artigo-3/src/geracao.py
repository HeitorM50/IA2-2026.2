"""Geração restrita às evidências recuperadas (#52).

O prompt numera os trechos e exige que cada afirmação cite o número do trecho
que a sustenta. Quando o contexto não basta, o modelo deve responder exatamente
o texto de abstenção definido em `config.py`; isso permite medir a abstenção
correta nas perguntas não respondíveis sem depender de interpretação.

O backend concreto do gerador (modelo aberto local ou API versionada) está em
aberto na #46; qualquer objeto que satisfaça `contratos.Gerador` serve.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass

from src.config import CONFIG, GeracaoConfig
from src.contratos import Gerador, Recuperador, SaidaGerador, Trecho

VERSAO_PROMPT = 1

INSTRUCOES = """Você responde perguntas sobre as aulas de Inteligência Artificial II usando SOMENTE os trechos numerados abaixo.
Regras:
1. Use apenas informação presente nos trechos; não use conhecimento próprio.
2. Depois de cada afirmação, cite entre colchetes o número do trecho que a sustenta, por exemplo [2].
3. Se os trechos não forem suficientes para responder, escreva exatamente: "{abstencao}"
4. Responda em português, de forma direta, em no máximo cinco frases."""


def montar_prompt(pergunta: str, trechos: list[Trecho], cfg: GeracaoConfig = CONFIG.geracao) -> str:
    blocos = [
        f"[{i}] (aula {t.aula}, p. {t.pagina_inicio}"
        + (f"–{t.pagina_fim}" if t.pagina_fim != t.pagina_inicio else "")
        + f")\n{t.texto}"
        for i, t in enumerate(trechos, start=1)
    ]
    contexto = "\n\n".join(blocos) if blocos else "(nenhum trecho recuperado)"
    return (
        INSTRUCOES.format(abstencao=cfg.texto_abstencao)
        + f"\n\nTrechos:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    )


def _sem_acento(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def eh_abstencao(resposta: str, cfg: GeracaoConfig = CONFIG.geracao) -> bool:
    """Abstenção = a resposta contém o texto combinado, ignorando caixa, acento e pontuação."""

    def limpa(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]", " ", _sem_acento(s)).split())

    return limpa(cfg.texto_abstencao) in limpa(resposta)


def extrair_citacoes(resposta: str, n_trechos: int) -> tuple[list[int], list[int]]:
    """Devolve (citações válidas, citações para trechos inexistentes), sem repetição."""
    numeros: list[int] = []
    for grupo in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", resposta):
        for n in re.split(r"\s*[,;]\s*", grupo):
            if int(n) not in numeros:
                numeros.append(int(n))
    validas = [n for n in numeros if 1 <= n <= n_trechos]
    invalidas = [n for n in numeros if not 1 <= n <= n_trechos]
    return validas, invalidas


@dataclass(frozen=True)
class RespostaRAG:
    pergunta_id: str
    config_id: str
    pergunta: str
    trechos: list[dict]
    prompt: str
    versao_prompt: int
    resposta: str
    abstencao: bool
    citacoes: list[int]
    citacoes_invalidas: list[int]
    modelo_gerador: str
    revisao_gerador: str
    temperatura: float
    max_tokens: int
    top_k: int
    tokens_entrada: int | None
    tokens_saida: int | None
    latencia_recuperacao_s: float
    latencia_geracao_s: float

    def como_dict(self) -> dict:
        return asdict(self)


def responder(
    pergunta_id: str,
    pergunta: str,
    recuperador: Recuperador,
    gerador: Gerador,
    *,
    top_k: int,
    cfg: GeracaoConfig = CONFIG.geracao,
) -> RespostaRAG:
    t0 = time.perf_counter()
    trechos = recuperador.buscar(pergunta, top_k)
    t1 = time.perf_counter()
    prompt = montar_prompt(pergunta, trechos, cfg)
    saida: SaidaGerador = gerador.gerar(
        prompt, temperatura=cfg.temperatura, max_tokens=cfg.max_tokens_resposta
    )
    t2 = time.perf_counter()
    validas, invalidas = extrair_citacoes(saida.texto, len(trechos))
    return RespostaRAG(
        pergunta_id=pergunta_id,
        config_id=recuperador.config_id,
        pergunta=pergunta,
        trechos=[t.como_dict() for t in trechos],
        prompt=prompt,
        versao_prompt=VERSAO_PROMPT,
        resposta=saida.texto,
        abstencao=eh_abstencao(saida.texto, cfg),
        citacoes=validas,
        citacoes_invalidas=invalidas,
        modelo_gerador=gerador.nome,
        revisao_gerador=gerador.revisao,
        temperatura=cfg.temperatura,
        max_tokens=cfg.max_tokens_resposta,
        top_k=top_k,
        tokens_entrada=saida.tokens_entrada,
        tokens_saida=saida.tokens_saida,
        latencia_recuperacao_s=t1 - t0,
        latencia_geracao_s=t2 - t1,
    )
