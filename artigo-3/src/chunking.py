"""Segmentação determinística do corpus por tokens, sem atravessar aulas (#50)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from src.config import CONFIG, ChunkingConfig
from src.corpus import Pagina

VERSAO_CHUNKER = 1
SEPARADOR_PAGINAS = "\n\n"


class ChunkingError(ValueError):
    pass


class Tokenizador(Protocol):
    def encode(self, texto: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


@dataclass(frozen=True)
class ParametrosJanela:
    config_id: str
    tamanho_tokens: int
    sobreposicao_tokens: int
    passo_tokens: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    config_id: str
    texto: str
    aula: int
    parte: str | None
    titulo: str | None
    origem: str
    pagina_inicio: int
    pagina_fim: int
    token_inicio: int
    token_fim: int
    n_tokens: int
    sha256_texto: str

    def como_dict(self) -> dict:
        return asdict(self)


def parametros_janela(
    config_id: str, cfg: ChunkingConfig = CONFIG.chunking
) -> ParametrosJanela | None:
    """Interpreta uma configuração e materializa a sobreposição inteira."""
    if config_id in cfg.estruturais:
        return None
    for tamanho in cfg.tamanhos_tokens:
        for proporcao in cfg.sobreposicoes:
            candidato = f"c{tamanho}-o{round(proporcao * 100)}"
            if candidato == config_id:
                sobreposicao = round(tamanho * proporcao)
                passo = tamanho - sobreposicao
                if passo <= 0:
                    raise ChunkingError(f"{config_id}: sobreposição deve ser menor que o tamanho")
                return ParametrosJanela(config_id, tamanho, sobreposicao, passo)
    raise ChunkingError(f"configuração desconhecida: {config_id}")


def hash_entrada(paginas: Sequence[Pagina]) -> str:
    """Hash do mesmo fluxo canônico de páginas entregue a toda configuração."""
    registros = [
        {
            "aula": p.aula,
            "parte": p.parte,
            "titulo": p.titulo,
            "pagina": p.pagina,
            "origem": p.origem,
            "sha256_pdf": p.sha256_pdf,
            "texto": p.texto,
        }
        for p in sorted(paginas, key=lambda p: (p.aula, p.pagina))
    ]
    serializado = json.dumps(registros, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serializado.encode("utf-8")).hexdigest()


def _agrupar_paginas(paginas: Sequence[Pagina]) -> dict[int, list[Pagina]]:
    por_aula: dict[int, list[Pagina]] = defaultdict(list)
    vistos: set[tuple[int, int]] = set()
    for pagina in sorted(paginas, key=lambda p: (p.aula, p.pagina)):
        chave = (pagina.aula, pagina.pagina)
        if chave in vistos:
            raise ChunkingError(f"página repetida: aula {pagina.aula}, página {pagina.pagina}")
        vistos.add(chave)
        por_aula[pagina.aula].append(pagina)
    for aula, grupo in por_aula.items():
        origens = {p.origem for p in grupo}
        partes = {p.parte for p in grupo}
        titulos = {p.titulo for p in grupo}
        if len(origens) != 1 or len(partes) != 1 or len(titulos) != 1:
            raise ChunkingError(f"aula {aula}: metadados inconsistentes entre páginas")
    return dict(sorted(por_aula.items()))


def _tokens_da_aula(
    paginas: Sequence[Pagina], tokenizador: Tokenizador
) -> tuple[list[int], list[int]]:
    tokens: list[int] = []
    paginas_por_token: list[int] = []
    for posicao, pagina in enumerate(paginas):
        ids = list(tokenizador.encode(pagina.texto, add_special_tokens=False))
        tokens.extend(ids)
        paginas_por_token.extend([pagina.pagina] * len(ids))
        if posicao < len(paginas) - 1:
            separador = list(tokenizador.encode(SEPARADOR_PAGINAS, add_special_tokens=False))
            tokens.extend(separador)
            paginas_por_token.extend([pagina.pagina] * len(separador))
    return tokens, paginas_por_token


def _criar_chunk(
    *,
    config_id: str,
    aula: int,
    pagina_modelo: Pagina,
    ids: Sequence[int],
    paginas_por_token: Sequence[int],
    inicio: int,
    fim: int,
    tokenizador: Tokenizador,
) -> Chunk | None:
    texto = tokenizador.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not texto:
        return None
    digest_texto = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    identidade = json.dumps(
        [config_id, aula, inicio, fim, digest_texto], separators=(",", ":")
    ).encode("utf-8")
    digest_id = hashlib.sha256(identidade).hexdigest()[:12]
    return Chunk(
        chunk_id=f"{config_id}-a{aula:02d}-t{inicio:06d}-{fim:06d}-{digest_id}",
        config_id=config_id,
        texto=texto,
        aula=aula,
        parte=pagina_modelo.parte,
        titulo=pagina_modelo.titulo,
        origem=pagina_modelo.origem,
        pagina_inicio=min(paginas_por_token),
        pagina_fim=max(paginas_por_token),
        token_inicio=inicio,
        token_fim=fim,
        n_tokens=len(ids),
        sha256_texto=digest_texto,
    )


def gerar_chunks(
    paginas: Sequence[Pagina],
    config_id: str,
    tokenizador: Tokenizador,
    cfg: ChunkingConfig = CONFIG.chunking,
) -> list[Chunk]:
    """Gera chunks na ordem aula→token; a fronteira de aula nunca é cruzada."""
    parametros = parametros_janela(config_id, cfg)
    por_aula = _agrupar_paginas(paginas)
    chunks: list[Chunk] = []
    for aula, grupo in por_aula.items():
        if parametros is None:
            deslocamento = 0
            for pagina in grupo:
                ids = list(tokenizador.encode(pagina.texto, add_special_tokens=False))
                if ids:
                    chunk = _criar_chunk(
                        config_id=config_id,
                        aula=aula,
                        pagina_modelo=pagina,
                        ids=ids,
                        paginas_por_token=[pagina.pagina] * len(ids),
                        inicio=deslocamento,
                        fim=deslocamento + len(ids),
                        tokenizador=tokenizador,
                    )
                    if chunk:
                        chunks.append(chunk)
                deslocamento += len(ids)
            continue

        tokens, mapa_paginas = _tokens_da_aula(grupo, tokenizador)
        inicio = 0
        while inicio < len(tokens):
            fim = min(inicio + parametros.tamanho_tokens, len(tokens))
            chunk = _criar_chunk(
                config_id=config_id,
                aula=aula,
                pagina_modelo=grupo[0],
                ids=tokens[inicio:fim],
                paginas_por_token=mapa_paginas[inicio:fim],
                inicio=inicio,
                fim=fim,
                tokenizador=tokenizador,
            )
            if chunk:
                chunks.append(chunk)
            if fim == len(tokens):
                break
            inicio += parametros.passo_tokens
    return chunks
