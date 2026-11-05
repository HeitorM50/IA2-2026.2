"""Leitura e validação do dataset de perguntas, respostas e evidências (#49).

O formato é um JSONL com um item por linha::

    {"id": "q001", "split": "dev", "parte": "V", "aula": 14, "pagina": 7,
     "categoria": "factual", "pergunta": "...", "resposta_referencia": "...",
     "evidencias_validas": [{"aula": 14, "pagina": 7}], "respondivel": true}

`validar` cobre os critérios de aceite da #49 que são verificáveis por código;
a revisão cruzada do conteúdo continua sendo humana e é registrada em um
arquivo separado, para não contaminar o contrato público do dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import CONFIG, CorpusConfig, DatasetConfig

CAMPOS = (
    "id", "split", "parte", "aula", "pagina", "categoria", "pergunta",
    "resposta_referencia", "evidencias_validas", "respondivel",
)
SPLITS = ("dev", "test")
PARTES = ("V", "VI", "VII")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    split: str
    parte: str | None
    aula: int | None
    pagina: int | None
    categoria: str
    pergunta: str
    resposta_referencia: str | None
    evidencias_validas: tuple[tuple[int, int], ...]  # (aula, página)
    respondivel: bool


def sha256_dataset(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _inteiro(valor: Any, campo: str, n_linha: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise DatasetError(f"linha {n_linha}: {campo} deve ser inteiro")
    return valor


def _item(bruto: dict, n_linha: int) -> Item:
    if not isinstance(bruto, dict):
        raise DatasetError(f"linha {n_linha}: registro deve ser um objeto JSON")
    faltando = [c for c in CAMPOS if c not in bruto]
    if faltando:
        raise DatasetError(f"linha {n_linha}: campos ausentes {faltando}")
    extras = sorted(set(bruto) - set(CAMPOS))
    if extras:
        raise DatasetError(f"linha {n_linha}: campos desconhecidos {extras}")
    if not isinstance(bruto["evidencias_validas"], list):
        raise DatasetError(f"linha {n_linha}: evidencias_validas deve ser uma lista")
    evidencias = []
    for posicao, evidencia in enumerate(bruto["evidencias_validas"], start=1):
        if not isinstance(evidencia, dict) or set(evidencia) != {"aula", "pagina"}:
            raise DatasetError(
                f"linha {n_linha}: evidência {posicao} deve conter somente aula e pagina"
            )
        evidencias.append(
            (
                _inteiro(evidencia["aula"], f"evidência {posicao}.aula", n_linha),
                _inteiro(evidencia["pagina"], f"evidência {posicao}.pagina", n_linha),
            )
        )
    if not isinstance(bruto["respondivel"], bool):
        raise DatasetError(f"linha {n_linha}: respondivel deve ser booleano")
    resposta = bruto["resposta_referencia"]
    if resposta is not None and not isinstance(resposta, str):
        raise DatasetError(f"linha {n_linha}: resposta_referencia deve ser texto ou null")
    return Item(
        id=str(bruto["id"]),
        split=bruto["split"],
        parte=bruto["parte"],
        aula=_inteiro(bruto["aula"], "aula", n_linha),
        pagina=_inteiro(bruto["pagina"], "pagina", n_linha),
        categoria=bruto["categoria"],
        pergunta=str(bruto["pergunta"]).strip(),
        resposta_referencia=resposta.strip() if resposta is not None else None,
        evidencias_validas=tuple(evidencias),
        respondivel=bruto["respondivel"],
    )


def _paginas_do_manifesto(manifesto: dict | None) -> dict[int, int]:
    if manifesto is None:
        return {}
    try:
        return {int(a["aula"]): int(a["n_paginas"]) for a in manifesto["arquivos"]}
    except (KeyError, TypeError, ValueError) as e:
        raise DatasetError("manifesto do corpus sem aula/n_paginas válidos") from e


def validar(
    itens: list[Item],
    cfg: DatasetConfig = CONFIG.dataset,
    estrito: bool = True,
    *,
    corpus: CorpusConfig = CONFIG.corpus,
    manifesto: dict | None = None,
) -> None:
    """Valida IDs, splits e evidências; ``estrito`` também confere as contagens do protocolo."""
    erros: list[str] = []
    paginas_por_aula = _paginas_do_manifesto(manifesto)
    repetidos = [i for i, n in Counter(it.id for it in itens).items() if n > 1]
    if repetidos:
        erros.append(f"IDs repetidos: {repetidos}")
    for it in itens:
        if not it.id.strip():
            erros.append("ID vazio")
        if it.split not in SPLITS:
            erros.append(f"{it.id}: split {it.split!r} inválido")
        if it.categoria not in cfg.categorias:
            erros.append(f"{it.id}: categoria {it.categoria!r} inválida")
        if it.parte not in PARTES:
            erros.append(f"{it.id}: parte {it.parte!r} inválida")
        if it.aula not in corpus.aulas:
            erros.append(f"{it.id}: aula {it.aula!r} fora do corpus")
        elif it.aula in corpus.catalogo and corpus.catalogo[it.aula][0] != it.parte:
            erros.append(
                f"{it.id}: aula {it.aula} pertence à parte {corpus.catalogo[it.aula][0]}, "
                f"não à {it.parte}"
            )
        if it.pagina is None or it.pagina < 1:
            erros.append(f"{it.id}: página principal inválida")
        elif paginas_por_aula and it.pagina > paginas_por_aula.get(it.aula, 0):
            erros.append(f"{it.id}: página principal não existe no manifesto")
        if not it.pergunta:
            erros.append(f"{it.id}: pergunta vazia")
        if it.respondivel and not it.resposta_referencia:
            erros.append(f"{it.id}: respondível sem resposta de referência")
        if not it.respondivel and it.resposta_referencia:
            erros.append(f"{it.id}: não respondível com resposta de referência")
        if it.respondivel and not it.evidencias_validas:
            erros.append(f"{it.id}: respondível sem evidência")
        if not it.respondivel and it.evidencias_validas:
            erros.append(f"{it.id}: não respondível com evidência")
        if (it.categoria == "nao_respondivel") == it.respondivel:
            erros.append(f"{it.id}: categoria e 'respondivel' inconsistentes")
        if len(set(it.evidencias_validas)) != len(it.evidencias_validas):
            erros.append(f"{it.id}: evidências repetidas")
        if it.categoria == "multitrecho" and len(set(it.evidencias_validas)) < 2:
            erros.append(f"{it.id}: multitrecho exige ao menos duas evidências")
        if it.respondivel and (it.aula, it.pagina) not in it.evidencias_validas:
            erros.append(f"{it.id}: localização principal ausente das evidências")
        for aula, pagina in it.evidencias_validas:
            if aula not in corpus.aulas or pagina < 1:
                erros.append(f"{it.id}: evidência ({aula}, {pagina}) inválida")
            elif paginas_por_aula and pagina > paginas_por_aula.get(aula, 0):
                erros.append(f"{it.id}: evidência ({aula}, {pagina}) não existe no manifesto")
    perguntas_repetidas = [p for p, n in Counter(it.pergunta.lower() for it in itens).items() if n > 1]
    if perguntas_repetidas:
        erros.append(f"perguntas repetidas (possível vazamento entre splits): {perguntas_repetidas[:3]}")
    if estrito:
        por_split = Counter(it.split for it in itens)
        if por_split["dev"] != cfg.n_dev or por_split["test"] != cfg.n_test:
            erros.append(f"split esperado {cfg.n_dev}/{cfg.n_test}, obtido {dict(por_split)}")
        por_cat = Counter(it.categoria for it in itens)
        if dict(por_cat) != cfg.categorias:
            erros.append(f"categorias esperadas {cfg.categorias}, obtidas {dict(por_cat)}")
        por_split_cat = {
            split: dict(Counter(it.categoria for it in itens if it.split == split))
            for split in SPLITS
        }
        if por_split_cat != cfg.categorias_por_split:
            erros.append(
                f"categorias por split esperadas {cfg.categorias_por_split}, "
                f"obtidas {por_split_cat}"
            )
        aulas = Counter(it.aula for it in itens)
        esperado_aulas = {aula: 6 for aula in corpus.aulas}
        if dict(sorted(aulas.items())) != esperado_aulas:
            erros.append(f"distribuição por aula esperada {esperado_aulas}, obtida {dict(aulas)}")
        partes_presentes = {it.parte for it in itens}
        if partes_presentes != set(PARTES):
            erros.append(f"partes esperadas {PARTES}, obtidas {sorted(partes_presentes)}")
    if erros:
        raise DatasetError("Dataset inválido — " + "; ".join(erros))


def carregar(
    caminho: Path | None = None,
    *,
    split: str | None = None,
    estrito: bool = True,
    manifesto_path: Path | None = None,
    cfg: DatasetConfig = CONFIG.dataset,
    corpus: CorpusConfig = CONFIG.corpus,
) -> list[Item]:
    """Lê e valida o dataset inteiro; devolve só o ``split`` pedido, se houver."""
    caminho = caminho or CONFIG.dataset.dataset_path
    if not caminho.is_file():
        raise DatasetError(f"Dataset não encontrado: {caminho}")
    itens = []
    for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            bruto = json.loads(linha)
        except json.JSONDecodeError as e:
            raise DatasetError(f"linha {n}: JSON inválido: {e.msg}") from e
        itens.append(_item(bruto, n))
    manifesto = None
    if manifesto_path is not None:
        if not manifesto_path.is_file():
            raise DatasetError(f"manifesto do corpus não encontrado: {manifesto_path}")
        manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    validar(itens, cfg=cfg, estrito=estrito, corpus=corpus, manifesto=manifesto)
    if split is None:
        return itens
    if split not in SPLITS:
        raise DatasetError(f"split {split!r} inválido")
    return [it for it in itens if it.split == split]


def resumo(itens: list[Item], caminho: Path) -> dict:
    """Resumo auditável usado na revisão e no congelamento da versão."""
    return {
        "arquivo": caminho.name,
        "sha256": sha256_dataset(caminho),
        "n_itens": len(itens),
        "por_split": dict(sorted(Counter(i.split for i in itens).items())),
        "por_categoria": dict(sorted(Counter(i.categoria for i in itens).items())),
        "por_parte": dict(sorted(Counter(i.parte for i in itens).items())),
        "por_aula": {str(k): v for k, v in sorted(Counter(i.aula for i in itens).items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset.dataset_path)
    parser.add_argument("--manifesto", type=Path, default=CONFIG.corpus.manifesto_path)
    parser.add_argument("--nao-estrito", action="store_true", help="aceita conjunto parcial em elaboração")
    args = parser.parse_args(argv)
    try:
        itens = carregar(
            args.dataset,
            estrito=not args.nao_estrito,
            manifesto_path=args.manifesto,
        )
    except (DatasetError, json.JSONDecodeError) as e:
        print(f"erro: {e}")
        return 1
    print(json.dumps(resumo(itens, args.dataset), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
