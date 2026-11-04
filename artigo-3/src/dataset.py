"""Leitura e validação do dataset de perguntas, respostas e evidências (#49).

O formato é um JSONL com um item por linha::

    {"id": "q001", "split": "dev", "parte": "V", "aula": 14, "pagina": 7,
     "categoria": "factual", "pergunta": "...", "resposta_referencia": "...",
     "evidencias_validas": [{"aula": 14, "pagina": 7}], "respondivel": true}

`validar` cobre os critérios de aceite da #49 que são verificáveis por código;
a revisão cruzada do conteúdo continua sendo humana.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config import CONFIG, DatasetConfig

CAMPOS = (
    "id", "split", "parte", "aula", "pagina", "categoria", "pergunta",
    "resposta_referencia", "evidencias_validas", "respondivel",
)
SPLITS = ("dev", "test")


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
    resposta_referencia: str
    evidencias_validas: tuple[tuple[int, int], ...]  # (aula, página)
    respondivel: bool


def sha256_dataset(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _item(bruto: dict, n_linha: int) -> Item:
    faltando = [c for c in CAMPOS if c not in bruto]
    if faltando:
        raise DatasetError(f"linha {n_linha}: campos ausentes {faltando}")
    evid = tuple((int(e["aula"]), int(e["pagina"])) for e in bruto["evidencias_validas"])
    return Item(
        id=str(bruto["id"]),
        split=bruto["split"],
        parte=bruto["parte"],
        aula=bruto["aula"],
        pagina=bruto["pagina"],
        categoria=bruto["categoria"],
        pergunta=bruto["pergunta"].strip(),
        resposta_referencia=bruto["resposta_referencia"].strip(),
        evidencias_validas=evid,
        respondivel=bool(bruto["respondivel"]),
    )


def validar(itens: list[Item], cfg: DatasetConfig = CONFIG.dataset, estrito: bool = True) -> None:
    """Valida IDs, splits e evidências; ``estrito`` também confere as contagens do protocolo."""
    erros: list[str] = []
    repetidos = [i for i, n in Counter(it.id for it in itens).items() if n > 1]
    if repetidos:
        erros.append(f"IDs repetidos: {repetidos}")
    for it in itens:
        if it.split not in SPLITS:
            erros.append(f"{it.id}: split {it.split!r} inválido")
        if it.categoria not in cfg.categorias:
            erros.append(f"{it.id}: categoria {it.categoria!r} inválida")
        if not it.pergunta:
            erros.append(f"{it.id}: pergunta vazia")
        if it.respondivel and not it.evidencias_validas:
            erros.append(f"{it.id}: respondível sem evidência")
        if not it.respondivel and it.evidencias_validas:
            erros.append(f"{it.id}: não respondível com evidência")
        if (it.categoria == "nao_respondivel") == it.respondivel:
            erros.append(f"{it.id}: categoria e 'respondivel' inconsistentes")
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
    if erros:
        raise DatasetError("Dataset inválido — " + "; ".join(erros))


def carregar(caminho: Path | None = None, *, split: str | None = None, estrito: bool = True) -> list[Item]:
    """Lê e valida o dataset inteiro; devolve só o ``split`` pedido, se houver."""
    caminho = caminho or CONFIG.dataset.dataset_path
    if not caminho.is_file():
        raise DatasetError(f"Dataset não encontrado: {caminho}")
    itens = [
        _item(json.loads(linha), n)
        for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1)
        if linha.strip()
    ]
    validar(itens, estrito=estrito)
    if split is None:
        return itens
    if split not in SPLITS:
        raise DatasetError(f"split {split!r} inválido")
    return [it for it in itens if it.split == split]
