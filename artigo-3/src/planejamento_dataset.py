"""Valida a matriz pré-anotação e o congelamento dos splits do dataset (#49).

A matriz contém somente metadados de planejamento. Perguntas, respostas e
evidências permanecem em ``perguntas.jsonl`` e são validadas por ``src.dataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config import ARTIGO, CONFIG, CorpusConfig, DatasetConfig
from src.dataset import Item

PLANO_PADRAO = ARTIGO / "data" / "planejamento-dataset.jsonl"
CAMPOS = {"id", "split", "parte", "aula", "categoria", "autor", "revisor", "status"}
STATUS = ("planejado", "redigido", "revisado", "corrigido", "aprovado", "congelado")


class PlanejamentoError(ValueError):
    pass


@dataclass(frozen=True)
class ItemPlanejado:
    id: str
    split: str
    parte: str
    aula: int
    categoria: str
    autor: str
    revisor: str
    status: str


def sha256_plano(caminho: Path = PLANO_PADRAO) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def carregar_plano(caminho: Path = PLANO_PADRAO) -> list[ItemPlanejado]:
    if not caminho.is_file():
        raise PlanejamentoError(f"planejamento não encontrado: {caminho}")
    itens = []
    for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            bruto = json.loads(linha)
        except json.JSONDecodeError as e:
            raise PlanejamentoError(f"linha {n}: JSON inválido: {e.msg}") from e
        if not isinstance(bruto, dict) or set(bruto) != CAMPOS:
            raise PlanejamentoError(f"linha {n}: campos esperados {sorted(CAMPOS)}")
        if isinstance(bruto["aula"], bool) or not isinstance(bruto["aula"], int):
            raise PlanejamentoError(f"linha {n}: aula deve ser inteira")
        itens.append(ItemPlanejado(**bruto))
    return itens


def validar_plano(
    itens: list[ItemPlanejado],
    cfg: DatasetConfig = CONFIG.dataset,
    corpus: CorpusConfig = CONFIG.corpus,
) -> None:
    erros = []
    repetidos = sorted(i for i, n in Counter(x.id for x in itens).items() if n > 1)
    if repetidos:
        erros.append(f"IDs repetidos: {repetidos}")
    if len(itens) != cfg.n_dev + cfg.n_test:
        erros.append(f"esperados {cfg.n_dev + cfg.n_test} itens, obtidos {len(itens)}")
    for item in itens:
        if item.split not in cfg.categorias_por_split:
            erros.append(f"{item.id}: split inválido")
        if item.categoria not in cfg.categorias:
            erros.append(f"{item.id}: categoria inválida")
        if item.aula not in corpus.catalogo:
            erros.append(f"{item.id}: aula fora do catálogo")
        elif corpus.catalogo[item.aula][0] != item.parte:
            erros.append(f"{item.id}: parte incompatível com a aula")
        if not item.autor.strip() or not item.revisor.strip():
            erros.append(f"{item.id}: autor e revisor são obrigatórios")
        if item.autor.casefold() == item.revisor.casefold():
            erros.append(f"{item.id}: autor e revisor devem ser pessoas diferentes")
        if item.status not in STATUS:
            erros.append(f"{item.id}: status {item.status!r} inválido")

    por_split = Counter(i.split for i in itens)
    esperado_split = {"dev": cfg.n_dev, "test": cfg.n_test}
    if dict(por_split) != esperado_split:
        erros.append(f"splits esperados {esperado_split}, obtidos {dict(por_split)}")
    por_categoria = Counter(i.categoria for i in itens)
    if dict(por_categoria) != cfg.categorias:
        erros.append(f"categorias esperadas {cfg.categorias}, obtidas {dict(por_categoria)}")
    por_split_categoria = {
        split: dict(Counter(i.categoria for i in itens if i.split == split))
        for split in ("dev", "test")
    }
    if por_split_categoria != cfg.categorias_por_split:
        erros.append(
            f"categorias por split esperadas {cfg.categorias_por_split}, "
            f"obtidas {por_split_categoria}"
        )
    por_aula = Counter(i.aula for i in itens)
    esperado_aula = {aula: 6 for aula in corpus.aulas}
    if dict(sorted(por_aula.items())) != esperado_aula:
        erros.append(f"itens por aula esperados {esperado_aula}, obtidos {dict(por_aula)}")
    aulas_dev = {i.aula for i in itens if i.split == "dev"}
    if aulas_dev != set(corpus.aulas):
        erros.append(f"dev deve cobrir as dez aulas; cobre {sorted(aulas_dev)}")
    if erros:
        raise PlanejamentoError("Planejamento inválido — " + "; ".join(erros))


def validar_contra_dataset(
    plano: list[ItemPlanejado],
    dataset: list[Item],
    *,
    exigir_aprovacao: bool = False,
) -> None:
    """Impede mudanças silenciosas de split, categoria ou localização primária."""
    por_id = {i.id: i for i in dataset}
    erros = []
    if set(p.id for p in plano) != set(por_id):
        erros.append("IDs do planejamento e do dataset não coincidem")
    for p in plano:
        item = por_id.get(p.id)
        if item is None:
            continue
        esperado = (p.split, p.parte, p.aula, p.categoria)
        observado = (item.split, item.parte, item.aula, item.categoria)
        if observado != esperado:
            erros.append(f"{p.id}: metadados alterados {esperado} -> {observado}")
        if exigir_aprovacao and p.status not in {"aprovado", "congelado"}:
            erros.append(f"{p.id}: ainda não aprovado")
    if erros:
        raise PlanejamentoError("Dataset diverge do planejamento — " + "; ".join(erros))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plano", type=Path, default=PLANO_PADRAO)
    args = parser.parse_args(argv)
    try:
        itens = carregar_plano(args.plano)
        validar_plano(itens)
    except PlanejamentoError as e:
        print(f"erro: {e}")
        return 1
    print(
        json.dumps(
            {
                "arquivo": args.plano.name,
                "sha256": sha256_plano(args.plano),
                "n_itens": len(itens),
                "por_split": dict(Counter(i.split for i in itens)),
                "por_categoria": dict(Counter(i.categoria for i in itens)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
