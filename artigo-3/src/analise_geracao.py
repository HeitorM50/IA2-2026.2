"""Análise da geração a partir dos resultados brutos do `run.py` (#52).

Três etapas, todas sobre ``results/raw/<split>/<config>/<pergunta>.json``:

1. ``amostra``: sorteia perguntas estratificadas por categoria e leva **todas as
   configurações** de cada pergunta sorteada para a planilha cega. A comparação
   entre configurações fica pareada por pergunta, mesmo quando a avaliação humana
   cobre só parte do teste.
2. ``automaticas``: métricas que não dependem de avaliador — abstenção correta e
   indevida, uso de citação, citação a trecho inexistente, tokens e latência.
3. ``desfechos``: junta a nota de consenso com a evidência no top-k e separa
   falha de recuperação de falha de geração (`avaliacao_humana.classificar_falha`).

Uso::

    python -m src.analise_geracao amostra --split test --n 10
    python -m src.analise_geracao automaticas --split test
    python -m src.analise_geracao desfechos --split test --consenso data/avaliacao/consenso.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from src.avaliacao_humana import DESFECHOS, classificar_falha, gerar_planilha_cega, ler_avaliacao
from src.config import CONFIG, ExperimentoConfig

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"
AVALIACAO_DIR = CONFIG.dataset.dataset_path.parent / "avaliacao"


class AnaliseError(ValueError):
    pass


def carregar_brutos(raiz: Path) -> list[dict]:
    """Lê um JSON por (configuração, pergunta), ignorando os manifestos."""
    if not raiz.is_dir():
        raise AnaliseError(f"Sem resultados brutos em {raiz}.")
    brutos = []
    for caminho in sorted(raiz.glob("*/*.json")):
        if caminho.name in {"run-manifest.json", "environment.json"}:
            continue
        brutos.append(json.loads(caminho.read_text(encoding="utf-8")))
    if not brutos:
        raise AnaliseError(f"Nenhum resultado em {raiz}.")
    return brutos


def recuperou_evidencia(bruto: dict) -> bool:
    """Algum trecho do top-k cobre alguma página de evidência válida da pergunta."""
    evidencias = {(int(a), int(p)) for a, p in bruto.get("evidencias_validas", [])}
    return any(
        (t["aula"], p) in evidencias
        for t in bruto["trechos"]
        for p in range(t["pagina_inicio"], t["pagina_fim"] + 1)
    )


def amostrar_perguntas(brutos: list[dict], n: int, seed: int) -> list[str]:
    """Sorteia ``n`` perguntas com alocação proporcional por categoria.

    Cada categoria presente recebe ao menos uma pergunta; o restante segue a
    maior fração (método de Hamilton), com desempate pelo nome da categoria.
    """
    por_categoria: dict[str, set[str]] = defaultdict(set)
    for b in brutos:
        por_categoria[b["categoria"]].add(b["pergunta_id"])
    total = sum(len(v) for v in por_categoria.values())
    if not 0 < n <= total:
        raise AnaliseError(f"n={n} fora de 1..{total} perguntas.")
    if n < len(por_categoria):
        raise AnaliseError(f"n={n} menor que o número de categorias ({len(por_categoria)}).")

    cats = sorted(por_categoria)
    cota = {c: n * len(por_categoria[c]) / total for c in cats}
    alocado = {c: max(1, int(cota[c])) for c in cats}
    while sum(alocado.values()) > n:  # o mínimo de 1 pode estourar n
        c = max((c for c in cats if alocado[c] > 1), key=lambda c: (alocado[c] - cota[c], c))
        alocado[c] -= 1
    for c in sorted(cats, key=lambda c: (-(cota[c] - int(cota[c])), c)):
        if sum(alocado.values()) == n:
            break
        if alocado[c] < len(por_categoria[c]):
            alocado[c] += 1

    rng = random.Random(seed)
    escolhidas: list[str] = []
    for c in cats:
        escolhidas += rng.sample(sorted(por_categoria[c]), alocado[c])
    return sorted(escolhidas)


def gerar_amostra(
    brutos: list[dict], referencias: dict[str, str], n: int, destino_dir: Path, seed: int
) -> list[str]:
    """Grava planilha cega, chave e a lista de perguntas sorteadas."""
    perguntas = amostrar_perguntas(brutos, n, seed)
    selecionados = [b for b in brutos if b["pergunta_id"] in set(perguntas)]
    configs = {b["config_id"] for b in brutos}
    faltando = [
        (q, c) for q in perguntas for c in sorted(configs)
        if not any(b["pergunta_id"] == q and b["config_id"] == c for b in selecionados)
    ]
    if faltando:
        raise AnaliseError(f"Grade incompleta para a amostra: {faltando[:5]}")
    gerar_planilha_cega(
        selecionados, referencias, destino_dir / "planilha-cega.csv", destino_dir / "chave.csv", seed
    )
    (destino_dir / "amostra.json").write_text(
        json.dumps({"seed": seed, "n_perguntas": n, "perguntas": perguntas}, indent=2) + "\n",
        encoding="utf-8",
    )
    return perguntas


def metricas_automaticas(brutos: list[dict]) -> dict[str, dict[str, float | int | None]]:
    """Métricas por configuração que não dependem de avaliação humana."""
    por_config: dict[str, list[dict]] = defaultdict(list)
    for b in brutos:
        por_config[b["config_id"]].append(b)

    def taxa(num: int, den: int) -> float | None:
        return num / den if den else None

    saida = {}
    for config_id in sorted(por_config):
        rs = por_config[config_id]
        nao_resp = [r for r in rs if not r["respondivel"]]
        resp = [r for r in rs if r["respondivel"]]
        substantivas = [r for r in rs if not r["abstencao"]]
        saida[config_id] = {
            "n": len(rs),
            "abstencao_correta": taxa(sum(r["abstencao"] for r in nao_resp), len(nao_resp)),
            "abstencao_indevida": taxa(sum(r["abstencao"] for r in resp), len(resp)),
            "abstencao_indevida_com_evidencia": taxa(
                sum(r["abstencao"] and recuperou_evidencia(r) for r in resp), len(resp)
            ),
            "respostas_com_citacao": taxa(sum(bool(r["citacoes"]) for r in substantivas), len(substantivas)),
            "respostas_com_citacao_invalida": taxa(
                sum(bool(r["citacoes_invalidas"]) for r in substantivas), len(substantivas)
            ),
            "tokens_entrada_medio": _media(r["tokens_entrada"] for r in rs),
            "tokens_saida_medio": _media(r["tokens_saida"] for r in rs),
            "latencia_geracao_media_s": _media(r["latencia_geracao_s"] for r in rs),
        }
    return saida


def _media(valores) -> float | None:
    vs = [v for v in valores if v is not None]
    return mean(vs) if vs else None


def ler_chave(caminho: Path) -> dict[str, tuple[str, str]]:
    with caminho.open(encoding="utf-8") as f:
        return {l["item"]: (l["pergunta_id"], l["config_id"]) for l in csv.DictReader(f)}


def desfechos(
    brutos: list[dict], consenso: dict[str, dict[str, int]], chave: dict[str, tuple[str, str]]
) -> dict[str, dict[str, int]]:
    """Conta os desfechos de `DESFECHOS` por configuração nos itens avaliados."""
    indice = {(b["pergunta_id"], b["config_id"]): b for b in brutos}
    faltam = sorted(set(chave) - set(consenso))
    if faltam:
        raise AnaliseError(f"Itens da chave sem nota de consenso: {faltam[:5]}")
    contagem: dict[str, Counter] = defaultdict(Counter)
    for item, (pergunta_id, config_id) in chave.items():
        b = indice.get((pergunta_id, config_id))
        if b is None:
            raise AnaliseError(f"{item}: sem resultado bruto para {pergunta_id}/{config_id}.")
        d = classificar_falha(
            respondivel=b["respondivel"],
            recuperou_evidencia=recuperou_evidencia(b),
            abstencao=b["abstencao"],
            correcao=consenso[item]["correcao"],
        )
        contagem[config_id][d] += 1
    return {c: {d: contagem[c][d] for d in DESFECHOS} for c in sorted(contagem)}


def main(argv: list[str] | None = None, cfg: ExperimentoConfig = CONFIG) -> int:
    p = argparse.ArgumentParser(description="Análise da geração sobre os resultados brutos.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for nome in ("amostra", "automaticas", "desfechos"):
        s = sub.add_parser(nome)
        s.add_argument("--split", choices=("dev", "test"), default="test")
        s.add_argument("--resultados", type=Path, default=RESULTADOS_DIR / "raw")
        if nome == "amostra":
            s.add_argument("--n", type=int, required=True, help="número de perguntas sorteadas")
            s.add_argument("--destino", type=Path, default=AVALIACAO_DIR)
        if nome == "desfechos":
            s.add_argument("--consenso", type=Path, required=True)
            s.add_argument("--chave", type=Path, default=AVALIACAO_DIR / "chave.csv")
    a = p.parse_args(argv)

    try:
        brutos = carregar_brutos(a.resultados / a.split)
        if a.cmd == "amostra":
            from src.dataset import carregar

            refs = {it.id: it.resposta_referencia for it in carregar(split=a.split)}
            perguntas = gerar_amostra(brutos, refs, a.n, a.destino, cfg.seed)
            n_configs = len({b["config_id"] for b in brutos})
            print(f"{len(perguntas)} perguntas × {n_configs} configurações em {a.destino}")
            return 0
        if a.cmd == "automaticas":
            resultado = metricas_automaticas(brutos)
        else:
            resultado = desfechos(brutos, ler_avaliacao(a.consenso), ler_chave(a.chave))
    except AnaliseError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
