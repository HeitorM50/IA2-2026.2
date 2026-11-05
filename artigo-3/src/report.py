"""Consolida os resultados do test em CSV, macros LaTeX, tabela e figura (#54).

Regra permanente do repositório: nenhum número do artigo é digitado à mão.
Este script lê os resultados brutos e escreve:

- ``src/results/resumo.csv``: uma linha por configuração;
- ``paper/results-generated.tex``: uma macro por número usado no texto;
- ``paper/tabela-principal.tex``: a tabela comparativa (booktabs);
- ``paper/figs/compromisso.pdf``: MRR@5 contra tokens de contexto.

Incerteza: há uma execução por configuração (decodificação gulosa), então a
dispersão vem das perguntas. Intervalos de 95% por bootstrap percentil sobre
as perguntas respondíveis; as diferenças contra a linha de base ``slide`` são
pareadas (a mesma reamostra de perguntas para as duas configurações).

A avaliação de qualidade da resposta é feita por dois juízes LLM, declarados
como tal no artigo; ela entra só se ``data/avaliacao/juiz-*.csv`` existirem.

Uso::

    python -m src.report
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from src.config import ARTIGO, CONFIG

RESULTADOS = CONFIG.results_dir
PAPER = ARTIGO / "paper"
AVALIACAO = CONFIG.dataset.dataset_path.parent / "avaliacao"
N_BOOTSTRAP = 10_000
LINHA_DE_BASE = "slide"

# Sufixo de macro por configuração (macros LaTeX só aceitam letras).
SUFIXO = {
    "slide": "Slide",
    "c128-o0": "CPeqSem",
    "c128-o20": "CPeqCom",
    "c256-o0": "CMedSem",
    "c256-o20": "CMedCom",
    "c512-o0": "CGrdSem",
    "c512-o20": "CGrdCom",
}
ROTULO = {
    "slide": "Slide",
    "c128-o0": "128 / 0\\%",
    "c128-o20": "128 / 20\\%",
    "c256-o0": "256 / 0\\%",
    "c256-o20": "256 / 20\\%",
    "c512-o0": "512 / 0\\%",
    "c512-o20": "512 / 20\\%",
}


class ReportError(RuntimeError):
    pass


# --------------------------------------------------------------------------- leitura


def carregar_rankings(caminho: Path) -> list[dict]:
    if not caminho.is_file():
        raise ReportError(f"rankings ausentes: {caminho}")
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def carregar_geracao(raiz: Path) -> list[dict]:
    brutos = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(raiz.glob("*/*.json"))
        if p.name not in {"run-manifest.json", "environment.json"}
    ]
    if not brutos:
        raise ReportError(f"sem resultados de geração em {raiz}")
    return brutos


def carregar_juizes(avaliacao: Path) -> dict[str, dict[str, dict]]:
    """{juiz: {item: notas}} para cada ``juiz-*.csv`` encontrado."""
    from src.avaliacao_humana import ler_avaliacao

    return {p.stem: ler_avaliacao(p) for p in sorted(avaliacao.glob("juiz-*.csv"))}


# --------------------------------------------------------------------------- estatística


def bootstrap_ic(valores: list[float], *, n: int = N_BOOTSTRAP, seed: int = CONFIG.seed) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(valores)
    medias = sorted(fmean(rng.choices(valores, k=k)) for _ in range(n))
    return medias[int(0.025 * n)], medias[int(0.975 * n) - 1]


def bootstrap_diferenca(
    a: dict[str, float], b: dict[str, float], *, n: int = N_BOOTSTRAP, seed: int = CONFIG.seed
) -> tuple[float, float, float]:
    """Diferença média pareada a − b e IC 95%, reamostrando as perguntas em comum."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ReportError("sem perguntas em comum para a diferença pareada")
    dif = [a[q] - b[q] for q in ids]
    lo, hi = bootstrap_ic(dif, n=n, seed=seed)
    return fmean(dif), lo, hi


# --------------------------------------------------------------------------- agregação


def por_config_recuperacao(rankings: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{config: {métrica: {pergunta: valor}}} só nas respondíveis."""
    saida: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rankings:
        if r["split"] != "test" or not r["respondivel"]:
            continue
        m = r["metricas_evidencia"]
        q, c = r["pergunta_id"], r["config_id"]
        saida[c]["recall1"][q] = m["recall_at_k"]["1"]
        saida[c]["recall5"][q] = m["recall_at_k"]["5"]
        saida[c]["mrr5"][q] = m["mrr_at_5"]
        saida[c]["precision5"][q] = m["precision_at_k"]["5"]
    return saida


def resumo_geracao(brutos: list[dict]) -> dict[str, dict[str, float]]:
    por: dict[str, list[dict]] = defaultdict(list)
    for b in brutos:
        por[b["config_id"]].append(b)
    saida = {}
    for c, rs in por.items():
        nao = [r for r in rs if not r["respondivel"]]
        sim = [r for r in rs if r["respondivel"]]
        subst = [r for r in rs if not r["abstencao"]]
        saida[c] = {
            "abstencao_correta": fmean(r["abstencao"] for r in nao) if nao else float("nan"),
            "abstencao_indevida": fmean(r["abstencao"] for r in sim) if sim else float("nan"),
            "com_citacao": fmean(bool(r["citacoes"]) for r in subst) if subst else float("nan"),
            "citacao_invalida": fmean(bool(r["citacoes_invalidas"]) for r in subst) if subst else float("nan"),
            "tokens_entrada": fmean(r["tokens_entrada"] for r in rs),
            "latencia_geracao": fmean(r["latencia_geracao_s"] for r in rs),
            "n_nao_respondiveis": len(nao),
        }
    return saida


def resumo_juizes(
    juizes: dict[str, dict[str, dict]], chave: dict[str, tuple[str, str]]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Média dos dois juízes por item, agregada por configuração, e kappa por critério."""
    from src.avaliacao_humana import RUBRICA, kappa_cohen

    if len(juizes) != 2:
        raise ReportError(f"esperados dois juízes, encontrados {sorted(juizes)}")
    (_, j1), (_, j2) = sorted(juizes.items())
    itens = sorted(set(j1) & set(j2) & set(chave))
    if len(itens) != len(chave):
        raise ReportError("os juízes não cobrem todos os itens da chave")
    kappas = {
        c: kappa_cohen([j1[i][c] for i in itens], [j2[i][c] for i in itens], RUBRICA[c])
        for c in RUBRICA
    }
    por: dict[str, list[dict]] = defaultdict(list)
    for i in itens:
        por[chave[i][1]].append({c: (j1[i][c] + j2[i][c]) / 2 for c in RUBRICA})
    agregado = {
        cfg: {
            "correcao": fmean(n["correcao"] for n in ns) / 2,  # 0–1
            "fidelidade": fmean(n["fidelidade"] for n in ns) / 2,
            "nao_sustentadas": fmean(n["afirmacoes_nao_sustentadas"] for n in ns),
        }
        for cfg, ns in por.items()
    }
    return agregado, kappas


# --------------------------------------------------------------------------- saída


def fmt(x: float, casas: int = 2) -> str:
    """Vírgula decimal sem espaço extra no modo matemático."""
    return f"{x:.{casas}f}".replace(".", "{,}")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def macro(nome: str, valor: str) -> str:
    if not nome.isalpha():
        raise ReportError(f"nome de macro inválido: {nome}")
    return f"\\newcommand{{\\{nome}}}{{{valor}}}"


def gerar(
    rankings: list[dict],
    brutos: list[dict],
    juizes: dict[str, dict[str, dict]] | None = None,
    chave: dict[str, tuple[str, str]] | None = None,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[list[dict], list[str], str]:
    """Devolve (linhas do resumo, macros, tabela LaTeX)."""
    rec = por_config_recuperacao(rankings)
    ger = resumo_geracao(brutos)
    configs = [c for c in CONFIG.chunking.ids if c in rec]
    if set(configs) != set(ger):
        raise ReportError(f"recuperação {sorted(rec)} e geração {sorted(ger)} cobrem configurações diferentes")
    if LINHA_DE_BASE not in rec:
        raise ReportError("a linha de base 'slide' não está nos resultados")
    qual, kappas = (resumo_juizes(juizes, chave) if juizes else (None, None))

    linhas, macros = [], []
    n_resp = len(rec[LINHA_DE_BASE]["recall5"])
    n_perg = len({b["pergunta_id"] for b in brutos})
    macros += [
        macro("NPerguntasTeste", str(n_perg)),
        macro("NRespondiveisTeste", str(n_resp)),
        macro("NNaoRespondiveisTeste", str(n_perg - n_resp)),
        macro("NConfiguracoes", str(len(configs))),
        macro("NPares", str(len(brutos))),
        macro("NBootstrap", f"{n_bootstrap:,}".replace(",", ".")),
    ]
    for c in configs:
        s = SUFIXO[c]
        linha = {"config_id": c}
        for met, nome in (("recall1", "RecallUm"), ("recall5", "RecallCinco"), ("mrr5", "MRR"), ("precision5", "PrecisaoCinco")):
            vals = list(rec[c][met].values())
            media = fmean(vals)
            lo, hi = bootstrap_ic(vals, n=n_bootstrap)
            linha |= {met: media, f"{met}_ic_inf": lo, f"{met}_ic_sup": hi}
            macros += [macro(f"{nome}{s}", fmt(media)), macro(f"{nome}{s}IC", f"[{fmt(lo)}; {fmt(hi)}]")]
            if c != LINHA_DE_BASE and met in ("recall5", "mrr5"):
                d, dlo, dhi = bootstrap_diferenca(rec[c][met], rec[LINHA_DE_BASE][met], n=n_bootstrap)
                linha |= {f"{met}_dif_slide": d, f"{met}_dif_ic_inf": dlo, f"{met}_dif_ic_sup": dhi}
                macros += [
                    macro(f"Dif{nome}{s}", fmt(d)),
                    macro(f"Dif{nome}{s}IC", f"[{fmt(dlo)}; {fmt(dhi)}]"),
                ]
        g = ger[c]
        linha |= g
        macros += [
            macro(f"AbstencaoCorreta{s}", fmt_pct(g["abstencao_correta"])),
            macro(f"AbstencaoIndevida{s}", fmt_pct(g["abstencao_indevida"])),
            macro(f"ComCitacao{s}", fmt_pct(g["com_citacao"])),
            macro(f"CitacaoInvalida{s}", fmt_pct(g["citacao_invalida"])),
            macro(f"TokensEntrada{s}", f"{g['tokens_entrada']:.0f}"),
            macro(f"LatenciaGeracao{s}", fmt(g["latencia_geracao"], 1)),
        ]
        if qual:
            q = qual[c]
            linha |= {f"juiz_{k}": v for k, v in q.items()}
            macros += [
                macro(f"Correcao{s}", fmt(q["correcao"])),
                macro(f"Fidelidade{s}", fmt(q["fidelidade"])),
                macro(f"NaoSustentadas{s}", fmt(q["nao_sustentadas"])),
            ]
        linhas.append(linha)

    if kappas:
        nomes = {"correcao": "Correcao", "completude": "Completude", "fidelidade": "Fidelidade",
                 "afirmacoes_nao_sustentadas": "NaoSustentadas"}
        macros += [macro(f"Kappa{nomes[k]}", fmt(v)) for k, v in kappas.items()]
        macros.append(macro("NItensJuizes", str(len(chave))))

    # Destaques derivados, para o texto não depender de leitura manual da tabela.
    melhor_mrr = max(linhas, key=lambda l: l["mrr5"])
    melhor_r5 = max(linhas, key=lambda l: l["recall5"])
    macros += [
        macro("MelhorMRRConfig", ROTULO[melhor_mrr["config_id"]]),
        macro("MelhorRecallCincoConfig", ROTULO[melhor_r5["config_id"]]),
    ]
    return linhas, macros, tabela_latex(linhas, com_juizes=bool(qual))


def tabela_latex(linhas: list[dict], *, com_juizes: bool) -> str:
    cab = ["Config.", "R@1", "R@5", "MRR@5", "Cit.", "Tok.", "Lat. (s)"]
    if com_juizes:
        cab += ["Corr.", "Fid."]
    melhor = {m: max(l[m] for l in linhas) for m in ("recall1", "recall5", "mrr5")}

    def cel(l, m):
        v = fmt(l[m])
        return f"\\textbf{{{v}}}" if l[m] == melhor[m] else v

    corpo = []
    for l in linhas:
        c = [ROTULO[l["config_id"]], cel(l, "recall1"), cel(l, "recall5"), cel(l, "mrr5"),
             fmt_pct(l["com_citacao"]), f"{l['tokens_entrada']:.0f}", fmt(l["latencia_geracao"], 1)]
        if com_juizes:
            c += [fmt(l["juiz_correcao"]), fmt(l["juiz_fidelidade"])]
        corpo.append(" & ".join(c) + " \\\\")
    return "\n".join([
        "% Gerado por src/report.py — não editar à mão.",
        "\\begin{tabular}{l" + "r" * (len(cab) - 1) + "}",
        "\\toprule",
        " & ".join(cab) + " \\\\",
        "\\midrule",
        *corpo,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])


def figura(linhas: list[dict], destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for l in linhas:
        x, y = l["tokens_entrada"], l["mrr5"]
        err = [[y - l["mrr5_ic_inf"]], [l["mrr5_ic_sup"] - y]]
        base = l["config_id"] == LINHA_DE_BASE
        ax.errorbar(x, y, yerr=err, fmt="s" if base else "o", ms=4, capsize=2,
                    color="black" if base else "tab:blue", lw=0.8)
        # no par de mesmo tamanho, o rótulo do ponto mais alto vai acima e o outro abaixo
        par = [o for o in linhas if o is not l and o["config_id"].split("-")[0] == l["config_id"].split("-")[0]]
        dy = 4 if base or not par or y >= par[0]["mrr5"] else -8
        ax.annotate(ROTULO[l["config_id"]].replace("\\%", "%"), (x, y), textcoords="offset points",
                    xytext=(5, dy), fontsize=6)
    ax.set_xlabel("Tokens de contexto por pergunta (média)", fontsize=7)
    ax.set_ylabel("MRR@5", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino)
    plt.close(fig)


def escrever(linhas: list[dict], macros: list[str], tabela: str, *, resultados: Path, paper: Path) -> None:
    campos = sorted({k for l in linhas for k in l} - {"config_id"})
    with (resultados / "resumo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config_id", *campos])
        w.writeheader()
        w.writerows(linhas)
    (paper / "results-generated.tex").write_text(
        "% Gerado por src/report.py a partir de src/results/ (#54).\n"
        "% NÃO editar à mão: todo número do artigo sai daqui.\n"
        + "\n".join(macros) + "\n",
        encoding="utf-8",
    )
    (paper / "tabela-principal.tex").write_text(tabela, encoding="utf-8")


def main() -> int:
    try:
        from src.analise_geracao import ler_chave

        rankings = carregar_rankings(RESULTADOS / "retrieval" / "test" / "rankings.jsonl")
        brutos = carregar_geracao(RESULTADOS / "raw" / "test")
        juizes = carregar_juizes(AVALIACAO)
        chave = ler_chave(AVALIACAO / "chave.csv") if juizes else None
        linhas, macros, tabela = gerar(rankings, brutos, juizes or None, chave)
        escrever(linhas, macros, tabela, resultados=RESULTADOS, paper=PAPER)
        figura(linhas, PAPER / "figs" / "compromisso.pdf")
    except ReportError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    print(f"{len(macros)} macros, {len(linhas)} configurações; juízes: {sorted(juizes) or 'nenhum'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
