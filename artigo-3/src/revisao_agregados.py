"""Gera artefatos provisórios apenas dos agregados versionados.

Este módulo existe para que a revisão textual possa ser compilada sem inventar
as métricas que dependem dos rankings e das avaliações humanas ausentes. Depois
da recuperação, ``src.report`` substitui estes artefatos pelo relatório completo.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.config import ARTIGO, CONFIG
from src.report import ROTULO, fmt, fmt_pct


def carregar(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return [{k: (v if k == "config_id" else (float(v) if v else None)) for k, v in linha.items()}
                for linha in csv.DictReader(f)]


def tabela(linhas: list[dict]) -> str:
    campos = (
        ("recall5", "R@5"), ("mrr5", "MRR@5"), ("todas5_multi", "Multi@5"),
        ("com_citacao", "Cob. cit."), ("precisao_citacao", "Rel. cit."),
        ("juiz_correcao", "Corr."), ("juiz_fidelidade", "Fid."),
        ("juiz_nao_sustentadas", "N. sust."), ("tokens_entrada", "Tok. ent."),
        ("latencia_geracao", "Lat. (s)"),
    )
    maiores = {c for c, _ in campos[:7]}
    menores = {"juiz_nao_sustentadas", "tokens_entrada", "latencia_geracao"}
    melhor = {c: (max(l[c] for l in linhas) if c in maiores else min(l[c] for l in linhas))
              for c, _ in campos if c in maiores | menores}

    def cel(linha: dict, campo: str) -> str:
        if campo == "com_citacao":
            valor = fmt_pct(linha[campo])
        elif campo == "tokens_entrada":
            valor = f"{linha[campo]:.0f}"
        elif campo == "latencia_geracao":
            valor = fmt(linha[campo], 1)
        else:
            valor = fmt(linha[campo])
        return f"\\textbf{{{valor}}}" if linha[campo] == melhor[campo] else valor

    corpo = [
        " & ".join([ROTULO[l["config_id"]], *(cel(l, c) for c, _ in campos)]) + " \\\\"
        for l in linhas
    ]
    return "\n".join([
        "% Gerado por src/revisao_agregados.py; contém somente agregados versionados.",
        "\\begin{tabular}{l" + "r" * len(campos) + "}",
        "\\toprule",
        "Config. & " + " & ".join(rotulo for _, rotulo in campos) + " \\\\ ",
        "\\midrule",
        *corpo,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])


def figura_sobreposicao(linhas: list[dict], destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    por = {l["config_id"]: l for l in linhas}
    pares = ((128, "c128-o20", "c128-o0"), (256, "c256-o20", "c256-o0"),
             (512, "c512-o20", "c512-o0"))
    metricas = (("recall5", "Recall@5", "o"), ("mrr5", "MRR@5", "s"),
                ("todas5_multi", "Multi@5", "^"))
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    x = range(3)
    for metrica, rotulo, marcador in metricas:
        y = [por[com][metrica] - por[sem][metrica] for _, com, sem in pares]
        ax.plot(list(x), y, marker=marcador, ms=4, lw=0.8, label=rotulo)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(list(x), [f"{t} tokens" for t, _, _ in pares], fontsize=6)
    ax.set_ylabel("Diferença observada: 20% − 0%", fontsize=6.2)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.3, ncol=3, frameon=False)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figura_custo(linhas: list[dict], destino: Path) -> None:
    """MRR@5 e IC marginal contra tokens de entrada do gerador."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    for l in linhas:
        y = l["mrr5"]
        ax.errorbar(l["tokens_entrada"], y,
                    yerr=[[y - l["mrr5_ic_inf"]], [l["mrr5_ic_sup"] - y]],
                    fmt="s" if l["config_id"] == "slide" else "o", ms=3.5,
                    capsize=2, lw=0.8)
        ax.annotate(ROTULO[l["config_id"]].replace("\\%", "%"),
                    (l["tokens_entrada"], y), xytext=(4, 3),
                    textcoords="offset points", fontsize=5.2)
    ax.set_xlabel("Tokens de entrada do gerador (média)", fontsize=6.2)
    ax.set_ylabel("MRR@5", fontsize=6.2)
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    linhas = carregar(CONFIG.results_dir / "resumo.csv")
    paper = ARTIGO / "paper"
    (paper / "tabela-principal-revisada.tex").write_text(tabela(linhas), encoding="utf-8")
    figura_sobreposicao(linhas, paper / "figs" / "sobreposicao-observada.pdf")
    figura_custo(linhas, paper / "figs" / "compromisso-revisado.pdf")
    print("Tabela e figura provisórias geradas somente dos agregados versionados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
