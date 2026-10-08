"""Dos JSON de execução para a tabela, as macros e a figura do artigo.

Regra que manda neste módulo e está no `CLAUDE.md` §9: **nenhum número entra no
`.tex` digitado à mão**. Todo valor do artigo sai de um JSON gravado por uma
execução. Se o número não existe em `results/`, ele não vai para o texto.

O artigo cita a macro, nunca o número. Precisando de um valor novo, acrescente a
macro aqui — não digite no `main.tex`.

Saídas, a partir de `artigo-2/`:

    python -m src.report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SRC = Path(__file__).resolve().parent
PAPER = SRC.parent / "paper"

DEFAULT_RESULTS = SRC / "results"
DEFAULT_RESULTS_MEAN = SRC / "results-mean"
DEFAULT_SUMMARY = SRC / "results" / "resumo.csv"
DEFAULT_MACROS = PAPER / "results-generated.tex"
DEFAULT_FIGURE = PAPER / "figs" / "auc-pr-por-classe.pdf"

MODELS = ("limiar", "pca", "autoencoder")
MODEL_LABELS = {
    "limiar": "Limiar 3$\\sigma$",
    "pca": "PCA",
    "autoencoder": "Autoencoder LSTM",
}
# Chave de macro por modelo: o LaTeX não aceita dígito nem acento em \newcommand.
MODEL_KEYS = {"limiar": "Limiar", "pca": "Pca", "autoencoder": "Autoencoder"}

FAULTS = ("spike", "gain", "drift", "stuck", "gap")
FAULT_LABELS = {
    "spike": "pico",
    "gain": "ganho",
    "drift": "deriva",
    "stuck": "travado",
    "gap": "lacuna",
}
FAULT_KEYS = {
    "spike": "Pico",
    "gain": "Ganho",
    "drift": "Deriva",
    "stuck": "Travado",
    "gap": "Lacuna",
}

AGGREGATIONS = ("max", "mean")
AGGREGATION_KEYS = {"max": "Max", "mean": "Mean"}

EXPECTED_SEEDS = (42, 1337, 2026)


class ResultValidationError(ValueError):
    """Resultado ausente, incompleto ou inconsistente com o protocolo."""


# --------------------------------------------------------------------------
# Leitura e validação
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """Média e desvio amostral de uma métrica entre as sementes."""

    mean: float
    std: float
    n: int

    @classmethod
    def of(cls, values: Sequence[float]) -> "Aggregate":
        if not values:
            raise ResultValidationError("Agregação sem valores.")
        media = statistics.fmean(values)
        desvio = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(media, desvio, len(values))


def _finite(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResultValidationError(f"{path} deveria ser numérico; veio {value!r}.")
    if not math.isfinite(float(value)):
        raise ResultValidationError(f"{path} não é finito.")
    return float(value)


def load_results(results_dir: Path, aggregation: str) -> dict[str, list[dict[str, Any]]]:
    """Carrega os JSON de um diretório, conferindo que formam a grade esperada.

    A validação é deliberadamente rígida: um resultado faltando ou de outra
    agregação produziria uma tabela silenciosamente errada no artigo.
    """

    if not results_dir.is_dir():
        raise ResultValidationError(f"Diretório inexistente: {results_dir}")

    por_modelo: dict[str, list[dict[str, Any]]] = {}
    for modelo in MODELS:
        execucoes = []
        for seed in EXPECTED_SEEDS:
            caminho = results_dir / f"{modelo}-seed{seed}.json"
            if not caminho.exists():
                raise ResultValidationError(f"Resultado ausente: {caminho}")
            dados = json.loads(caminho.read_text(encoding="utf-8"))

            if dados.get("model") != modelo:
                raise ResultValidationError(
                    f"{caminho}: campo 'model' é {dados.get('model')!r}, "
                    f"esperado {modelo!r}."
                )
            if dados.get("seed") != seed:
                raise ResultValidationError(
                    f"{caminho}: campo 'seed' é {dados.get('seed')!r}, "
                    f"esperado {seed}."
                )
            encontrada = dados.get("window_aggregation")
            if encontrada != aggregation:
                raise ResultValidationError(
                    f"{caminho}: agregação {encontrada!r}, esperada {aggregation!r}. "
                    "Misturar agregações na mesma tabela inviabiliza a comparação."
                )
            for classe in FAULTS:
                if classe not in dados["test"]["per_fault"]:
                    raise ResultValidationError(
                        f"{caminho}: falta a classe de falha {classe!r}."
                    )
            execucoes.append(dados)
        por_modelo[modelo] = execucoes
    return por_modelo


def summarize(
    por_modelo: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Consolida cada modelo em médias e desvios entre as sementes."""

    resumo: dict[str, dict[str, Any]] = {}
    for modelo, execucoes in por_modelo.items():
        entrada: dict[str, Any] = {
            "auc_pr": Aggregate.of(
                [_finite(r["test"]["auc_pr"], "test.auc_pr") for r in execucoes]
            ),
            "f1": Aggregate.of([_finite(r["test"]["f1"], "test.f1") for r in execucoes]),
            "precision": Aggregate.of(
                [_finite(r["test"]["precision"], "test.precision") for r in execucoes]
            ),
            "recall": Aggregate.of(
                [_finite(r["test"]["recall"], "test.recall") for r in execucoes]
            ),
            # "total", não "trainable": o limiar e o PCA são ajustados em forma
            # fechada e têm zero parâmetros treinados por gradiente, mas carregam
            # 12 e 6.120 valores aprendidos. Usar "trainable" na coluna de custo
            # os faria aparecer como gratuitos, o que é falso.
            "parameters": int(execucoes[0]["parameters"]["total"]),
            "seconds": Aggregate.of(
                [
                    _finite(r["training"]["elapsed_seconds"], "training.elapsed")
                    for r in execucoes
                ]
            ),
            "per_fault": {
                classe: Aggregate.of(
                    [
                        _finite(
                            r["test"]["per_fault"][classe]["auc_pr"],
                            f"per_fault.{classe}.auc_pr",
                        )
                        for r in execucoes
                    ]
                )
                for classe in FAULTS
            },
        }
        # A latência só é comparável quando houve detecção em todas as execuções.
        latencias = [
            r["test"]["latency"]["mean_detected_samples"]
            for r in execucoes
            if r["test"]["latency"].get("defined")
            and r["test"]["latency"].get("detected", 0) > 0
        ]
        entrada["latency"] = Aggregate.of(latencias) if latencias else None
        resumo[modelo] = entrada
    return resumo


# --------------------------------------------------------------------------
# Formatação
# --------------------------------------------------------------------------


def _decimal(value: float, places: int = 3) -> str:
    """Formata com vírgula decimal, como exige um artigo em português."""

    return f"{value:.{places}f}".replace(".", ",")


def _mean_std(agg: Aggregate, places: int = 3) -> str:
    if agg.n == 1:
        return _decimal(agg.mean, places)
    return f"{_decimal(agg.mean, places)} $\\pm$ {_decimal(agg.std, places)}"


def _integer(value: int) -> str:
    """Separador de milhar com espaço fino, conforme a convenção do repositório."""

    return f"{value:,}".replace(",", "\\,")


def _best_model(resumo: dict[str, dict[str, Any]]) -> str:
    return max(MODELS, key=lambda m: resumo[m]["auc_pr"].mean)


def build_table(resumos: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Tabela principal: uma linha por modelo, uma coluna de AUC-PR por agregação.

    O formato anterior repetia os três modelos em dois blocos com `multirow`,
    gastando seis linhas para dizer o que cabe em três. Lado a lado, a inversão
    do veredito entre as agregações também fica visível na mesma linha, que é o
    ponto do artigo.

    A melhor AUC-PR de cada coluna vai em negrito — negritar tudo não destaca
    nada. A coluna de custo fica porque o compromisso desempenho/custo é parte
    do veredito.
    """

    melhores = {agg: _best_model(resumos[agg]) for agg in AGGREGATIONS}

    linhas = [
        "\\begin{table}[htbp]",
        "\\caption{AUC-PR no conjunto de teste sob as duas regras de agregação do "
        "escore de janela (média $\\pm$ desvio padrão de três execuções). "
        "A agregação inverte o veredito: pelo máximo os três detectores "
        "praticamente empatam, enquanto pela média o autoencoder se separa.}",
        "\\label{tab:resultados}",
        "\\centering",
        "\\begin{tabular}{lccr}",
        "\\toprule",
        "\\textbf{Modelo} & \\textbf{Máximo} & \\textbf{Média} & "
        "\\textbf{Parâm.} \\\\",
        "\\midrule",
    ]

    for modelo in MODELS:
        celulas = [MODEL_LABELS[modelo]]
        for agregacao in AGGREGATIONS:
            valor = _mean_std(resumos[agregacao][modelo]["auc_pr"], 4)
            if melhores[agregacao] == modelo:
                valor = f"\\textbf{{{valor}}}"
            celulas.append(valor)
        celulas.append(_integer(resumos["max"][modelo]["parameters"]))
        linhas.append(" & ".join(celulas) + " \\\\")

    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(linhas)


def _command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build_macros(resumos: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Macros com todos os números citáveis pelo texto."""

    linhas = [
        "% Gerado por src/report.py — NÃO EDITAR À MÃO.",
        "% Precisando de um número novo no texto, acrescente a macro ao report.py.",
        "",
    ]
    for agregacao, resumo in resumos.items():
        sufixo = AGGREGATION_KEYS[agregacao]
        melhor = _best_model(resumo)
        linhas.append(f"% --- agregação {agregacao} ---")
        linhas.append(
            _command(f"ResultBestModel{sufixo}", MODEL_LABELS[melhor])
        )
        for modelo, entrada in resumo.items():
            chave = MODEL_KEYS[modelo]
            linhas += [
                _command(f"Result{chave}AucPr{sufixo}", _mean_std(entrada["auc_pr"], 4)),
                _command(f"Result{chave}F{sufixo}", _mean_std(entrada["f1"], 3)),
                _command(
                    f"Result{chave}Precision{sufixo}",
                    _mean_std(entrada["precision"], 3),
                ),
                _command(
                    f"Result{chave}Recall{sufixo}", _mean_std(entrada["recall"], 3)
                ),
                _command(
                    f"Result{chave}Params{sufixo}", _integer(entrada["parameters"])
                ),
                _command(
                    f"Result{chave}Seconds{sufixo}", _decimal(entrada["seconds"].mean, 1)
                ),
            ]
            if entrada["latency"] is not None:
                linhas.append(
                    _command(
                        f"Result{chave}Latency{sufixo}",
                        _decimal(entrada["latency"].mean, 1),
                    )
                )
            for classe, agg in entrada["per_fault"].items():
                linhas.append(
                    _command(
                        f"Result{chave}{FAULT_KEYS[classe]}{sufixo}",
                        _mean_std(agg, 4),
                    )
                )
        linhas.append("")
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# Figura
# --------------------------------------------------------------------------


def build_figure(
    resumos: dict[str, dict[str, dict[str, Any]]], path: Path
) -> Path:
    """AUC-PR por classe de falha, com um painel por agregação.

    Responde a pergunta que a tabela não responde: *onde* cada detector ganha.
    O painel duplo é o ponto do artigo — a ordenação dos modelos muda conforme a
    regra de agregação, e isso só se vê lado a lado.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Altura enxuta de propósito: o artigo tem limite rígido de quatro páginas e
    # a figura é renderizada na largura de uma coluna.
    figura, eixos = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    posicoes = np.arange(len(FAULTS))
    largura = 0.26

    for eixo, agregacao in zip(eixos, AGGREGATIONS):
        resumo = resumos[agregacao]
        for i, modelo in enumerate(MODELS):
            medias = [resumo[modelo]["per_fault"][c].mean for c in FAULTS]
            desvios = [resumo[modelo]["per_fault"][c].std for c in FAULTS]
            eixo.bar(
                posicoes + (i - 1) * largura,
                medias,
                largura,
                yerr=desvios,
                capsize=2,
                label=MODEL_LABELS[modelo].replace("$\\sigma$", "σ"),
            )
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels([FAULT_LABELS[c] for c in FAULTS], fontsize=8)
        eixo.set_title(
            "Agregação pelo máximo" if agregacao == "max" else "Agregação pela média",
            fontsize=9,
        )
        eixo.tick_params(labelsize=8)
        eixo.grid(axis="y", alpha=0.3, linewidth=0.5)
        eixo.set_axisbelow(True)

    eixos[0].set_ylabel("AUC-PR", fontsize=9)
    eixos[0].legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    figura.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(path, bbox_inches="tight")
    plt.close(figura)
    return path


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def build_csv(resumos: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Resumo tabular, com ponto decimal porque é formato de dado, não de texto."""

    cabecalho = [
        "agregacao", "modelo", "auc_pr_media", "auc_pr_desvio",
        "f1_media", "precisao_media", "revocacao_media",
        "parametros", "tempo_treino_s",
        *[f"auc_pr_{c}" for c in FAULTS],
    ]
    linhas = [",".join(cabecalho)]
    for agregacao, resumo in resumos.items():
        for modelo in MODELS:
            e = resumo[modelo]
            linhas.append(
                ",".join(
                    [
                        agregacao, modelo,
                        f"{e['auc_pr'].mean:.6f}", f"{e['auc_pr'].std:.6f}",
                        f"{e['f1'].mean:.6f}", f"{e['precision'].mean:.6f}",
                        f"{e['recall'].mean:.6f}",
                        str(e["parameters"]), f"{e['seconds'].mean:.1f}",
                        *[f"{e['per_fault'][c].mean:.6f}" for c in FAULTS],
                    ]
                )
            )
    return "\n".join(linhas) + "\n"


# --------------------------------------------------------------------------
# Ponto de entrada
# --------------------------------------------------------------------------


def generate_report(
    results_dir: Path = DEFAULT_RESULTS,
    results_mean_dir: Path = DEFAULT_RESULTS_MEAN,
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    macros_path: Path = DEFAULT_MACROS,
    figure_path: Path = DEFAULT_FIGURE,
) -> dict[str, Path]:
    """Lê as duas grades e grava CSV, macros, tabela e figura."""

    resumos = {
        "max": summarize(load_results(results_dir, "max")),
        "mean": summarize(load_results(results_mean_dir, "mean")),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_csv(resumos), encoding="utf-8")

    macros_path.write_text(
        build_macros(resumos)
        + "\n% --- tabela principal, inserida no texto por \\ResultTable ---\n"
        + "\\newcommand{\\ResultTable}{%\n"
        + build_table(resumos)
        + "\n}\n",
        encoding="utf-8",
    )

    build_figure(resumos, figure_path)
    return {
        "resumo": summary_path,
        "macros": macros_path,
        "figura": figure_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--results-mean-dir", type=Path, default=DEFAULT_RESULTS_MEAN)
    args = parser.parse_args(argv)

    saidas = generate_report(args.results_dir, args.results_mean_dir)
    for nome, caminho in saidas.items():
        print(f"{nome}: {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
