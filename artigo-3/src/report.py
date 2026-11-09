"""Consolida os resultados do test em CSV, macros LaTeX, tabela e figura (#54).

Regra permanente do repositório: nenhum número do artigo é digitado à mão.
Este script lê os resultados brutos e escreve:

- ``src/results/resumo.csv``: uma linha por configuração;
- ``paper/results-generated.tex``: uma macro por número usado no texto;
- ``paper/tabela-principal.tex``: a tabela comparativa (booktabs);
- ``paper/figs/compromisso.pdf``: MRR@5 contra tokens de entrada do gerador;
- ``paper/figs/sobreposicao-pareada.pdf``: diferenças de 20% menos 0%.

Incerteza: há uma execução por configuração (decodificação gulosa), então a
dispersão vem das perguntas. Intervalos de 95% por bootstrap percentil sobre
as perguntas respondíveis; as diferenças contra a linha de base ``slide`` são
pareadas (a mesma reamostra de perguntas para as duas configurações).

A avaliação principal pode ser humana ou automática, mas deve ser selecionada
explicitamente. As planilhas individuais medem concordância; quando houver um
arquivo de consenso, ele fornece as notas agregadas usadas no artigo.

Uso::

    python -m src.report
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
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


def carregar_juizes(avaliacao: Path, caminhos: list[Path] | None = None) -> dict[str, dict[str, dict]]:
    """{juiz: {item: notas}} para caminhos explícitos ou ``juiz-*.csv``."""
    from src.avaliacao_humana import ler_avaliacao

    arquivos = caminhos if caminhos is not None else sorted(avaliacao.glob("juiz-*.csv"))
    return {p.stem: ler_avaliacao(p) for p in arquivos}


def carregar_consenso(caminho: Path | None, juizes: dict[str, dict[str, dict]]) -> dict[str, dict] | None:
    """Lê o consenso humano, incluindo automaticamente os itens sem divergência."""
    if caminho is None:
        return None
    if len(juizes) != 2:
        raise ReportError("o consenso exige exatamente duas avaliações individuais")
    from src.avaliacao_humana import ler_consenso

    (_, j1), (_, j2) = sorted(juizes.items())
    return ler_consenso(caminho, j1, j2)


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
        if r.get("categoria") == "multitrecho":
            sucesso = float(m["todas_evidencias_at_k"]["5"])
        else:
            # Em factual/contextual, páginas múltiplas são evidências alternativas.
            sucesso = float(m["recall_at_k"]["5"] > 0)
        saida[c]["sucesso5"][q] = sucesso
        if r.get("categoria") == "multitrecho":
            saida[c]["todas5_multi"][q] = float(m["todas_evidencias_at_k"]["5"])
    return saida


# Palavras vazias frequentes do português; removidas antes do F1 de tokens.
PALAVRAS_VAZIAS_V1 = set(
    "a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para "
    "com sem e ou que se ao aos a as e nao mais como mas sua seu suas seus ele ela eles elas isso "
    "esse essa este esta entre sobre ate quando ja tambem so pode podem ser sao foi ha".split()
)
PALAVRAS_VAZIAS = PALAVRAS_VAZIAS_V1 - {"nao", "sem"}


def tokens_normalizados(texto: str, *, versao_original: bool = False) -> list[str]:
    """Minúsculas, sem acento, sem marcadores de citação e sem palavras vazias."""
    texto = re.sub(r"\[\d+(?:\s*[,;]\s*\d+)*\]", " ", texto)
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    vazias = PALAVRAS_VAZIAS_V1 if versao_original else PALAVRAS_VAZIAS
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in vazias]


def _f1_tokens(resposta: str, referencia: str, *, versao_original: bool) -> float:
    from collections import Counter

    r = Counter(tokens_normalizados(resposta, versao_original=versao_original))
    g = Counter(tokens_normalizados(referencia, versao_original=versao_original))
    comum = sum((r & g).values())
    if comum == 0:
        return 0.0
    precisao, revocacao = comum / sum(r.values()), comum / sum(g.values())
    return 2 * precisao * revocacao / (precisao + revocacao)


def f1_tokens_v1(resposta: str, referencia: str) -> float:
    """Versão publicada originalmente, que removia também negações."""
    return _f1_tokens(resposta, referencia, versao_original=True)


def f1_tokens(resposta: str, referencia: str) -> float:
    """Diagnóstico lexical que preserva ``não`` e ``sem``."""
    return _f1_tokens(resposta, referencia, versao_original=False)


def _trecho_relevante(trecho: dict, evidencias: set[tuple[int, int]]) -> bool:
    return any((trecho["aula"], p) in evidencias for p in range(trecho["pagina_inicio"], trecho["pagina_fim"] + 1))


def por_config_geracao(brutos: list[dict], referencias: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    """{config: {métrica: {pergunta: valor}}} nas respondíveis.

    - ``f1_tokens_v1``: versão publicada, mantida para rastreabilidade;
    - ``f1_lexical``: variante que preserva negações;
    - ``precisao_citacao``: fração dos trechos citados que contêm uma página-evidência,
      só nas respostas com ao menos uma citação válida.
    """
    saida: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for b in brutos:
        if not b["respondivel"]:
            continue
        c, q = b["config_id"], b["pergunta_id"]
        saida[c]["f1_tokens_v1"][q] = f1_tokens_v1(b["resposta"], referencias[q])
        saida[c]["f1_lexical"][q] = f1_tokens(b["resposta"], referencias[q])
        if b["citacoes"] and not b["abstencao"]:
            ev = {(int(a), int(p)) for a, p in b["evidencias_validas"]}
            citados = [b["trechos"][i - 1] for i in b["citacoes"] if 1 <= i <= len(b["trechos"])]
            if citados:
                saida[c]["precisao_citacao"][q] = fmean(_trecho_relevante(t, ev) for t in citados)
    return saida


def resumo_geracao(brutos: list[dict]) -> dict[str, dict[str, float]]:
    por: dict[str, list[dict]] = defaultdict(list)
    for b in brutos:
        por[b["config_id"]].append(b)
    saida = {}
    for c, rs in por.items():
        nao = [r for r in rs if not r["respondivel"]]
        sim = [r for r in rs if r["respondivel"]]
        # Cobertura de citação só é definida em respostas respondíveis e não
        # abstidas. A relevância é calculada separadamente apenas onde existe
        # ao menos uma citação válida.
        subst = [r for r in sim if not r["abstencao"]]
        com_citacao = [r for r in subst if r["citacoes"]]
        saida[c] = {
            "abstencao_correta": fmean(r["abstencao"] for r in nao) if nao else float("nan"),
            "abstencao_indevida": fmean(r["abstencao"] for r in sim) if sim else float("nan"),
            "com_citacao": fmean(bool(r["citacoes"]) for r in subst) if subst else float("nan"),
            "citacao_invalida": fmean(bool(r["citacoes_invalidas"]) for r in subst) if subst else float("nan"),
            "tokens_entrada": fmean(r["tokens_entrada"] for r in rs),
            "latencia_geracao": fmean(r["latencia_geracao_s"] for r in rs),
            "n_respondiveis": len(sim),
            "n_nao_respondiveis": len(nao),
            "n_respostas_substantivas": len(subst),
            "n_respostas_com_citacao": len(com_citacao),
        }
    return saida


def resumo_juizes(
    juizes: dict[str, dict[str, dict]],
    chave: dict[str, tuple[str, str]],
    consenso: dict[str, dict] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Concordância individual e qualidade agregada pela avaliação principal.

    Quando ``consenso`` é informado, suas notas são o resultado principal. Sem
    consenso, a média dos dois avaliadores é mantida apenas para compatibilidade
    com a análise automática antiga.
    """
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
        notas = consenso[i] if consenso is not None else {c: (j1[i][c] + j2[i][c]) / 2 for c in RUBRICA}
        por[chave[i][1]].append(notas)
    agregado = {
        cfg: {
            "correcao": fmean(n["correcao"] for n in ns) / 2,  # 0–1
            "completude": fmean(n["completude"] for n in ns) / 2,
            "fidelidade": fmean(n["fidelidade"] for n in ns) / 2,
            "nao_sustentadas": fmean(n["afirmacoes_nao_sustentadas"] for n in ns),
        }
        for cfg, ns in por.items()
    }
    return agregado, kappas


# --------------------------------------------------------------------------- saída


def fmt(x: float, casas: int = 2) -> str:
    """Vírgula decimal sem espaço extra no modo matemático."""
    texto = f"{x:.{casas}f}".replace(".", "{,}")
    if texto.startswith("-"):
        # sinal de menos tipográfico; "-0,00" vira "0,00"
        texto = texto[1:] if set(texto[1:]) <= set("0{,}") else "\\ensuremath{-}" + texto[1:]
    return texto


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
    referencias: dict[str, str] | None = None,
    consenso: dict[str, dict] | None = None,
    avaliacao_principal: str = "automatica",
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[list[dict], list[str], str]:
    """Devolve (linhas do resumo, macros, tabela LaTeX)."""
    rec = por_config_recuperacao(rankings)
    ger = resumo_geracao(brutos)
    gpq = por_config_geracao(brutos, referencias) if referencias else {}
    configs = [c for c in CONFIG.chunking.ids if c in rec]
    if set(configs) != set(ger):
        raise ReportError(f"recuperação {sorted(rec)} e geração {sorted(ger)} cobrem configurações diferentes")
    if LINHA_DE_BASE not in rec:
        raise ReportError("a linha de base 'slide' não está nos resultados")
    if avaliacao_principal not in {"humana", "automatica"}:
        raise ReportError("avaliação principal deve ser 'humana' ou 'automatica'")
    if avaliacao_principal == "humana" and consenso is None:
        raise ReportError("a avaliação humana principal exige um arquivo de consenso")
    qual, kappas = (resumo_juizes(juizes, chave, consenso) if juizes else (None, None))

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
        for met, nome in (("recall1", "RecallUm"), ("recall5", "RecallCinco"), ("mrr5", "MRR"),
                          ("precision5", "PrecisaoCinco"), ("sucesso5", "SucessoCinco")):
            vals = list(rec[c][met].values())
            media = fmean(vals)
            lo, hi = bootstrap_ic(vals, n=n_bootstrap)
            linha |= {met: media, f"{met}_ic_inf": lo, f"{met}_ic_sup": hi}
            macros += [macro(f"{nome}{s}", fmt(media)), macro(f"{nome}{s}IC", f"\\mbox{{[{fmt(lo)}; {fmt(hi)}]}}")]
            if c != LINHA_DE_BASE and met in ("recall5", "mrr5", "sucesso5"):
                d, dlo, dhi = bootstrap_diferenca(rec[c][met], rec[LINHA_DE_BASE][met], n=n_bootstrap)
                linha |= {f"{met}_dif_slide": d, f"{met}_dif_ic_inf": dlo, f"{met}_dif_ic_sup": dhi}
                macros += [
                    macro(f"Dif{nome}{s}", fmt(d)),
                    macro(f"Dif{nome}{s}IC", f"\\mbox{{[{fmt(dlo)}; {fmt(dhi)}]}}"),
                ]
        if rec[c].get("todas5_multi"):
            v = fmean(rec[c]["todas5_multi"].values())
            linha["todas5_multi"] = v
            macros.append(macro(f"TodasMulti{s}", fmt(v)))
        for met, nome in (("f1_tokens_v1", "FUmVUm"), ("f1_lexical", "FUmLexical"),
                          ("precisao_citacao", "RelevanciaCitacao")):
            if c in gpq and gpq[c].get(met):
                vals = list(gpq[c][met].values())
                lo, hi = bootstrap_ic(vals, n=n_bootstrap)
                linha |= {met: fmean(vals), f"{met}_ic_inf": lo, f"{met}_ic_sup": hi}
                macros += [macro(f"{nome}{s}", fmt(fmean(vals))), macro(f"{nome}{s}IC", f"\\mbox{{[{fmt(lo)}; {fmt(hi)}]}}")]
                if c != LINHA_DE_BASE:
                    d, dlo, dhi = bootstrap_diferenca(gpq[c][met], gpq[LINHA_DE_BASE][met], n=n_bootstrap)
                    macros += [macro(f"Dif{nome}{s}", fmt(d)), macro(f"Dif{nome}{s}IC", f"\\mbox{{[{fmt(dlo)}; {fmt(dhi)}]}}")]
        g = ger[c]
        linha |= g
        macros += [
            macro(f"AbstencaoCorreta{s}", fmt_pct(g["abstencao_correta"])),
            macro(f"AbstencaoIndevida{s}", fmt_pct(g["abstencao_indevida"])),
            macro(f"ComCitacao{s}", fmt_pct(g["com_citacao"])),
            macro(f"CitacaoInvalida{s}", fmt_pct(g["citacao_invalida"])),
            macro(f"TokensEntrada{s}", f"{g['tokens_entrada']:.0f}"),
            macro(f"LatenciaGeracao{s}", fmt(g["latencia_geracao"], 1)),
            macro(f"NRespondiveis{s}", str(g["n_respondiveis"])),
            macro(f"NRespostasSubstantivas{s}", str(g["n_respostas_substantivas"])),
            macro(f"NRespostasComCitacao{s}", str(g["n_respostas_com_citacao"])),
        ]
        for parte in ("V", "VI", "VII"):
            vals = [
                r["metricas_evidencia"]["recall_at_k"]["5"]
                for r in rankings
                if r["config_id"] == c and r["respondivel"] and r.get("parte") == parte and r["split"] == "test"
            ]
            if vals:
                linha[f"recall5_parte_{parte}"] = fmean(vals)
                macros.append(macro(f"RecallCinco{s}Parte{parte}", fmt(fmean(vals))))
        if qual:
            q = qual[c]
            linha |= {f"juiz_{k}": v for k, v in q.items()}
            macros += [
                macro(f"Correcao{s}", fmt(q["correcao"])),
                macro(f"Completude{s}", fmt(q["completude"])),
                macro(f"Fidelidade{s}", fmt(q["fidelidade"])),
                macro(f"NaoSustentadas{s}", fmt(q["nao_sustentadas"])),
            ]
        linhas.append(linha)

    if kappas:
        nomes = {"correcao": "Correcao", "completude": "Completude", "fidelidade": "Fidelidade",
                 "afirmacoes_nao_sustentadas": "NaoSustentadas"}
        macros += [macro(f"Kappa{nomes[k]}", fmt(v)) for k, v in kappas.items()]
        macros.append(macro("NItensJuizes", str(len(chave))))

    if juizes:
        from src.analise_geracao import desfechos

        (_, j1), (_, j2) = sorted(juizes.items())
        notas_desfecho = consenso or {i: {k: min(j1[i][k], j2[i][k]) for k in j1[i]} for i in chave}
        por_cfg = desfechos(brutos, notas_desfecho, chave)
        for l in linhas:
            l |= {f"desfecho_{d}": n for d, n in por_cfg[l["config_id"]].items()}
        total = {d: sum(v[d] for v in por_cfg.values()) for d in next(iter(por_cfg.values()))}
        nomes = {
            "correta_cobertura_completa": "CorretaCoberturaCompleta",
            "correta_cobertura_parcial": "CorretaCoberturaParcial",
            "correta_cobertura_ausente": "CorretaCoberturaAusente",
            "incorreta_cobertura_completa": "IncorretaCoberturaCompleta",
            "incorreta_cobertura_parcial": "IncorretaCoberturaParcial",
            "incorreta_cobertura_ausente": "IncorretaCoberturaAusente",
            "abstencao_correta": "AbstencaoCorretaAmostra",
            "resposta_inventada": "RespostaInventada",
        }
        macros += [macro(f"N{nomes[d]}", str(n)) for d, n in total.items()]
        categorias_resp = tuple(d for d in total if d.startswith("correta_") or d.startswith("incorreta_"))
        macros.append(macro("NItensRespondiveisAmostra", str(sum(total[d] for d in categorias_resp))))

    # Comparações pareadas que respondem diretamente à pergunta sobreposição:
    # 20% menos 0%, dentro de cada tamanho e sobre as mesmas perguntas.
    for com, sem, nome in (("c128-o20", "c128-o0", "CPeq"),
                           ("c256-o20", "c256-o0", "CMed"),
                           ("c512-o20", "c512-o0", "CGrd")):
        if com not in rec or sem not in rec:
            continue
        for met, rotulo in (("recall5", "RecallCinco"), ("mrr5", "MRR"), ("sucesso5", "SucessoCinco")):
            d, lo, hi = bootstrap_diferenca(rec[com][met], rec[sem][met], n=n_bootstrap)
            ids = set(rec[com][met]) & set(rec[sem][met])
            macros += [
                macro(f"DifOverlap{rotulo}{nome}", fmt(d)),
                macro(f"DifOverlap{rotulo}{nome}IC", f"\\mbox{{[{fmt(lo)}; {fmt(hi)}]}}"),
                macro(f"NOverlap{rotulo}{nome}", str(len(ids))),
            ]
            for linha in linhas:
                if linha["config_id"] == com:
                    linha[f"{met}_dif_overlap"] = d
                    linha[f"{met}_dif_overlap_ic_inf"] = lo
                    linha[f"{met}_dif_overlap_ic_sup"] = hi
                    linha[f"{met}_dif_overlap_n"] = len(ids)

    # Destaques derivados, para o texto não depender de leitura manual da tabela.
    melhor_mrr = max(linhas, key=lambda l: l["mrr5"])
    melhor_r5 = max(linhas, key=lambda l: l["recall5"])
    macros += [
        macro("MelhorMRRConfig", ROTULO[melhor_mrr["config_id"]]),
        macro("MelhorRecallCincoConfig", ROTULO[melhor_r5["config_id"]]),
    ]
    if gpq:
        n_cit = sum(len(gpq[c].get("precisao_citacao", {})) for c in gpq)
        macros.append(macro("NObservacoesRelevanciaCitacao", str(n_cit)))
    macros.append(macro("AvaliacaoPrincipal", "humana" if avaliacao_principal == "humana" else "automática"))
    return linhas, macros, tabela_latex(linhas, com_juizes=bool(qual))


def tabela_latex(linhas: list[dict], *, com_juizes: bool) -> str:
    """Tabela larga (duas colunas), agrupada em recuperação, resposta e custo."""
    def tem(m):
        return all(m in l for l in linhas)

    rec = [("recall5", "R@5"), ("mrr5", "MRR@5"), ("sucesso5", "Suc.@5")]
    if tem("todas5_multi"):
        rec.append(("todas5_multi", "Multi@5"))
    resp = [("com_citacao", "Cob. cit.")]
    resp += [(m, n) for m, n in (("precisao_citacao", "Rel. cit."),) if tem(m)]
    if com_juizes:
        resp += [("juiz_correcao", "Corr."), ("juiz_completude", "Compl."),
                 ("juiz_fidelidade", "Fid."), ("juiz_nao_sustentadas", "N. sust.")]
    custo = [("tokens_entrada", "Tok. ent."), ("latencia_geracao", "Lat. (s)")]
    maior_melhor = {m for m, _ in rec + resp if m != "juiz_nao_sustentadas"}
    melhor = {m: max(l[m] for l in linhas) for m in maior_melhor}
    if com_juizes:
        melhor["juiz_nao_sustentadas"] = min(l["juiz_nao_sustentadas"] for l in linhas)
    melhor |= {m: min(l[m] for l in linhas) for m, _ in custo}

    def cel(l, m):
        if m == "com_citacao":
            v = fmt_pct(l[m])
        elif m == "tokens_entrada":
            v = f"{l[m]:.0f}"
        elif m == "latencia_geracao":
            v = fmt(l[m], 1)
        else:
            v = fmt(l[m])
        return f"\\textbf{{{v}}}" if l[m] == melhor[m] else v

    grupos = [("Recuperação", rec), ("Resposta", resp), ("Custo", custo)]
    cols = [m for _, g in grupos for m, _ in g]
    topo, regras, inicio = ["Config."], [], 2
    for nome, g in grupos:
        topo.append(f"\\multicolumn{{{len(g)}}}{{c}}{{{nome}}}")
        regras.append(f"\\cmidrule(lr){{{inicio}-{inicio + len(g) - 1}}}")
        inicio += len(g)
    cab = [""] + [n for _, g in grupos for _, n in g]
    corpo = [" & ".join([ROTULO[l["config_id"]], *(cel(l, m) for m in cols)]) + " \\\\" for l in linhas]
    return "\n".join([
        "% Gerado por src/report.py — não editar à mão.",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        " & ".join(topo) + " \\\\",
        " ".join(regras),
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
    ax.set_xlabel("Tokens de entrada do gerador por pergunta (média)", fontsize=7)
    ax.set_ylabel("MRR@5", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figura_partes(linhas: list[dict], destino: Path) -> None:
    """Mapa de calor do Recall@5 por parte do curso e configuração."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    partes = ("V", "VI", "VII")
    nomes = {"V": "V: PLN e LLMs", "VI": "VI: Aprend. por reforço", "VII": "VII: XAI e equidade"}
    dados = [[l[f"recall5_parte_{p}"] for l in linhas] for p in partes]
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    im = ax.imshow(dados, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
    for i, linha in enumerate(dados):
        for j, v in enumerate(linha):
            ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=6,
                    color="white" if v > 0.85 else "black")
    ax.set_xticks(range(len(linhas)), [ROTULO[l["config_id"]].replace("\\%", "%") for l in linhas],
                  fontsize=5.5, rotation=30, ha="right")
    ax.set_yticks(range(len(partes)), [nomes[p] for p in partes], fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Recall@5", fontsize=6)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


DESFECHOS_FIG = (
    ("desfecho_correta_cobertura_completa", "Correta, cob. completa", "#2b8a3e"),
    ("desfecho_correta_cobertura_parcial", "Correta, cob. parcial", "#74b816"),
    ("desfecho_correta_cobertura_ausente", "Correta, cob. ausente", "#a9e34b"),
    ("desfecho_incorreta_cobertura_completa", "Incorreta, cob. completa", "#e8590c"),
    ("desfecho_incorreta_cobertura_parcial", "Incorreta, cob. parcial", "#f08c00"),
    ("desfecho_incorreta_cobertura_ausente", "Incorreta, cob. ausente", "#c92a2a"),
    ("desfecho_abstencao_correta", "Abstenção correta", "#868e96"),
)


def figura_sobreposicao(linhas: list[dict], destino: Path) -> None:
    """Diferenças pareadas de 20% menos 0% de sobreposição por tamanho."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pares = [l for l in linhas if "recall5_dif_overlap" in l]
    if not pares:
        raise ReportError("não há pares de sobreposição para a figura")
    metricas = (("recall5", "Recall@5", "o"), ("mrr5", "MRR@5", "s"),
                ("sucesso5", "Sucesso@5", "^"))
    x = list(range(len(pares)))
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    deslocamentos = (-0.15, 0.0, 0.15)
    for (met, rotulo, marcador), dx in zip(metricas, deslocamentos):
        y = [l[f"{met}_dif_overlap"] for l in pares]
        lo = [l[f"{met}_dif_overlap_ic_inf"] for l in pares]
        hi = [l[f"{met}_dif_overlap_ic_sup"] for l in pares]
        ax.errorbar([v + dx for v in x], y,
                    yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]],
                    fmt=marcador, ms=3.5, capsize=2, lw=0.8, label=rotulo)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [l["config_id"].split("-")[0][1:] + " tokens" for l in pares], fontsize=6)
    ax.set_ylabel("Diferença: 20% − 0%", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.3, ncol=3, frameon=False, loc="upper center")
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figura_desfechos(linhas: list[dict], destino: Path) -> None:
    """Barras empilhadas dos desfechos da amostra julgada, por configuração."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    x = range(len(linhas))
    base = [0] * len(linhas)
    for chave, rotulo, cor in DESFECHOS_FIG:
        vals = [l.get(chave, 0) for l in linhas]
        ax.bar(x, vals, bottom=base, color=cor, label=rotulo, width=0.7)
        base = [b + v for b, v in zip(base, vals)]
    ax.set_xticks(list(x), [ROTULO[l["config_id"]].replace("\\%", "%") for l in linhas], fontsize=5.5,
                  rotation=30, ha="right")
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Itens", fontsize=6)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=4.4, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
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


def macros_protocolo(resultados: Path = RESULTADOS, avaliacao: Path = AVALIACAO) -> list[str]:
    """Números de protocolo lidos dos artefatos, para não serem digitados no texto."""
    corpus = json.loads(CONFIG.corpus.manifesto_path.read_text(encoding="utf-8"))
    selecao = json.loads((resultados / "retrieval" / "dev" / "top-k-selection.json").read_text(encoding="utf-8"))
    saida = [
        macro("NPaginasCorpus", str(sum(a["n_paginas"] for a in corpus["arquivos"]))),
        macro("NAulasCorpus", str(len(corpus["arquivos"]))),
        macro("TopK", str(selecao["top_k_selecionado"])),
    ]
    amostra = avaliacao / "amostra.json"
    if amostra.is_file():
        saida.append(macro("NPerguntasAmostra", str(json.loads(amostra.read_text())["n_perguntas"])))
    return saida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados-dir", type=Path, default=RESULTADOS)
    parser.add_argument("--avaliacao-dir", type=Path, default=AVALIACAO)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--avaliacao-principal", choices=("humana", "automatica"), required=True)
    parser.add_argument("--avaliador", type=Path, action="append", dest="avaliadores",
                        help="CSV individual; informe exatamente duas vezes")
    parser.add_argument("--consenso", type=Path, help="CSV de divergências resolvidas; obrigatório para avaliação humana")
    args = parser.parse_args()
    try:
        from src.analise_geracao import ler_chave

        rankings = carregar_rankings(args.resultados_dir / "retrieval" / "test" / "rankings.jsonl")
        brutos = carregar_geracao(args.resultados_dir / "raw" / "test")
        if args.avaliadores is not None and len(args.avaliadores) != 2:
            raise ReportError("--avaliador deve ser informado exatamente duas vezes")
        juizes = carregar_juizes(args.avaliacao_dir, args.avaliadores)
        chave = ler_chave(args.avaliacao_dir / "chave.csv") if juizes else None
        consenso = carregar_consenso(args.consenso, juizes)
        from src.dataset import carregar

        referencias = {it.id: it.resposta_referencia for it in carregar(split="test") if it.respondivel}
        linhas, macros, tabela = gerar(
            rankings, brutos, juizes or None, chave, referencias=referencias,
            consenso=consenso, avaliacao_principal=args.avaliacao_principal,
        )
        macros += macros_protocolo(args.resultados_dir, args.avaliacao_dir)
        escrever(linhas, macros, tabela, resultados=args.resultados_dir, paper=args.paper_dir)
        figura(linhas, args.paper_dir / "figs" / "compromisso.pdf")
        figura_sobreposicao(linhas, args.paper_dir / "figs" / "sobreposicao-pareada.pdf")
        if juizes:
            figura_desfechos(linhas, args.paper_dir / "figs" / "desfechos.pdf")
    except (ReportError, ValueError, FileNotFoundError) as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    print(f"{len(macros)} macros, {len(linhas)} configurações; juízes: {sorted(juizes) or 'nenhum'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
