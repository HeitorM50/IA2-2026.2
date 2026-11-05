"""Métricas auditáveis da recuperação e seleção de ``top-k`` somente em dev.

O módulo mantém o ranking bruto separado da geração. Cada chunk é relacionado
às páginas-evidência do dataset e os agregados podem, portanto, ser refeitos sem
consultar novamente o índice ou o gerador.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from src import dataset as ds
from src.config import ARTIGO, CONFIG, ExperimentoConfig, como_dict
from src.contratos import Recuperador, Trecho


VERSAO_AVALIACAO = 1
REGRA_SELECAO = "maior_recall_medio_dev_desempate_menor_k"
GRUPOS = ("geral", "V", "VI", "VII")


class AvaliacaoRecuperacaoError(ValueError):
    pass


def _json_bytes(objeto: object) -> bytes:
    return json.dumps(
        objeto, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(objeto: object) -> str:
    return hashlib.sha256(_json_bytes(objeto)).hexdigest()


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_protocolo(cfg: ExperimentoConfig = CONFIG) -> str:
    return sha256_json(como_dict(cfg, sem_caminhos=True))


def sha256_itens(itens: Sequence[ds.Item]) -> str:
    """Hash canônico de um split, sem depender da serialização do outro split."""
    return sha256_json([asdict(item) for item in sorted(itens, key=lambda i: i.id)])


def _evidencia_id(evidencia: tuple[int, int]) -> str:
    return f"a{evidencia[0]:02d}-p{evidencia[1]:04d}"


def evidencias_do_trecho(
    trecho: Trecho, evidencias: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Relaciona por aula e intervalo inclusivo de páginas do chunk."""
    return tuple(
        evidencia
        for evidencia in evidencias
        if trecho.aula == evidencia[0]
        and trecho.pagina_inicio <= evidencia[1] <= trecho.pagina_fim
    )


def _iou_tokens(a: Trecho, b: Trecho) -> float:
    if a.aula != b.aula:
        return 0.0
    intersecao = max(0, min(a.token_fim, b.token_fim) - max(a.token_inicio, b.token_inicio))
    uniao = max(a.token_fim, b.token_fim) - min(a.token_inicio, b.token_inicio)
    return intersecao / uniao if uniao > 0 else 0.0


def redundancia_tokens(trechos: Sequence[Trecho], k: int) -> float:
    selecionados = list(trechos[:k])
    if len(selecionados) < 2:
        return 0.0
    pares = [
        _iou_tokens(selecionados[i], selecionados[j])
        for i in range(len(selecionados))
        for j in range(i + 1, len(selecionados))
    ]
    return statistics.fmean(pares)


def avaliar_ranking(
    trechos: Sequence[Trecho],
    evidencias: Sequence[tuple[int, int]],
    *,
    respondivel: bool,
    ks: Sequence[int] = CONFIG.recuperacao.ks_metricas,
) -> dict:
    """Materializa ranking, correspondências e métricas de uma pergunta."""
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] <= 0:
        raise AvaliacaoRecuperacaoError("ks deve conter inteiros positivos")
    profundidade = max(ks)
    trechos = list(trechos[:profundidade])
    evidencias = tuple(dict.fromkeys(evidencias))
    if respondivel and not evidencias:
        raise AvaliacaoRecuperacaoError("pergunta respondível sem evidência")
    if not respondivel and evidencias:
        raise AvaliacaoRecuperacaoError("pergunta não respondível com evidência")

    ranking = []
    primeiras: dict[tuple[int, int], int | None] = {e: None for e in evidencias}
    for posicao, trecho in enumerate(trechos, start=1):
        correspondentes = evidencias_do_trecho(trecho, evidencias)
        for evidencia in correspondentes:
            if primeiras[evidencia] is None:
                primeiras[evidencia] = posicao
        ranking.append(
            {
                "posicao": posicao,
                **trecho.como_dict(),
                "evidencias_correspondentes": [
                    _evidencia_id(e) for e in correspondentes
                ],
                "relevante": bool(correspondentes),
            }
        )

    diagnostico_evidencias = [
        {
            "id": _evidencia_id(evidencia),
            "aula": evidencia[0],
            "pagina": evidencia[1],
            "primeira_posicao": primeiras[evidencia],
            "estado": (
                "recuperada"
                if primeiras[evidencia] is not None
                else f"ausente_no_top_{profundidade}"
            ),
        }
        for evidencia in evidencias
    ]

    metricas = None
    if respondivel:
        recall: dict[str, float] = {}
        precision: dict[str, float] = {}
        todas: dict[str, bool] = {}
        for k in ks:
            recuperadas = sum(
                posicao is not None and posicao <= k for posicao in primeiras.values()
            )
            relevantes = sum(entrada["relevante"] for entrada in ranking[:k])
            recall[str(k)] = recuperadas / len(evidencias)
            precision[str(k)] = relevantes / k
            todas[str(k)] = recuperadas == len(evidencias)
        posicoes = [p for p in primeiras.values() if p is not None]
        primeira = min(posicoes) if posicoes else None
        metricas = {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "todas_evidencias_at_k": todas,
            f"mrr_at_{profundidade}": 1.0 / primeira if primeira else 0.0,
        }

    return {
        "ranking": ranking,
        "evidencias": diagnostico_evidencias,
        "metricas_evidencia": metricas,
        "redundancia_at_k": {
            str(k): redundancia_tokens(trechos, k) for k in ks
        },
    }


def _percentil(valores: Sequence[float], q: float) -> float | None:
    if not valores:
        return None
    ordenados = sorted(valores)
    posicao = (len(ordenados) - 1) * q
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    if inferior == superior:
        return ordenados[inferior]
    peso = posicao - inferior
    return ordenados[inferior] * (1 - peso) + ordenados[superior] * peso


def _media(valores: Iterable[float]) -> float | None:
    valores = list(valores)
    return statistics.fmean(valores) if valores else None


def _agregar_grupo(registros: Sequence[dict], ks: Sequence[int]) -> dict:
    avaliaveis = [r for r in registros if r["respondivel"]]
    latencias = [float(r["latencia_recuperacao_s"]) for r in registros]
    profundidade = max(ks)
    resultado = {
        "n_total": len(registros),
        "n_respondiveis": len(avaliaveis),
        "n_nao_respondiveis": len(registros) - len(avaliaveis),
        "recall_at_k": {},
        "precision_at_k": {},
        "todas_evidencias_at_k": {},
        f"mrr_at_{profundidade}": _media(
            r["metricas_evidencia"][f"mrr_at_{profundidade}"] for r in avaliaveis
        ),
        "redundancia_at_k": {},
        "latencia_recuperacao_s": {
            "media": _media(latencias),
            "mediana": statistics.median(latencias) if latencias else None,
            "p95": _percentil(latencias, 0.95),
        },
    }
    for k in ks:
        chave = str(k)
        resultado["recall_at_k"][chave] = _media(
            r["metricas_evidencia"]["recall_at_k"][chave] for r in avaliaveis
        )
        resultado["precision_at_k"][chave] = _media(
            r["metricas_evidencia"]["precision_at_k"][chave] for r in avaliaveis
        )
        resultado["todas_evidencias_at_k"][chave] = _media(
            float(r["metricas_evidencia"]["todas_evidencias_at_k"][chave])
            for r in avaliaveis
        )
        resultado["redundancia_at_k"][chave] = _media(
            r["redundancia_at_k"][chave] for r in registros
        )
    return resultado


def agregar(
    registros: Sequence[dict],
    ks: Sequence[int] = CONFIG.recuperacao.ks_metricas,
    *,
    manifesto_sha256: str | None = None,
    top_k_principal: int | None = None,
) -> dict:
    ks = tuple(sorted(set(ks)))
    configs = sorted({r["config_id"] for r in registros})
    por_configuracao = {}
    for config_id in configs:
        da_config = [r for r in registros if r["config_id"] == config_id]
        por_configuracao[config_id] = {
            grupo: _agregar_grupo(
                da_config if grupo == "geral" else [r for r in da_config if r["parte"] == grupo],
                ks,
            )
            for grupo in GRUPOS
        }
    return {
        "schema_version": VERSAO_AVALIACAO,
        "ks": list(ks),
        "top_k_principal": top_k_principal,
        "manifesto_avaliacao_sha256": manifesto_sha256,
        "por_configuracao": por_configuracao,
    }


def escolher_top_k(
    registros: Sequence[dict],
    *,
    configs_esperadas: Sequence[str],
    perguntas_esperadas: Sequence[str],
    ks: Sequence[int] = CONFIG.recuperacao.ks_metricas,
) -> tuple[int, dict[str, float]]:
    """Escolhe k pela média macro de Recall nos pares config×pergunta de dev."""
    if not registros or any(r.get("split") != "dev" for r in registros):
        raise AvaliacaoRecuperacaoError("seleção de top-k aceita somente rankings de dev")
    esperados = {
        (config_id, pergunta_id)
        for config_id in configs_esperadas
        for pergunta_id in perguntas_esperadas
    }
    observados = [(r["config_id"], r["pergunta_id"]) for r in registros]
    if len(observados) != len(set(observados)):
        raise AvaliacaoRecuperacaoError("ranking repetido para configuração/pergunta")
    if set(observados) != esperados:
        faltando = sorted(esperados - set(observados))[:5]
        extras = sorted(set(observados) - esperados)[:5]
        raise AvaliacaoRecuperacaoError(
            f"grade dev incompleta ou inesperada; faltando={faltando}, extras={extras}"
        )
    avaliaveis = [r for r in registros if r["respondivel"]]
    if not avaliaveis:
        raise AvaliacaoRecuperacaoError("dev não contém perguntas respondíveis")
    resultados = {
        str(k): statistics.fmean(
            r["metricas_evidencia"]["recall_at_k"][str(k)] for r in avaliaveis
        )
        for k in sorted(set(ks))
    }
    melhor = max(resultados.values())
    escolhido = min(k for k in sorted(set(ks)) if resultados[str(k)] == melhor)
    return escolhido, resultados


def montar_artefato_selecao(
    registros: Sequence[dict],
    manifesto: dict,
    rankings_sha256: str,
    cfg: ExperimentoConfig = CONFIG,
) -> dict:
    if manifesto.get("split") != "dev":
        raise AvaliacaoRecuperacaoError("o manifesto da seleção precisa ser de dev")
    escolhido, resultados = escolher_top_k(
        registros,
        configs_esperadas=cfg.chunking.ids,
        perguntas_esperadas=manifesto["perguntas"],
        ks=cfg.recuperacao.ks_metricas,
    )
    return {
        "schema_version": VERSAO_AVALIACAO,
        "regra": REGRA_SELECAO,
        "candidatos": list(cfg.recuperacao.ks_metricas),
        "top_k_selecionado": escolhido,
        "recall_medio_dev": resultados,
        "configs": list(cfg.chunking.ids),
        "perguntas_dev": list(manifesto["perguntas"]),
        "dev_dataset_sha256": manifesto["dataset_split_sha256"],
        "protocolo_sha256": manifesto["protocolo_sha256"],
        "corpus_manifesto_sha256": manifesto["corpus_manifesto_sha256"],
        "avaliacao_manifesto_sha256": sha256_json(manifesto),
        "rankings_sha256": rankings_sha256,
    }


def validar_selecao(
    artefato: dict,
    *,
    cfg: ExperimentoConfig,
    corpus_sha256: str,
    dev_dataset_sha256: str,
) -> int:
    exigidos = {
        "regra", "candidatos", "top_k_selecionado", "recall_medio_dev",
        "configs", "dev_dataset_sha256", "protocolo_sha256",
        "corpus_manifesto_sha256", "avaliacao_manifesto_sha256", "rankings_sha256",
    }
    faltando = sorted(exigidos - set(artefato))
    if faltando:
        raise AvaliacaoRecuperacaoError(f"seleção de top-k incompleta: {faltando}")
    if artefato["regra"] != REGRA_SELECAO:
        raise AvaliacaoRecuperacaoError("regra de seleção de top-k incompatível")
    if artefato["candidatos"] != list(cfg.recuperacao.ks_metricas):
        raise AvaliacaoRecuperacaoError("candidatos de top-k incompatíveis")
    if artefato["configs"] != list(cfg.chunking.ids):
        raise AvaliacaoRecuperacaoError("grade de configurações da seleção é incompatível")
    comparacoes = {
        "protocolo": (artefato["protocolo_sha256"], sha256_protocolo(cfg)),
        "corpus": (artefato["corpus_manifesto_sha256"], corpus_sha256),
        "dev": (artefato["dev_dataset_sha256"], dev_dataset_sha256),
    }
    divergentes = [nome for nome, (a, b) in comparacoes.items() if a != b]
    if divergentes:
        raise AvaliacaoRecuperacaoError(
            "seleção de top-k incompatível com " + ", ".join(divergentes)
        )
    resultados = artefato["recall_medio_dev"]
    try:
        valores = [float(resultados[str(k)]) for k in cfg.recuperacao.ks_metricas]
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in valores):
            raise ValueError("Recall fora de [0, 1]")
        if any(atual > seguinte for atual, seguinte in zip(valores, valores[1:])):
            raise ValueError("Recall@k não pode diminuir quando k aumenta")
        melhor = max(valores)
        esperado = min(
            k for k in cfg.recuperacao.ks_metricas
            if float(resultados[str(k)]) == melhor
        )
        observado = int(artefato["top_k_selecionado"])
    except (KeyError, TypeError, ValueError) as e:
        raise AvaliacaoRecuperacaoError("resultados da seleção de top-k inválidos") from e
    if observado != esperado:
        raise AvaliacaoRecuperacaoError("top-k não corresponde à regra registrada")
    return observado


def carregar_selecao(
    caminho: Path,
    *,
    cfg: ExperimentoConfig,
    corpus_sha256: str,
    dev_dataset_sha256: str,
) -> tuple[int, str]:
    try:
        artefato = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AvaliacaoRecuperacaoError(f"seleção de top-k inválida: {caminho}") from e
    top_k = validar_selecao(
        artefato,
        cfg=cfg,
        corpus_sha256=corpus_sha256,
        dev_dataset_sha256=dev_dataset_sha256,
    )
    return top_k, sha256_arquivo(caminho)


def _proveniencia_indice(recuperador: Recuperador) -> dict:
    cache_dir = getattr(recuperador, "cache_dir", None)
    caminho = Path(cache_dir) / "manifest.json" if cache_dir is not None else None
    if caminho is None or not caminho.is_file():
        return {"cache_key": None, "manifesto_sha256": None}
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    return {
        "cache_key": manifesto.get("cache_key"),
        "manifesto_sha256": sha256_arquivo(caminho),
    }


def _git_sha() -> str | None:
    processo = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ARTIGO, capture_output=True, text=True
    )
    return processo.stdout.strip() if processo.returncode == 0 else None


def _gravar_json(caminho: Path, dados: object) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gravar_csv(caminho: Path, resumo: dict) -> None:
    linhas = []
    profundidade = max(resumo["ks"])
    for config_id, grupos in resumo["por_configuracao"].items():
        for grupo, valores in grupos.items():
            linha = {
                "config_id": config_id,
                "grupo": grupo,
                "n_total": valores["n_total"],
                "n_respondiveis": valores["n_respondiveis"],
                "n_nao_respondiveis": valores["n_nao_respondiveis"],
                f"mrr_at_{profundidade}": valores[f"mrr_at_{profundidade}"],
                "latencia_media_s": valores["latencia_recuperacao_s"]["media"],
                "latencia_mediana_s": valores["latencia_recuperacao_s"]["mediana"],
                "latencia_p95_s": valores["latencia_recuperacao_s"]["p95"],
            }
            for k in resumo["ks"]:
                chave = str(k)
                linha[f"recall_at_{k}"] = valores["recall_at_k"][chave]
                linha[f"precision_at_{k}"] = valores["precision_at_k"][chave]
                linha[f"todas_evidencias_at_{k}"] = valores["todas_evidencias_at_k"][chave]
                linha[f"redundancia_at_{k}"] = valores["redundancia_at_k"][chave]
            linhas.append(linha)
    if not linhas:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def coletar(
    split: str,
    *,
    fabrica_recuperador: Callable[[str], Recuperador],
    cfg: ExperimentoConfig = CONFIG,
    saida: Path | None = None,
    selecao_top_k: Path | None = None,
) -> Path:
    """Consulta os sete índices uma vez por pergunta e grava artefatos auditáveis."""
    if split not in ds.SPLITS:
        raise AvaliacaoRecuperacaoError(f"split inválido: {split}")
    if split == "test" and selecao_top_k is None:
        raise AvaliacaoRecuperacaoError("test exige --selecao-top-k produzida em dev")
    itens = ds.carregar(
        cfg.dataset.dataset_path,
        split=split,
        estrito=True,
        manifesto_path=cfg.corpus.manifesto_path,
        cfg=cfg.dataset,
        corpus=cfg.corpus,
    )
    dev = ds.carregar(
        cfg.dataset.dataset_path,
        split="dev",
        estrito=True,
        manifesto_path=cfg.corpus.manifesto_path,
        cfg=cfg.dataset,
        corpus=cfg.corpus,
    )
    corpus_sha = sha256_arquivo(cfg.corpus.manifesto_path)
    top_k_principal = None
    selecao_sha = None
    if selecao_top_k is not None:
        top_k_principal, selecao_sha = carregar_selecao(
            selecao_top_k,
            cfg=cfg,
            corpus_sha256=corpus_sha,
            dev_dataset_sha256=sha256_itens(dev),
        )

    registros: list[dict] = []
    indices = {}
    profundidade = max(cfg.recuperacao.ks_metricas)
    for config_id in cfg.chunking.ids:
        recuperador = fabrica_recuperador(config_id)
        if recuperador.config_id != config_id:
            raise AvaliacaoRecuperacaoError(
                f"fábrica devolveu {recuperador.config_id} para {config_id}"
            )
        indices[config_id] = _proveniencia_indice(recuperador)
        for item in itens:
            inicio = time.perf_counter()
            trechos = recuperador.buscar(item.pergunta, profundidade)
            latencia = time.perf_counter() - inicio
            avaliacao = avaliar_ranking(
                trechos,
                item.evidencias_validas,
                respondivel=item.respondivel,
                ks=cfg.recuperacao.ks_metricas,
            )
            registros.append(
                {
                    "schema_version": VERSAO_AVALIACAO,
                    "split": split,
                    "config_id": config_id,
                    "pergunta_id": item.id,
                    "parte": item.parte,
                    "categoria": item.categoria,
                    "respondivel": item.respondivel,
                    "evidencias_validas": [
                        {"aula": aula, "pagina": pagina}
                        for aula, pagina in item.evidencias_validas
                    ],
                    "latencia_recuperacao_s": latencia,
                    **avaliacao,
                }
            )

    manifesto = {
        "schema_version": VERSAO_AVALIACAO,
        "split": split,
        "git_sha": _git_sha(),
        "protocolo_sha256": sha256_protocolo(cfg),
        "corpus_manifesto_sha256": corpus_sha,
        "dataset_sha256": ds.sha256_dataset(cfg.dataset.dataset_path),
        "dataset_split_sha256": sha256_itens(itens),
        "configs": list(cfg.chunking.ids),
        "perguntas": [item.id for item in itens],
        "ks": list(cfg.recuperacao.ks_metricas),
        "profundidade_ranking": profundidade,
        "indices": indices,
        "top_k_principal": top_k_principal,
        "selecao_top_k_sha256": selecao_sha,
    }
    manifesto_sha = sha256_json(manifesto)
    for registro in registros:
        registro["manifesto_avaliacao_sha256"] = manifesto_sha
        registro["indice_manifesto_sha256"] = indices[registro["config_id"]]["manifesto_sha256"]
        registro["indice_cache_key"] = indices[registro["config_id"]]["cache_key"]
        registro["ranking_id"] = sha256_json(
            {
                "manifesto": manifesto_sha,
                "config_id": registro["config_id"],
                "pergunta_id": registro["pergunta_id"],
                "chunks": [
                    [r["posicao"], r["chunk_id"], r["score"]] for r in registro["ranking"]
                ],
            }
        )

    saida = saida or cfg.results_dir / "retrieval" / split
    saida.mkdir(parents=True, exist_ok=True)
    manifesto_path = saida / "run-manifest.json"
    rankings_path = saida / "rankings.jsonl"
    _gravar_json(manifesto_path, manifesto)
    rankings_path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in registros
        ),
        encoding="utf-8",
    )
    resumo = agregar(
        registros,
        cfg.recuperacao.ks_metricas,
        manifesto_sha256=manifesto_sha,
        top_k_principal=top_k_principal,
    )
    _gravar_json(saida / "summary.json", resumo)
    _gravar_csv(saida / "summary.csv", resumo)
    return saida


def selecionar(avaliacao_dir: Path, saida: Path | None = None, cfg: ExperimentoConfig = CONFIG) -> Path:
    manifesto_path = avaliacao_dir / "run-manifest.json"
    rankings_path = avaliacao_dir / "rankings.jsonl"
    try:
        manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
        registros = [
            json.loads(linha)
            for linha in rankings_path.read_text(encoding="utf-8").splitlines()
            if linha.strip()
        ]
    except (OSError, json.JSONDecodeError) as e:
        raise AvaliacaoRecuperacaoError("artefatos de avaliação inválidos") from e
    if any(r.get("manifesto_avaliacao_sha256") != sha256_json(manifesto) for r in registros):
        raise AvaliacaoRecuperacaoError("rankings não correspondem ao manifesto")
    artefato = montar_artefato_selecao(
        registros, manifesto, sha256_arquivo(rankings_path), cfg
    )
    destino = saida or avaliacao_dir / "top-k-selection.json"
    _gravar_json(destino, artefato)
    return destino


def _fabrica_padrao() -> Callable[[str], Recuperador]:
    return importlib.import_module("src.indice").construir_recuperador


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    avaliar = sub.add_parser("avaliar", help="coleta rankings e gera métricas")
    avaliar.add_argument("--split", choices=ds.SPLITS, required=True)
    avaliar.add_argument("--saida", type=Path)
    avaliar.add_argument("--selecao-top-k", type=Path)
    escolher = sub.add_parser("selecionar-k", help="seleciona k usando a avaliação dev")
    escolher.add_argument("--avaliacao-dir", type=Path, required=True)
    escolher.add_argument("--saida", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.comando == "avaliar":
            destino = coletar(
                args.split,
                fabrica_recuperador=_fabrica_padrao(),
                saida=args.saida,
                selecao_top_k=args.selecao_top_k,
            )
        else:
            destino = selecionar(args.avaliacao_dir, args.saida)
    except (AvaliacaoRecuperacaoError, ds.DatasetError, RuntimeError) as e:
        parser.error(str(e))
    print(destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
