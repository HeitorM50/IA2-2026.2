"""Orquestrador da grade de chunking (#53).

Para cada configuração e cada pergunta do split, recupera, gera e grava um
JSON bruto em ``src/results/raw/<split>/<config>/<pergunta>.json``. O manifesto
da execução amarra os resultados ao commit, ao protocolo, ao corpus e ao
dataset; uma retomada com qualquer um deles diferente é recusada, para que
resultados de versões distintas nunca se misturem.

Uso::

    python -m src.run --split dev --limite 3      # depuração
    python -m src.run --split test                # execução canônica
    python -m src.run --split test --resume       # retoma após interrupção
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import dataset as ds
from src.config import ARTIGO, CONFIG, ExperimentoConfig, como_dict, set_seed, validar_congelamento
from src.contratos import Gerador, Recuperador
from src.geracao import VERSAO_PROMPT, responder

log = logging.getLogger("artigo3.run")

# Pacotes que identificam o ambiente no environment.json.
PACOTES_RELEVANTES = ("pymupdf", "numpy", "faiss-cpu", "sentence-transformers", "torch", "transformers")


class ExecucaoError(RuntimeError):
    pass


# --- verificações prévias --------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def head_destacado(repo: Path) -> bool:
    r = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo, capture_output=True)
    return r.returncode != 0


def estado_git(repo: Path, ignorar: Path) -> tuple[str, list[str]]:
    """SHA do HEAD e arquivos sujos, desconsiderando a pasta de resultados."""
    sha = _git(repo, "rev-parse", "HEAD")
    raiz = Path(_git(repo, "rev-parse", "--show-toplevel"))
    try:
        prefixo = ignorar.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        prefixo = None
    sujos = []
    for linha in _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines():
        caminho = linha[3:].strip().strip('"')
        if prefixo and (caminho + "/").startswith(prefixo.rstrip("/") + "/"):
            continue
        sujos.append(caminho)
    return sha, sujos


def verificar_corpus(cfg: ExperimentoConfig) -> str:
    """Confere o texto extraído contra o manifesto versionado; devolve o hash do manifesto."""
    m_path = cfg.corpus.manifesto_path
    if not m_path.is_file():
        raise ExecucaoError(f"Manifesto do corpus ausente: {m_path}. Rode `python -m src.corpus`.")
    manifesto = json.loads(m_path.read_text(encoding="utf-8"))
    for entrada in manifesto["arquivos"]:
        arq = cfg.corpus.extraido_dir / entrada["extraido"]
        if not arq.is_file():
            raise ExecucaoError(f"Texto extraído ausente: {arq}")
        digest = hashlib.sha256(arq.read_bytes()).hexdigest()
        if digest != entrada["sha256_extraido"]:
            raise ExecucaoError(
                f"{arq.name} não corresponde ao manifesto do corpus "
                f"({digest[:12]} ≠ {entrada['sha256_extraido'][:12]})."
            )
    return hashlib.sha256(m_path.read_bytes()).hexdigest()


def _hash_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --- registro de ambiente --------------------------------------------------


def gravar_ambiente(saida: Path) -> None:
    pacotes = {}
    for p in PACOTES_RELEVANTES:
        try:
            pacotes[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pacotes[p] = None
    ambiente = {
        "python": sys.version,
        "plataforma": platform.platform(),
        "pacotes": pacotes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (saida / "environment.json").write_text(json.dumps(ambiente, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    (saida / "pip-freeze.txt").write_text(freeze, encoding="utf-8")


def _gravar_atomico(caminho: Path, dados: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, caminho)  # atômico: interrupção nunca deixa JSON pela metade


def _resultado_valido(caminho: Path, pergunta_id: str, config_id: str) -> bool:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return dados.get("pergunta_id") == pergunta_id and dados.get("config_id") == config_id


# --- execução --------------------------------------------------------------


@dataclass(frozen=True)
class Resumo:
    executados: int
    pulados: int
    saida: Path


def executar(
    split: str,
    *,
    fabrica_recuperador: Callable[[str], Recuperador],
    gerador: Gerador,
    cfg: ExperimentoConfig = CONFIG,
    saida: Path | None = None,
    resume: bool = False,
    limite: int | None = None,
    configs: tuple[str, ...] | None = None,
    repo: Path = ARTIGO,
) -> Resumo:
    if split not in ds.SPLITS:
        raise ExecucaoError(f"split {split!r} inválido")
    canonica = split == "test"
    if canonica and limite is not None:
        raise ExecucaoError("--limite só é permitido no split dev.")
    configs = configs or cfg.chunking.ids
    desconhecidas = set(configs) - set(cfg.chunking.ids)
    if desconhecidas:
        raise ExecucaoError(f"configurações fora da grade: {sorted(desconhecidas)}")

    saida = saida or cfg.results_dir / "raw" / split
    sha, sujos = estado_git(repo, cfg.results_dir)
    if canonica:
        validar_congelamento(cfg)
        if sujos:
            raise ExecucaoError(f"Git sujo; a execução canônica exige commit limpo: {sujos[:5]}")
        if not head_destacado(repo):
            raise ExecucaoError("A execução canônica roda de um SHA destacado: `git checkout <sha>`.")
        congelado = (cfg.geracao.modelo_gerador, cfg.geracao.revisao_gerador)
        if (gerador.nome, gerador.revisao) != congelado:
            raise ExecucaoError(
                f"Gerador {gerador.nome}@{gerador.revisao} difere do congelado em config.py: "
                f"{congelado[0]}@{congelado[1]}."
            )

    corpus_sha = verificar_corpus(cfg)
    dataset_path = cfg.dataset.dataset_path
    itens = ds.carregar(
        dataset_path,
        split=split,
        estrito=canonica,
        manifesto_path=cfg.corpus.manifesto_path,
        cfg=cfg.dataset,
        corpus=cfg.corpus,
    )
    if limite is not None:
        itens = itens[:limite]

    manifesto = {
        "split": split,
        "git_sha": sha,
        "git_sujo": bool(sujos),
        "protocolo_sha256": _hash_json(como_dict(cfg, sem_caminhos=True)),
        "corpus_manifesto_sha256": corpus_sha,
        "dataset_sha256": ds.sha256_dataset(dataset_path),
        "configs": list(configs),
        "perguntas": [it.id for it in itens],
        "modelo_gerador": gerador.nome,
        "revisao_gerador": gerador.revisao,
        "versao_prompt": VERSAO_PROMPT,
        "top_k": cfg.recuperacao.top_k,
        "seed": cfg.seed,
    }
    m_path = saida / "run-manifest.json"
    if m_path.exists():
        anterior = json.loads(m_path.read_text(encoding="utf-8"))
        if not resume:
            raise ExecucaoError(f"{saida} já contém uma execução; use --resume ou outra --saida.")
        diferentes = sorted(k for k in manifesto if anterior.get(k) != manifesto[k])
        if diferentes:
            raise ExecucaoError(
                "Manifesto incompatível com a execução anterior em "
                f"{saida}: {', '.join(diferentes)}. Não é seguro retomar."
            )
    elif resume and any(saida.glob("*/*.json")):
        raise ExecucaoError(f"{saida} tem resultados sem manifesto; não é seguro retomar.")

    saida.mkdir(parents=True, exist_ok=True)
    _gravar_atomico(m_path, manifesto)
    gravar_ambiente(saida)
    handler = logging.FileHandler(saida / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    executados = pulados = 0
    try:
        log.info("início split=%s sha=%s sujo=%s configs=%s n=%d", split, sha[:12], bool(sujos), configs, len(itens))
        set_seed(cfg.seed)
        for config_id in configs:
            recuperador = fabrica_recuperador(config_id)
            if recuperador.config_id != config_id:
                raise ExecucaoError(f"fábrica devolveu {recuperador.config_id} para {config_id}")
            for it in itens:
                destino = saida / config_id / f"{it.id}.json"
                if destino.exists() and _resultado_valido(destino, it.id, config_id):
                    pulados += 1
                    continue
                r = responder(it.id, it.pergunta, recuperador, gerador, top_k=cfg.recuperacao.top_k, cfg=cfg.geracao)
                dados = r.como_dict() | {
                    "split": split,
                    "categoria": it.categoria,
                    "respondivel": it.respondivel,
                    "evidencias_validas": [list(e) for e in it.evidencias_validas],
                    "git_sha": sha,
                }
                _gravar_atomico(destino, dados)
                executados += 1
                log.info("ok %s %s", config_id, it.id)
        log.info("fim executados=%d pulados=%d", executados, pulados)
    finally:
        log.removeHandler(handler)
        handler.close()
    return Resumo(executados=executados, pulados=pulados, saida=saida)


def _fabricas_padrao() -> tuple[Callable[[str], Recuperador], Gerador]:
    """Carrega o índice (#50) e o gerador (#52) quando existirem."""
    try:
        indice = importlib.import_module("src.indice")
        backend = importlib.import_module("src.gerador_backend")
    except ModuleNotFoundError as e:
        raise ExecucaoError(
            f"Módulo {e.name} ainda não implementado; o run precisa de "
            "src.indice.construir_recuperador (#50) e src.gerador_backend.construir_gerador (#52)."
        ) from e
    return indice.construir_recuperador, backend.construir_gerador()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=ds.SPLITS, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limite", type=int, help="só no split dev: número de perguntas")
    p.add_argument("--configs", nargs="+", help="subconjunto da grade (padrão: todas)")
    p.add_argument("--saida", type=Path)
    a = p.parse_args(argv)
    try:
        fabrica, gerador = _fabricas_padrao()
        r = executar(
            a.split,
            fabrica_recuperador=fabrica,
            gerador=gerador,
            saida=a.saida,
            resume=a.resume,
            limite=a.limite,
            configs=tuple(a.configs) if a.configs else None,
        )
    except (ExecucaoError, ds.DatasetError, RuntimeError) as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    print(f"{r.executados} executados, {r.pulados} pulados → {r.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
