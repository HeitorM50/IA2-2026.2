"""Valida um pacote recuperado da execução canônica do Artigo 3.

O comando é deliberadamente somente leitura. Ele verifica os manifestos, os
hashes encadeados, a grade de 45 perguntas por sete configurações e o top-k
antes que qualquer relatório novo seja produzido.

Uso::

    python -m src.validar_artefatos /content/drive/MyDrive/IA2/art3-resultados
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.config import CONFIG

SHA_CANONICO = "4f347d62c167bf152ff32c6e5c54f570dc705c38"


class ArtefatosError(ValueError):
    pass


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def _json(caminho: Path) -> dict:
    if not caminho.is_file():
        raise ArtefatosError(f"arquivo obrigatório ausente: {caminho}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def _localizar(raiz: Path, relativo: str) -> Path:
    """Aceita tanto o diretório `results` quanto a raiz de uma cópia do projeto."""
    direto = raiz / relativo
    aninhado = raiz / "src" / "results" / relativo
    return direto if direto.exists() else aninhado


def validar(raiz: Path) -> dict:
    ger = _json(_localizar(raiz, "raw/test/run-manifest.json"))
    ret_path = _localizar(raiz, "retrieval/test/run-manifest.json")
    ret = _json(ret_path)
    sel_path = _localizar(raiz, "retrieval/dev/top-k-selection.json")
    sel = _json(sel_path)

    configs = list(CONFIG.chunking.ids)
    perguntas = ger.get("perguntas", [])
    erros: list[str] = []
    for nome, m in (("geração", ger), ("recuperação", ret)):
        if m.get("git_sha") != SHA_CANONICO:
            erros.append(f"{nome}: git_sha={m.get('git_sha')!r}, esperado {SHA_CANONICO}")
        if m.get("configs") != configs:
            erros.append(f"{nome}: configurações diferentes das sete congeladas")
        if m.get("top_k", m.get("top_k_principal")) != 5:
            erros.append(f"{nome}: top-k diferente de 5")
        if len(m.get("perguntas", [])) != 45:
            erros.append(f"{nome}: esperado 45 perguntas")
    if sel.get("top_k_selecionado") != 5:
        erros.append("seleção de desenvolvimento não fixou top-k=5")
    if sel.get("configs") != configs:
        erros.append("seleção de desenvolvimento não contém as sete configurações")
    if sha256(sel_path) != ger.get("selecao_top_k_sha256"):
        erros.append("hash de top-k-selection.json diverge do manifesto de geração")
    if sha256(sel_path) != ret.get("selecao_top_k_sha256"):
        erros.append("hash de top-k-selection.json diverge do manifesto de recuperação")
    for campo in ("corpus_manifesto_sha256", "dataset_sha256", "protocolo_sha256"):
        if ger.get(campo) != ret.get(campo):
            erros.append(f"hash {campo} diverge entre geração e recuperação")

    rankings = _localizar(raiz, "retrieval/test/rankings.jsonl")
    if not rankings.is_file():
        erros.append("rankings.jsonl ausente")
    else:
        linhas = [json.loads(l) for l in rankings.read_text(encoding="utf-8").splitlines() if l.strip()]
        pares = {(l.get("pergunta_id"), l.get("config_id")) for l in linhas}
        esperado = {(q, c) for q in perguntas for c in configs}
        if pares != esperado:
            erros.append(f"grade de rankings incompleta: {len(pares)}/{len(esperado)} pares")

    brutos = _localizar(raiz, "raw/test")
    jsons = [p for p in brutos.glob("*/*.json") if p.name != "environment.json"] if brutos.is_dir() else []
    pares_brutos = set()
    for p in jsons:
        b = _json(p)
        if "pergunta_id" in b and "config_id" in b:
            pares_brutos.add((b["pergunta_id"], b["config_id"]))
    esperado = {(q, c) for q in perguntas for c in configs}
    if pares_brutos != esperado:
        erros.append(f"grade de respostas incompleta: {len(pares_brutos)}/{len(esperado)} pares")
    if erros:
        raise ArtefatosError("Pacote inválido:\n- " + "\n- ".join(erros))
    return {
        "git_sha": ger["git_sha"],
        "perguntas": len(perguntas),
        "configuracoes": len(configs),
        "pares": len(esperado),
        "top_k": 5,
        "rankings_sha256": sha256(rankings),
        "selecao_top_k_sha256": sha256(sel_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz", type=Path)
    a = p.parse_args(argv)
    try:
        resumo = validar(a.raiz)
    except (ArtefatosError, OSError, json.JSONDecodeError) as e:
        p.error(str(e))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
