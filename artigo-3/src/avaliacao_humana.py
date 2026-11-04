"""Avaliação humana cega das respostas e concordância entre avaliadores (#52).

Fluxo:
1. `gerar_planilha_cega` embaralha as respostas de todas as configurações e
   troca o par (pergunta, configuração) por um identificador opaco. A chave que
   desfaz o embaralhamento fica em arquivo separado, que os avaliadores não abrem.
2. Cada autor preenche a própria cópia seguindo `RUBRICA.md`.
3. `concordancia` mede o kappa de Cohen por critério; `divergencias` lista os
   itens que precisam de consenso antes de entrar no artigo.
4. `classificar_falha` separa falha de recuperação de falha de geração.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Critério -> valores aceitos. Escalas ordinais curtas para facilitar a concordância.
RUBRICA: dict[str, tuple[int, ...]] = {
    "correcao": (0, 1, 2),
    "completude": (0, 1, 2),
    "fidelidade": (0, 1, 2),
    "afirmacoes_nao_sustentadas": (0, 1, 2, 3),  # 3 = três ou mais
}
CAMPOS_AVALIADOR = tuple(RUBRICA) + ("observacao",)


class AvaliacaoError(ValueError):
    pass


def _id_cego(pergunta_id: str, config_id: str, seed: int) -> str:
    bruto = f"{seed}:{pergunta_id}:{config_id}".encode()
    return "R" + hashlib.sha256(bruto).hexdigest()[:8]


def gerar_planilha_cega(
    respostas: list[dict],
    referencias: dict[str, str],
    destino: Path,
    chave: Path,
    seed: int,
) -> None:
    """Grava a planilha sem o ID da configuração e, à parte, a chave de desembaralhamento.

    ``respostas`` são os dicionários de `geracao.RespostaRAG`; ``referencias``
    mapeia ``pergunta_id`` para a resposta de referência do dataset.
    """
    linhas = []
    for r in respostas:
        contexto = "\n\n".join(
            f"[{i}] (aula {t['aula']}, p. {t['pagina_inicio']}) {t['texto']}"
            for i, t in enumerate(r["trechos"], start=1)
        )
        linhas.append(
            {
                "item": _id_cego(r["pergunta_id"], r["config_id"], seed),
                "pergunta": r["pergunta"],
                "resposta_referencia": referencias[r["pergunta_id"]],
                "contexto": contexto,
                "resposta": r["resposta"],
                "_pergunta_id": r["pergunta_id"],
                "_config_id": r["config_id"],
            }
        )
    if len({l["item"] for l in linhas}) != len(linhas):
        raise AvaliacaoError("Par (pergunta, configuração) repetido nas respostas.")

    # Ordena antes de embaralhar para que a ordem dependa só da seed.
    linhas.sort(key=lambda l: l["item"])
    random.Random(seed).shuffle(linhas)

    destino.parent.mkdir(parents=True, exist_ok=True)
    visiveis = ["item", "pergunta", "resposta_referencia", "contexto", "resposta"]
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=visiveis + list(CAMPOS_AVALIADOR))
        w.writeheader()
        for l in linhas:
            w.writerow({k: l[k] for k in visiveis})

    chave.parent.mkdir(parents=True, exist_ok=True)
    with chave.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "pergunta_id", "config_id"])
        w.writeheader()
        for l in sorted(linhas, key=lambda l: l["item"]):
            w.writerow({"item": l["item"], "pergunta_id": l["_pergunta_id"], "config_id": l["_config_id"]})


def ler_avaliacao(caminho: Path) -> dict[str, dict[str, int]]:
    """Lê a planilha preenchida e valida cada nota contra a rubrica."""
    notas: dict[str, dict[str, int]] = {}
    with caminho.open(encoding="utf-8") as f:
        for n_linha, linha in enumerate(csv.DictReader(f), start=2):
            item = linha["item"]
            registro = {}
            for criterio, aceitos in RUBRICA.items():
                bruto = (linha.get(criterio) or "").strip()
                if not bruto:
                    raise AvaliacaoError(f"{caminho.name}:{n_linha}: '{criterio}' em branco ({item}).")
                valor = int(bruto)
                if valor not in aceitos:
                    raise AvaliacaoError(
                        f"{caminho.name}:{n_linha}: '{criterio}'={valor} fora de {aceitos}."
                    )
                registro[criterio] = valor
            notas[item] = registro
    return notas


def kappa_cohen(a: list[int], b: list[int], categorias: tuple[int, ...], ponderado: bool = True) -> float:
    """Kappa de Cohen; com ``ponderado`` usa pesos quadráticos (escala ordinal)."""
    if len(a) != len(b) or not a:
        raise AvaliacaoError("Sequências de notas vazias ou de tamanhos diferentes.")
    idx = {c: i for i, c in enumerate(categorias)}
    k = len(categorias)
    n = len(a)

    def peso(i: int, j: int) -> float:
        if ponderado and k > 1:
            return ((i - j) / (k - 1)) ** 2
        return 0.0 if i == j else 1.0

    obs = Counter((idx[x], idx[y]) for x, y in zip(a, b))
    ma = Counter(idx[x] for x in a)
    mb = Counter(idx[y] for y in b)
    d_obs = sum(peso(i, j) * c for (i, j), c in obs.items()) / n
    d_esp = sum(peso(i, j) * ma[i] * mb[j] for i in range(k) for j in range(k)) / (n * n)
    if d_esp == 0:
        return 1.0  # os dois avaliadores usaram uma única categoria e concordaram
    return 1.0 - d_obs / d_esp


def concordancia(av1: dict[str, dict[str, int]], av2: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    if set(av1) != set(av2):
        faltando = sorted(set(av1) ^ set(av2))
        raise AvaliacaoError(f"Planilhas com itens diferentes: {faltando[:5]}…")
    itens = sorted(av1)
    resultado = {}
    for criterio, cats in RUBRICA.items():
        a = [av1[i][criterio] for i in itens]
        b = [av2[i][criterio] for i in itens]
        resultado[criterio] = {
            "kappa_quadratico": kappa_cohen(a, b, cats, ponderado=True),
            "concordancia_exata": sum(x == y for x, y in zip(a, b)) / len(a),
            "n": len(a),
        }
    return resultado


def divergencias(av1: dict[str, dict[str, int]], av2: dict[str, dict[str, int]]) -> list[tuple[str, str, int, int]]:
    """Itens e critérios em que os avaliadores discordam, para a reunião de consenso."""
    return [
        (item, c, av1[item][c], av2[item][c])
        for item in sorted(av1)
        for c in RUBRICA
        if av1[item][c] != av2[item][c]
    ]


@dataclass(frozen=True)
class Desfecho:
    categoria: str
    descricao: str


DESFECHOS = {
    "acerto": "evidência recuperada e resposta correta",
    "falha_recuperacao": "nenhuma evidência válida no top-k; o gerador não tinha como acertar",
    "falha_geracao": "evidência recuperada, mas resposta incorreta ou abstenção indevida",
    "acerto_sem_evidencia": "resposta correta sem evidência válida no top-k (conhecimento prévio ou evidência não anotada)",
    "abstencao_correta": "pergunta não respondível e o sistema se absteve",
    "resposta_inventada": "pergunta não respondível e o sistema respondeu",
}


def classificar_falha(*, respondivel: bool, recuperou_evidencia: bool, abstencao: bool, correcao: int) -> str:
    """Atribui um desfecho de `DESFECHOS`; ``correcao`` usa a escala da rubrica (2 = correta)."""
    if not respondivel:
        return "abstencao_correta" if abstencao else "resposta_inventada"
    correta = correcao == 2 and not abstencao
    if recuperou_evidencia:
        return "acerto" if correta else "falha_geracao"
    return "acerto_sem_evidencia" if correta else "falha_recuperacao"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Concordância entre as duas avaliações cegas.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("concordancia")
    c.add_argument("avaliador1", type=Path)
    c.add_argument("avaliador2", type=Path)
    a = p.parse_args(argv)
    av1, av2 = ler_avaliacao(a.avaliador1), ler_avaliacao(a.avaliador2)
    print(json.dumps(concordancia(av1, av2), indent=2, ensure_ascii=False))
    div = divergencias(av1, av2)
    print(f"{len(div)} divergências para consenso:")
    for item, criterio, n1, n2 in div:
        print(f"  {item} {criterio}: {n1} vs {n2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
