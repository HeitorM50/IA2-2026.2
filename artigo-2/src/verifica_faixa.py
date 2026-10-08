"""Verificação de faixa física: quanto uma regra de limites detecta de cada falha.

Reproduz os números citados na Metodologia do artigo e registrados em
PLANO-EXPERIMENTAL.md. Rodar de dentro de artigo-2/:

    python -m src.verifica_faixa

O argumento central do artigo é que três das cinco classes de falha atravessam
intactas uma verificação de faixa. Esse é o número que justifica a existência de
um detector aprendido, então ele não pode ser afirmado de memória: as classes
ganho, sensor travado e lacuna passam em 100 % dos casos de forma determinística,
enquanto deriva e pico variam com a semente e são reportados como média.

O caminho de construção do conjunto de avaliação é o mesmo de ``run.py``: a
escala por canal é o desvio do normalizador ajustado no treino, e o gerador
aleatório é compartilhado entre validação e teste, nessa ordem. Replicar essa
ordem importa — um gerador novo para o teste produz outra atribuição de falhas.
"""

from __future__ import annotations

import numpy as np

from src import faults
from src.config import DATA_CONFIG, FAULT_CONFIG, set_seed
from src.data import load_dataset

CLASSES = ("gain", "stuck", "gap", "drift", "spike")


def taxa_de_deteccao(
    conjunto: faults.EvaluationSet, classe: str
) -> float:
    """Fração das janelas de uma classe que a verificação de faixa sinaliza."""

    indices = [i for i, nome in enumerate(conjunto.fault) if nome == classe]
    if not indices:
        raise ValueError(f"A classe {classe!r} não recebeu janelas.")
    sinalizadas = sum(
        faults.violates_range(conjunto.X[i], DATA_CONFIG, FAULT_CONFIG)
        for i in indices
    )
    return sinalizadas / len(indices)


def mede() -> dict[str, object]:
    """Mede a verificação de faixa sobre o teste, nas três sementes canônicas."""

    dataset = load_dataset(normalize=False)
    teste = dataset.test.X

    # Antes da injeção o teste é inteiramente operação normal: a regra de faixa
    # não pode sinalizar nada, ou produziria falso positivo em dado íntegro.
    falsos_positivos = sum(
        faults.violates_range(teste[i], DATA_CONFIG, FAULT_CONFIG)
        for i in range(len(teste))
    )

    por_classe: dict[str, list[float]] = {classe: [] for classe in CLASSES}
    for semente in DATA_CONFIG.canonical_seeds:
        rng = set_seed(semente)
        # Mesma ordem de run.py: a validação consome o gerador antes do teste.
        faults.build_evaluation_set(dataset.val.X, dataset.std, rng)
        avaliacao = faults.build_evaluation_set(teste, dataset.std, rng)
        for classe in CLASSES:
            por_classe[classe].append(taxa_de_deteccao(avaliacao, classe))

    return {
        "janelas_teste": len(teste),
        "falsos_positivos": falsos_positivos,
        "por_classe": por_classe,
    }


def main() -> None:
    medida = mede()
    print(f"janelas de teste: {medida['janelas_teste']}")
    print(
        "normais sinalizadas antes da injeção: "
        f"{medida['falsos_positivos']}/{medida['janelas_teste']}"
    )
    print("\nclasse   detecção pela faixa, por semente (%)   média   passa")
    por_classe: dict[str, list[float]] = medida["por_classe"]  # type: ignore[assignment]
    for classe, taxas in por_classe.items():
        linha = "  ".join(f"{t * 100:5.1f}" for t in taxas)
        media = float(np.mean(taxas)) * 100
        print(f"{classe:8s} {linha:38s} {media:5.1f}  {100 - media:5.1f}")


if __name__ == "__main__":
    main()
