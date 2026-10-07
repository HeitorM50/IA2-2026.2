"""Testes do Modelo A — limiar 3σ por canal.

Esta é a linha de base ingênua: o `if` que um firmware faria. Sem ela, nenhum
número do autoencoder significa coisa alguma.

Os dois testes que importam são `test_dispara_em_pico_obvio` e
`test_nao_dispara_em_ganho_dentro_da_faixa`: juntos eles delimitam exatamente o
que uma regra estatística enxerga e o que ela deixa passar.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DATA_CONFIG, DETECTION_CONFIG
from src.models.baseline import ThresholdDetector


JANELA, CANAIS = DATA_CONFIG.window_s, DATA_CONFIG.n_channels

# Operação normal sintética: média e desvio conhecidos, próximos do dataset.
MEDIA = np.array([1520.0, 65.0, 23.0, 127.0, 21.0, 21.0], dtype=np.float32)
DESVIO = np.array([537.0, 46.0, 17.0, 32.0, 20.0, 13.0], dtype=np.float32)


def normais(n: int, rng: np.random.Generator) -> np.ndarray:
    """Janelas de operação normal, com a média e o desvio declarados acima."""

    ruido = rng.normal(0.0, 1.0, size=(n, JANELA, CANAIS))
    return (MEDIA + DESVIO * ruido).astype(np.float32)


def ajustado(rng: np.random.Generator) -> ThresholdDetector:
    return ThresholdDetector().fit(normais(600, rng))


# --------------------------------------------------------------------------
# Interface comum aos três detectores
# --------------------------------------------------------------------------


def test_fit_devolve_o_proprio_detector():
    rng = np.random.default_rng(0)
    detector = ThresholdDetector()

    assert detector.fit(normais(50, rng)) is detector


def test_escore_por_amostra_tem_uma_entrada_por_instante():
    """A latência de detecção depende disso: escore por amostra, não só por janela."""

    rng = np.random.default_rng(1)
    detector = ajustado(rng)
    X = normais(20, rng)

    escores = detector.score_samples(X)

    assert escores.shape == (20, JANELA)
    assert np.all(np.isfinite(escores))


def test_escore_por_janela_tem_uma_entrada_por_janela():
    rng = np.random.default_rng(2)
    detector = ajustado(rng)
    X = normais(20, rng)

    escores = detector.score_windows(X)

    assert escores.shape == (20,)


def test_escore_de_janela_agrega_o_escore_por_amostra():
    """A agregação é a mesma para os três modelos, vinda do DetectionConfig."""

    rng = np.random.default_rng(3)
    detector = ajustado(rng)
    X = normais(20, rng)

    por_amostra = detector.score_samples(X)
    por_janela = detector.score_windows(X)

    assert DETECTION_CONFIG.window_aggregation == "max"
    assert np.allclose(por_janela, por_amostra.max(axis=1))


def test_usar_antes_de_ajustar_falha_alto():
    rng = np.random.default_rng(4)

    with pytest.raises(RuntimeError, match="ajustado"):
        ThresholdDetector().score_samples(normais(2, rng))


def test_detector_e_deterministico():
    rng = np.random.default_rng(5)
    treino, avaliacao = normais(300, rng), normais(20, rng)

    a = ThresholdDetector().fit(treino).score_windows(avaliacao)
    b = ThresholdDetector().fit(treino).score_windows(avaliacao)

    assert np.array_equal(a, b)


def test_contagem_de_parametros():
    """Uma média e um desvio por canal — entra na tabela de custo do artigo."""

    rng = np.random.default_rng(6)

    assert ajustado(rng).n_parameters() == 2 * CANAIS


# --------------------------------------------------------------------------
# As estatísticas vêm só do treino
# --------------------------------------------------------------------------


def test_estatisticas_vem_do_conjunto_de_ajuste():
    rng = np.random.default_rng(7)
    detector = ajustado(rng)

    assert np.allclose(detector.mean_, MEDIA, rtol=0.05)
    assert np.allclose(detector.std_, DESVIO, rtol=0.05)


def test_ajuste_ignora_dados_posteriores():
    """Pontuar um conjunto não pode alterar as estatísticas aprendidas."""

    rng = np.random.default_rng(8)
    detector = ajustado(rng)
    antes = detector.mean_.copy(), detector.std_.copy()

    detector.score_windows(normais(50, rng) * 10.0)

    assert np.array_equal(detector.mean_, antes[0])
    assert np.array_equal(detector.std_, antes[1])


def test_canal_constante_nao_gera_divisao_por_zero():
    constante = np.full((40, JANELA, CANAIS), 7.0, dtype=np.float32)

    detector = ThresholdDetector().fit(constante)
    escores = detector.score_samples(constante)

    assert np.all(np.isfinite(escores))


# --------------------------------------------------------------------------
# O que a regra enxerga e o que ela deixa passar
# --------------------------------------------------------------------------


def test_dispara_em_pico_obvio():
    """Outlier é exatamente aquilo para que um limiar foi feito."""

    rng = np.random.default_rng(10)
    detector = ajustado(rng)

    janela = normais(1, rng)
    canal = DATA_CONFIG.channels.index("pressao_adm")
    janela[0, 30, canal] += 6.0 * DESVIO[canal]

    assert detector.predict(janela)[0] == 1
    assert detector.score_windows(janela)[0] > DETECTION_CONFIG.threshold_sigmas


def test_nao_dispara_em_ganho_dentro_da_faixa():
    """Ganho de 2× sobre RPM típico fica em ~2,8 desvios: abaixo do limiar.

    É a lacuna que justifica o artigo. A regra não é inútil — ela pega o ganho
    quando o RPM já estava alto —, mas é cega no regime mais comum de operação.
    """

    rng = np.random.default_rng(11)
    detector = ajustado(rng)
    canal = DATA_CONFIG.channels.index("rpm")

    # Janela no valor médio de operação, sem ruído: o caso mais frequente.
    janela = np.tile(MEDIA, (1, JANELA, 1)).astype(np.float32)
    janela[0, :, canal] *= 2.0

    desvios = (2.0 * MEDIA[canal] - MEDIA[canal]) / DESVIO[canal]
    assert desvios < DETECTION_CONFIG.threshold_sigmas, "premissa do teste"
    assert detector.predict(janela)[0] == 0


def test_nao_dispara_em_sensor_travado():
    """Um valor congelado dentro da faixa normal não se afasta da média."""

    rng = np.random.default_rng(12)
    detector = ajustado(rng)
    canal = DATA_CONFIG.channels.index("velocidade")

    janela = normais(1, rng)
    janela[0, 20:, canal] = janela[0, 19, canal]

    # O valor congelado é um valor normal; a regra não tem como perceber.
    assert detector.predict(janela)[0] == 0


def test_regra_nominal_de_3_sigmas_e_inutilizavel_como_enunciada():
    """A regra de 3σ dispara na MAIORIA das janelas normais. Não é defeito: é aritmética.

    Cada janela traz 60 amostras × 6 canais = 360 leituras. A chance de ao menos
    uma exceder três desvios por puro acaso é

        1 − (1 − 0,0027)^360 ≈ 0,62

    ou seja, cerca de 62 % de falso positivo sobre operação estritamente normal.
    É comparação múltipla, e é um achado que merece uma frase na Discussão: a
    regra ingênua não é apenas cega para falha silenciosa, ela é inutilizável no
    ponto de operação que lhe dá nome.

    Isso **não** prejudica a comparação do artigo. O AUC-PR é baseado em
    ordenação e não depende de limiar, e o ponto de operação usado na avaliação
    vem da calibração na validação normal (#30), não destes 3σ nominais.
    """

    rng = np.random.default_rng(13)
    detector = ajustado(rng)

    disparos = detector.predict(normais(500, rng)).mean()

    assert 0.55 < disparos < 0.72, (
        f"observado {disparos:.1%}; o esperado teórico é ~62 %"
    )


def test_escore_ordena_anomalia_acima_de_normal():
    """O que a avaliação usa é a ORDENAÇÃO do escore, não o limiar nominal.

    Enquanto janela com pico pontuar acima de janela normal, o AUC-PR do modelo
    é bom, independentemente de onde o corte caia.
    """

    rng = np.random.default_rng(14)
    detector = ajustado(rng)
    canal = DATA_CONFIG.channels.index("pressao_adm")

    limpas = normais(200, rng)
    com_pico = normais(200, rng)
    com_pico[:, 30, canal] += 8.0 * DESVIO[canal]

    assert detector.score_windows(com_pico).mean() > detector.score_windows(limpas).mean()
