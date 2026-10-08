"""Testes da injeção de falhas do Artigo 2.

O teste que mais importa aqui é `test_ganho_nao_dispara_checagem_de_faixa`: ele
verifica mecanicamente o argumento central do artigo, em vez de deixá-lo como
afirmação no texto.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import faults
from src.config import DATA_CONFIG, FAULT_CONFIG, FAULT_NAMES


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------


def janela_realista(rng: np.random.Generator) -> np.ndarray:
    """Janela sintética em unidades físicas, próxima do observado no dataset."""

    centro = np.array([1520.0, 65.0, 23.0, 127.0, 21.0, 21.0], dtype=np.float32)
    desvio = np.array([537.0, 46.0, 17.0, 32.0, 20.0, 13.0], dtype=np.float32)
    ruido = rng.normal(0.0, 0.3, size=(DATA_CONFIG.window_s, DATA_CONFIG.n_channels))
    return (centro + desvio * ruido).astype(np.float32)


ESCALA = np.array([537.0, 46.0, 17.0, 32.0, 20.0, 13.0], dtype=np.float32)


def injeta(nome: str, janela: np.ndarray, rng: np.random.Generator):
    canal = DATA_CONFIG.channels.index(FAULT_CONFIG.channel_for(nome))
    return faults.INJECTORS[nome](
        janela, canal, rng, FAULT_CONFIG, float(ESCALA[canal])
    )


# --------------------------------------------------------------------------
# Invariantes comuns às cinco classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_injecao_altera_a_janela(nome: str):
    """Toda falha precisa mudar alguma coisa, senão não há o que detectar."""

    rng = np.random.default_rng(0)
    original = janela_realista(rng)

    resultado = injeta(nome, original, rng)

    assert not np.array_equal(resultado.X, original)
    assert resultado.mask.any()


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_injecao_nao_toca_outros_canais(nome: str):
    """A falha é de um sensor: os demais canais ficam intactos."""

    rng = np.random.default_rng(1)
    original = janela_realista(rng)

    resultado = injeta(nome, original, rng)

    outros = [i for i in range(DATA_CONFIG.n_channels) if i != resultado.channel]
    assert np.array_equal(resultado.X[:, outros], original[:, outros])


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_mascara_marca_exatamente_o_que_mudou(nome: str):
    """Critério de aceite: a máscara corresponde às amostras alteradas."""

    rng = np.random.default_rng(2)
    original = janela_realista(rng)

    resultado = injeta(nome, original, rng)

    mudou = resultado.X[:, resultado.channel] != original[:, resultado.channel]
    assert np.array_equal(resultado.mask, mudou)


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_nada_muda_antes_do_inicio_da_falha(nome: str):
    """O trecho anterior ao início precisa ficar idêntico: a latência depende disso."""

    rng = np.random.default_rng(3)
    original = janela_realista(rng)

    resultado = injeta(nome, original, rng)

    assert np.array_equal(resultado.X[: resultado.start], original[: resultado.start])
    assert not resultado.mask[: resultado.start].any()


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_injecao_e_deterministica_pela_semente(nome: str):
    """Mesma semente, mesma falha. Sem isso a execução não é reproduzível."""

    base = janela_realista(np.random.default_rng(4))

    primeira = injeta(nome, base, np.random.default_rng(99))
    segunda = injeta(nome, base, np.random.default_rng(99))

    assert np.array_equal(primeira.X, segunda.X)
    assert primeira.start == segunda.start


@pytest.mark.parametrize("nome", FAULT_NAMES)
def test_injecao_preserva_formato_e_tipo(nome: str):
    rng = np.random.default_rng(5)

    resultado = injeta(nome, janela_realista(rng), rng)

    assert resultado.X.shape == (DATA_CONFIG.window_s, DATA_CONFIG.n_channels)
    assert resultado.X.dtype == np.float32
    assert resultado.mask.dtype == np.bool_
    assert resultado.fault == nome


# --------------------------------------------------------------------------
# Comportamento específico de cada classe
# --------------------------------------------------------------------------


def test_ganho_multiplica_pelo_fator():
    rng = np.random.default_rng(10)
    original = janela_realista(rng)

    resultado = injeta("gain", original, rng)
    canal, inicio = resultado.channel, resultado.start

    esperado = original[inicio:, canal] * FAULT_CONFIG.gain_factor
    assert np.allclose(resultado.X[inicio:, canal], esperado, rtol=1e-5)


def test_travado_congela_o_valor():
    rng = np.random.default_rng(11)
    original = janela_realista(rng)

    resultado = injeta("stuck", original, rng)
    trecho = resultado.X[resultado.start :, resultado.channel]

    assert np.allclose(trecho, trecho[0])


@pytest.mark.parametrize("semente", [12, 15, 16, 17, 18])
def test_deriva_cresce_monotonicamente_em_magnitude(semente: int):
    """A deriva é uma rampa cujo sentido é sorteado; o que cresce é o módulo."""

    rng = np.random.default_rng(semente)
    original = janela_realista(rng)

    resultado = injeta("drift", original, rng)
    canal, inicio = resultado.channel, resultado.start
    viés = resultado.X[inicio:, canal] - original[inicio:, canal]

    assert np.all(np.diff(np.abs(viés)) >= -1e-4), "a deriva não pode recuar"
    assert abs(viés[0]) < abs(viés[-1])


def test_deriva_atinge_o_valor_final_configurado():
    """O viés final é `drift_final_std` desvios do canal, para cima ou para baixo."""

    rng = np.random.default_rng(19)
    original = janela_realista(rng)

    resultado = injeta("drift", original, rng)
    canal, inicio = resultado.channel, resultado.start
    viés_final = resultado.X[-1, canal] - original[-1, canal]
    esperado = FAULT_CONFIG.drift_final_std * float(ESCALA[canal])

    assert abs(viés_final) == pytest.approx(esperado, rel=1e-3)


def test_pico_afeta_poucas_amostras():
    rng = np.random.default_rng(13)
    original = janela_realista(rng)

    resultado = injeta("spike", original, rng)

    assert resultado.mask.sum() <= FAULT_CONFIG.spike_count


def test_lacuna_repete_valores_por_duracao_limitada():
    """A lacuna curta vira valor repetido pela reamostragem; a longa quebraria o segmento."""

    rng = np.random.default_rng(14)
    original = janela_realista(rng)

    resultado = injeta("gap", original, rng)
    duracao = resultado.stop - resultado.start

    assert FAULT_CONFIG.gap_min_s <= duracao <= FAULT_CONFIG.gap_max_s
    assert duracao < DATA_CONFIG.gap_s * DATA_CONFIG.window_s


# --------------------------------------------------------------------------
# O argumento central do artigo
# --------------------------------------------------------------------------


def test_ganho_nao_dispara_checagem_de_faixa():
    """A falha de ganho permanece dentro da faixa válida na quase totalidade dos casos.

    É o que sustenta o artigo inteiro: nenhuma validação de faixa a detecta, e
    por isso um detector aprendido tem vantagem real sobre uma regra.
    """

    rng = np.random.default_rng(20)
    silenciosas = 0
    total = 500

    for _ in range(total):
        original = janela_realista(rng)
        assert not faults.violates_range(original, DATA_CONFIG, FAULT_CONFIG)

        corrompida = injeta("gain", original, rng)
        if not faults.violates_range(corrompida.X, DATA_CONFIG, FAULT_CONFIG):
            silenciosas += 1

    fracao = silenciosas / total
    assert fracao > 0.95, (
        f"só {fracao:.1%} das janelas com ganho passaram pela checagem de faixa; "
        "o argumento central do artigo depende de essa fração ser alta"
    )


def test_travado_tambem_e_silencioso():
    """O sensor travado congela num valor válido: nenhuma regra de faixa o pega."""

    rng = np.random.default_rng(21)

    for _ in range(200):
        corrompida = injeta("stuck", janela_realista(rng), rng)
        assert not faults.violates_range(corrompida.X, DATA_CONFIG, FAULT_CONFIG)


def test_checagem_de_faixa_pega_valor_absurdo():
    """A checagem precisa funcionar, senão o teste acima não significa nada."""

    rng = np.random.default_rng(22)
    janela = janela_realista(rng)
    janela[10, DATA_CONFIG.channels.index("rpm")] = 50_000.0

    assert faults.violates_range(janela, DATA_CONFIG, FAULT_CONFIG)


# --------------------------------------------------------------------------
# Conjunto de avaliação
# --------------------------------------------------------------------------


def test_conjunto_de_avaliacao_respeita_a_fracao_de_anomalias():
    rng = np.random.default_rng(30)
    janelas = np.stack([janela_realista(rng) for _ in range(400)])

    conjunto = faults.build_evaluation_set(janelas, ESCALA, rng, FAULT_CONFIG)

    esperado = round(400 * FAULT_CONFIG.anomaly_fraction)
    assert conjunto.y.sum() == esperado
    assert len(conjunto.X) == len(janelas)


def test_conjunto_de_avaliacao_cobre_as_cinco_classes():
    rng = np.random.default_rng(31)
    janelas = np.stack([janela_realista(rng) for _ in range(400)])

    conjunto = faults.build_evaluation_set(janelas, ESCALA, rng, FAULT_CONFIG)

    presentes = set(conjunto.fault[conjunto.y == 1].tolist())
    assert presentes == set(FAULT_NAMES)


def test_janelas_normais_ficam_intactas_no_conjunto():
    """O que não foi corrompido precisa sair idêntico à entrada."""

    rng = np.random.default_rng(32)
    janelas = np.stack([janela_realista(rng) for _ in range(300)])

    conjunto = faults.build_evaluation_set(janelas, ESCALA, rng, FAULT_CONFIG)
    normais = conjunto.y == 0

    assert np.array_equal(conjunto.X[normais], janelas[normais])
    assert not conjunto.mask[normais].any()


def test_conjunto_de_avaliacao_e_deterministico():
    rng_base = np.random.default_rng(33)
    janelas = np.stack([janela_realista(rng_base) for _ in range(200)])

    primeira = faults.build_evaluation_set(
        janelas, ESCALA, np.random.default_rng(7), FAULT_CONFIG
    )
    segunda = faults.build_evaluation_set(
        janelas, ESCALA, np.random.default_rng(7), FAULT_CONFIG
    )

    assert np.array_equal(primeira.X, segunda.X)
    assert np.array_equal(primeira.y, segunda.y)
    assert list(primeira.fault) == list(segunda.fault)


def test_nenhuma_janela_anomala_fica_sem_efeito():
    """Uma falha que não altera nenhuma amostra é indistinguível de normal.

    Congelar um canal já constante, ou dobrar um valor que vale zero, devolve a
    janela intacta. Rotular esse caso como anômalo criaria um positivo idêntico
    a um negativo — impossível de acertar por qualquer detector e indefensável
    na Metodologia. A injeção precisa descartar e resortear.
    """

    rng = np.random.default_rng(40)
    # Metade das janelas tem canais constantes, onde travar e lacuna não surtem
    # efeito; é o caso que o dataset real apresenta quando o veículo está parado.
    variaveis = np.stack([janela_realista(rng) for _ in range(200)])
    constantes = np.zeros((200, DATA_CONFIG.window_s, DATA_CONFIG.n_channels), np.float32)
    janelas = np.concatenate([variaveis, constantes])

    conjunto = faults.build_evaluation_set(janelas, ESCALA, rng, FAULT_CONFIG)

    anomalas = conjunto.y == 1
    assert anomalas.sum() > 0
    assert conjunto.mask[anomalas].any(axis=1).all(), (
        "há janela marcada como anômala sem nenhuma amostra alterada"
    )


def test_injecao_falha_alto_quando_nao_ha_janela_utilizavel():
    """Se nenhuma janela admitir falha observável, a execução precisa parar."""

    constantes = np.zeros((100, DATA_CONFIG.window_s, DATA_CONFIG.n_channels), np.float32)
    escala_nula = np.zeros(DATA_CONFIG.n_channels, dtype=np.float32)

    with pytest.raises(RuntimeError, match="acabaram"):
        faults.build_evaluation_set(
            constantes, escala_nula, np.random.default_rng(41), FAULT_CONFIG
        )
