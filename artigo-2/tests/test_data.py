"""Testes do pipeline de dados do Artigo 2.

Os critérios de aceite da issue #27 são verificados aqui mecanicamente, não por
inspeção: split por sessão sem interseção, normalização calculada só no treino e
determinismo entre execuções.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import DATA_CONFIG


# --------------------------------------------------------------------------
# Auxiliares: quadros sintéticos, para testar sem depender do arquivo real.
# --------------------------------------------------------------------------


def frame_sintetico(segundos: float, taxa_hz: float = 11.0, inicio_s: float = 0.0):
    """Monta um quadro com a forma do CSV original, já com nomes curtos."""

    n = int(segundos * taxa_hz)
    tempo = pd.to_timedelta(inicio_s + np.arange(n) / taxa_hz, unit="s")
    quadro = pd.DataFrame({"Time": tempo})
    for posicao, canal in enumerate(DATA_CONFIG.channels):
        quadro[canal] = np.linspace(posicao, posicao + 1, n)
    return quadro


def quadro_com_lacuna(antes_s: float, lacuna_s: float, depois_s: float):
    """Dois trechos contínuos separados por uma lacuna maior que o limiar."""

    a = frame_sintetico(antes_s)
    fim = a["Time"].iloc[-1].total_seconds()
    b = frame_sintetico(depois_s, inicio_s=fim + lacuna_s)
    return pd.concat([a, b], ignore_index=True)


# --------------------------------------------------------------------------
# Segmentação e janelamento
# --------------------------------------------------------------------------


def test_segmento_reamostra_para_1hz():
    """O quadro entra a 11 Hz e sai a 1 Hz, uma amostra por segundo."""

    segmentos = data.segment_frame(frame_sintetico(300), DATA_CONFIG)

    assert len(segmentos) == 1
    assert segmentos[0].shape[1] == DATA_CONFIG.n_channels
    # 300 s a 1 Hz, com tolerância de uma amostra nas bordas.
    assert 299 <= len(segmentos[0]) <= 301


def test_lacuna_grande_quebra_o_segmento():
    """Uma lacuna acima do limiar separa os trechos em segmentos distintos."""

    quadro = quadro_com_lacuna(antes_s=120, lacuna_s=600, depois_s=120)

    segmentos = data.segment_frame(quadro, DATA_CONFIG)

    assert len(segmentos) == 2, "a lacuna de 600 s deveria encerrar o segmento"


def test_lacuna_pequena_nao_quebra_o_segmento():
    """Uma lacuna abaixo do limiar é apenas uma falha de amostragem."""

    quadro = quadro_com_lacuna(antes_s=120, lacuna_s=1.0, depois_s=120)

    assert len(data.segment_frame(quadro, DATA_CONFIG)) == 1


def test_segmento_curto_e_descartado():
    """Trecho menor que uma janela não gera amostra e é descartado."""

    assert data.segment_frame(frame_sintetico(30), DATA_CONFIG) == []


def test_formato_da_janela():
    """O contrato publicado na issue #27: (n, 60, 6) em float32."""

    segmento = data.segment_frame(frame_sintetico(300), DATA_CONFIG)[0]

    janelas = data.make_windows(segmento, DATA_CONFIG)

    assert janelas.ndim == 3
    assert janelas.shape[1:] == DATA_CONFIG.window_shape
    assert janelas.dtype == np.float32


def test_passo_entre_janelas():
    """Com janela de 60 s e passo de 10 s, 300 s rendem 25 janelas."""

    segmento = data.segment_frame(frame_sintetico(300), DATA_CONFIG)[0]
    esperado = (len(segmento) - DATA_CONFIG.window_s) // DATA_CONFIG.stride_s + 1

    assert len(data.make_windows(segmento, DATA_CONFIG)) == esperado


# --------------------------------------------------------------------------
# Divisão por sessão
# --------------------------------------------------------------------------


def test_divisao_nao_compartilha_sessao():
    """Critério de aceite: nenhuma sessão aparece em dois splits."""

    sessoes = np.array([f"sessao-{i:02d}" for i in range(40)])

    divisao = data.split_sessions(sessoes, DATA_CONFIG)

    treino, validacao, teste = (set(divisao[k]) for k in ("train", "val", "test"))
    assert treino & validacao == set()
    assert treino & teste == set()
    assert validacao & teste == set()
    assert treino | validacao | teste == set(sessoes)


def test_divisao_e_estavel_entre_chamadas():
    """A divisão usa semente própria e fixa, não a seed de treinamento."""

    sessoes = np.array([f"sessao-{i:02d}" for i in range(40)])

    primeira = data.split_sessions(sessoes, DATA_CONFIG)
    segunda = data.split_sessions(sessoes, DATA_CONFIG)

    for chave in ("train", "val", "test"):
        assert list(primeira[chave]) == list(segunda[chave])


def test_divisao_respeita_as_fracoes():
    """As proporções saem próximas das declaradas em DataConfig."""

    sessoes = np.array([f"sessao-{i:02d}" for i in range(100)])

    divisao = data.split_sessions(sessoes, DATA_CONFIG)

    assert len(divisao["train"]) == 70
    assert len(divisao["val"]) == 15
    assert len(divisao["test"]) == 15


def test_divisao_exige_sessoes_suficientes():
    """Poucas sessões deixariam algum split vazio; isso precisa falhar alto."""

    with pytest.raises(ValueError, match="sessões"):
        data.split_sessions(np.array(["a", "b"]), DATA_CONFIG)


# --------------------------------------------------------------------------
# Normalização
# --------------------------------------------------------------------------


def test_normalizacao_usa_somente_o_treino():
    """Calcular estatísticas com validação ou teste é vazamento de dados."""

    gerador = np.random.default_rng(0)
    treino = gerador.normal(10.0, 2.0, size=(50, 60, 6)).astype(np.float32)
    # Escala deliberadamente diferente: se as estatísticas vazarem, aparece.
    teste = gerador.normal(900.0, 70.0, size=(20, 60, 6)).astype(np.float32)

    media, desvio = data.fit_normalizer(treino)
    treino_normalizado = data.apply_normalizer(treino, media, desvio)
    teste_normalizado = data.apply_normalizer(teste, media, desvio)

    assert np.allclose(treino_normalizado.mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(treino_normalizado.std(axis=(0, 1)), 1.0, atol=1e-4)
    # O teste continua deslocado justamente porque não entrou no ajuste.
    assert np.abs(teste_normalizado.mean()) > 10.0


def test_normalizacao_tolera_canal_constante():
    """Desvio zero não pode virar divisão por zero."""

    constante = np.full((10, 60, 6), 3.0, dtype=np.float32)

    media, desvio = data.fit_normalizer(constante)
    saida = data.apply_normalizer(constante, media, desvio)

    assert np.all(np.isfinite(saida))


# --------------------------------------------------------------------------
# Integração: exige o arquivo real baixado.
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_reproduz_os_numeros_do_plano_experimental():
    """85 segmentos e 23.073 janelas, como registrado em PLANO-EXPERIMENTAL.md."""

    janelas, sessoes = data.build_windows(DATA_CONFIG)

    assert len(janelas) == 23_073
    assert janelas.shape[1:] == DATA_CONFIG.window_shape
    assert len(sessoes) == len(janelas)


@pytest.mark.integration
def test_sessoes_especiais_ficam_de_fora():
    """As quatro sessões de nome especial não entram no conjunto principal."""

    _, sessoes = data.build_windows(DATA_CONFIG)

    for sufixo in DATA_CONFIG.special_sessions:
        assert not any(nome.endswith(sufixo) for nome in sessoes)


@pytest.mark.integration
def test_carregamento_completo_e_deterministico():
    """Rodar duas vezes com a mesma configuração dá exatamente o mesmo tensor."""

    primeira = data.load_dataset(DATA_CONFIG, use_cache=False)
    segunda = data.load_dataset(DATA_CONFIG, use_cache=False)

    assert np.array_equal(primeira.train.X, segunda.train.X)
    assert np.array_equal(primeira.test.X, segunda.test.X)
