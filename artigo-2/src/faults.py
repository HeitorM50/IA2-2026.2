"""Injeção controlada de falhas de sensor na telemetria do Artigo 2.

O modelo é treinado apenas em operação normal e nunca vê uma falha. Para medir
se ele é bom, as anomalias do conjunto de avaliação são **injetadas de propósito**
em janelas normais: só assim se sabe exatamente onde a falha está, porque foi
este módulo que a colocou lá.

Nenhuma classe é inventada. As cinco correspondem a modos de falha que a
documentação do `baja-telemetry-api` descreve como reais, e essa procedência é o
que separa este trabalho de "apliquei um autoencoder num dataset".

As injeções operam em **unidades físicas** (RPM, km/h, kPa), nunca em valor
normalizado: uma troca de constante de escala no firmware multiplica o RPM real,
e o efeito só é interpretável antes da normalização. Carregue os dados com
`data.load_dataset(..., normalize=False)` e normalize depois.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.config import (
    DATA_CONFIG,
    FAULT_CONFIG,
    FAULT_NAMES,
    DataConfig,
    FaultConfig,
)


@dataclass(frozen=True)
class Injection:
    """Resultado de uma injeção em uma única janela."""

    X: np.ndarray
    """Janela corrompida, `(window_s, n_channels)` em unidades físicas."""

    mask: np.ndarray
    """`(window_s,)` booleano: amostras que de fato mudaram."""

    fault: str
    """Nome da classe de falha."""

    channel: int
    """Índice do canal afetado."""

    start: int
    """Amostra em que a falha começa. É daqui que a latência é medida."""

    stop: int
    """Amostra em que a falha termina (exclusivo)."""


@dataclass(frozen=True)
class EvaluationSet:
    """Conjunto de avaliação com janelas normais e corrompidas misturadas."""

    X: np.ndarray
    """`(n, window_s, n_channels)` em unidades físicas."""

    y: np.ndarray
    """`(n,)` 1 para janela corrompida, 0 para normal."""

    fault: np.ndarray
    """`(n,)` nome da classe, ou cadeia vazia quando a janela é normal."""

    mask: np.ndarray
    """`(n, window_s)` amostras alteradas de cada janela."""

    start: np.ndarray
    """`(n,)` amostra de início da falha, ou -1 quando a janela é normal."""

    def counts(self) -> dict[str, int]:
        """Quantas janelas de cada classe, para registrar no JSON da execução."""

        contagem = {"normal": int((self.y == 0).sum())}
        for nome in FAULT_NAMES:
            contagem[nome] = int((self.fault == nome).sum())
        return contagem


def _onset(rng: np.random.Generator, length: int, cfg: FaultConfig) -> int:
    """Sorteia onde a falha começa, dentro da primeira parte da janela.

    A falha nunca começa na última amostra: precisa sobrar trecho afetado para a
    latência de detecção significar alguma coisa.
    """

    limite = max(1, int(length * cfg.onset_max_fraction))
    return int(rng.integers(1, limite + 1))


def _finish(
    original: np.ndarray,
    corrupted: np.ndarray,
    fault: str,
    channel: int,
    start: int,
    stop: int,
) -> Injection:
    """Monta o resultado, derivando a máscara por comparação com o original.

    A máscara sai da diferença real, e não da região pretendida: se a injeção não
    alterou uma amostra — por exemplo, um valor travado que já era igual ao
    anterior — ela não deve ser contada como alterada.
    """

    mudou = corrupted[:, channel] != original[:, channel]
    return Injection(
        X=corrupted.astype(np.float32, copy=False),
        mask=mudou,
        fault=fault,
        channel=channel,
        start=start,
        stop=stop,
    )


# --------------------------------------------------------------------------
# As cinco classes
# --------------------------------------------------------------------------


def inject_gain(
    window: np.ndarray,
    channel: int,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    channel_std: float = 1.0,
) -> Injection:
    """Ganho/escala errada — `docs/05`, a falha central do artigo.

    O firmware troca `RPM_ESCALA` de 0,25 para 0,125 e o backend passa a gravar o
    dobro do RPM real. O valor continua plausível (6.000 RPM é um número válido
    para um motor), nenhuma checagem de faixa dispara, e o erro só aparece na
    **relação** com os demais sinais: a velocidade, o fluxo de ar e a pressão
    continuam se comportando como o RPM anterior.
    """

    corrompida = window.copy()
    inicio = _onset(rng, len(window), cfg)
    corrompida[inicio:, channel] *= cfg.gain_factor
    return _finish(window, corrompida, "gain", channel, inicio, len(window))


def inject_stuck(
    window: np.ndarray,
    channel: int,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    channel_std: float = 1.0,
) -> Injection:
    """Sensor travado — a leitura congela sem sair da faixa válida.

    Também é silenciosa para uma regra de faixa: o valor congelado é um valor
    legítimo. O que denuncia é ele parar de acompanhar os outros canais.
    """

    corrompida = window.copy()
    inicio = _onset(rng, len(window), cfg)
    corrompida[inicio:, channel] = window[inicio - 1, channel]
    return _finish(window, corrompida, "stuck", channel, inicio, len(window))


def inject_drift(
    window: np.ndarray,
    channel: int,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    channel_std: float = 1.0,
) -> Injection:
    """Deriva — viés que cresce devagar, típico de deriva térmica de sensor.

    O valor final do viés é dado em múltiplos do desvio padrão do canal, para a
    severidade não depender da unidade. O sinal da deriva é sorteado: ela tanto
    pode puxar para cima quanto para baixo.
    """

    corrompida = window.copy()
    inicio = _onset(rng, len(window), cfg)
    n = len(window) - inicio
    sentido = 1.0 if rng.random() < 0.5 else -1.0
    rampa = np.linspace(0.0, cfg.drift_final_std * channel_std * sentido, n)
    corrompida[inicio:, channel] += rampa.astype(window.dtype)
    return _finish(window, corrompida, "drift", channel, inicio, len(window))


def inject_spike(
    window: np.ndarray,
    channel: int,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    channel_std: float = 1.0,
) -> Injection:
    """Pico/outlier — ruído elétrico no barramento CAN.

    É a classe em que o limiar estatístico deve ganhar do autoencoder: detectar
    outlier é exatamente aquilo para que um limiar foi feito.
    """

    corrompida = window.copy()
    inicio = _onset(rng, len(window), cfg)

    disponiveis = np.arange(inicio, len(window))
    quantos = min(cfg.spike_count, len(disponiveis))
    posicoes = rng.choice(disponiveis, size=quantos, replace=False)
    sinais = rng.choice(np.array([-1.0, 1.0]), size=quantos)

    corrompida[posicoes, channel] += (
        sinais * cfg.spike_magnitude_std * channel_std
    ).astype(window.dtype)
    return _finish(window, corrompida, "spike", channel, inicio, int(posicoes.max()) + 1)


def inject_gap(
    window: np.ndarray,
    channel: int,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    channel_std: float = 1.0,
) -> Injection:
    """Lacuna — perda curta de pacotes, `docs/06`.

    A reamostragem repete o último valor conhecido, então a perda aparece como um
    trecho de valores congelados. A diferença para `inject_stuck` é a **duração**:
    aqui a leitura volta ao normal depois de poucos segundos.

    Perdas longas não entram nesta classe por construção: lacunas acima de
    `DataConfig.gap_s` encerram o segmento no `data.py` e nunca chegam a existir
    dentro de uma janela. Essa dependência entre as duas classes precisa ser dita
    na Discussão, e não escondida.
    """

    corrompida = window.copy()
    inicio = _onset(rng, len(window), cfg)
    duracao = int(rng.integers(cfg.gap_min_s, cfg.gap_max_s + 1))
    fim = min(inicio + duracao, len(window))
    corrompida[inicio:fim, channel] = window[inicio - 1, channel]
    return _finish(window, corrompida, "gap", channel, inicio, fim)


Injector = Callable[
    [np.ndarray, int, np.random.Generator, FaultConfig, float], Injection
]

INJECTORS: dict[str, Injector] = {
    "gain": inject_gain,
    "stuck": inject_stuck,
    "drift": inject_drift,
    "spike": inject_spike,
    "gap": inject_gap,
}


# --------------------------------------------------------------------------
# Checagem de faixa — o que um `if` de firmware faria
# --------------------------------------------------------------------------


def violates_range(
    window: np.ndarray,
    data_cfg: DataConfig = DATA_CONFIG,
    fault_cfg: FaultConfig = FAULT_CONFIG,
) -> bool:
    """Diz se algum canal saiu da faixa fisicamente plausível.

    Não é um modelo: é a regra ingênua contra a qual o artigo argumenta. As
    falhas de ganho e de sensor travado passam por aqui sem disparar nada, e é
    justamente isso que justifica um detector aprendido.
    """

    for posicao, canal in enumerate(data_cfg.channels):
        minimo, maximo = fault_cfg.range_for(canal)
        coluna = window[..., posicao]
        if np.any(coluna < minimo) or np.any(coluna > maximo):
            return True
    return False


# --------------------------------------------------------------------------
# Montagem do conjunto de avaliação
# --------------------------------------------------------------------------


def build_evaluation_set(
    windows: np.ndarray,
    channel_std: np.ndarray,
    rng: np.random.Generator,
    cfg: FaultConfig = FAULT_CONFIG,
    data_cfg: DataConfig = DATA_CONFIG,
) -> EvaluationSet:
    """Corrompe uma fração das janelas, distribuindo as cinco classes por igual.

    `windows` precisa estar em unidades físicas e conter **somente operação
    normal**. A fração corrompida é deliberadamente pequena: é esse
    desbalanceamento que justifica o AUC-PR como métrica principal, e inflá-lo
    deixaria o problema artificialmente fácil.
    """

    if windows.ndim != 3:
        raise ValueError("Esperado um tensor (n, amostras, canais).")
    if len(channel_std) != windows.shape[2]:
        raise ValueError("Um desvio por canal é necessário para escalar as falhas.")

    n = len(windows)
    n_anomalas = round(n * cfg.anomaly_fraction)
    if n_anomalas < len(FAULT_NAMES):
        raise ValueError(
            f"{n} janelas geram apenas {n_anomalas} anomalias, insuficiente para "
            f"cobrir as {len(FAULT_NAMES)} classes."
        )

    corrompidas = np.zeros_like(windows)
    corrompidas[:] = windows

    rotulos = np.zeros(n, dtype=np.int8)
    classes = np.full(n, "", dtype="<U8")
    mascaras = np.zeros((n, windows.shape[1]), dtype=bool)
    inicios = np.full(n, -1, dtype=np.int32)

    escolhidas = rng.choice(n, size=n_anomalas, replace=False)
    # Rodízio entre as classes: cada uma recebe aproximadamente a mesma fatia,
    # o que mantém a tabela por classe de falha com amostra comparável.
    atribuidas = np.array(FAULT_NAMES)[np.arange(n_anomalas) % len(FAULT_NAMES)]

    for indice, nome in zip(escolhidas, atribuidas):
        canal = data_cfg.channels.index(cfg.channel_for(nome))
        resultado = INJECTORS[nome](
            windows[indice], canal, rng, cfg, float(channel_std[canal])
        )
        corrompidas[indice] = resultado.X
        rotulos[indice] = 1
        classes[indice] = nome
        mascaras[indice] = resultado.mask
        inicios[indice] = resultado.start

    return EvaluationSet(
        X=corrompidas.astype(np.float32, copy=False),
        y=rotulos,
        fault=classes,
        mask=mascaras,
        start=inicios,
    )
