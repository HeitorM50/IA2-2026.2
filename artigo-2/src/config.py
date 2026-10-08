"""Configuração única do pipeline do Artigo 2.

Nenhuma constante usada por `data.py`, pelos modelos ou pelo orquestrador pode
ser redefinida fora deste módulo. A reprodutibilidade pela leitura é critério de
correção declarado, e ela se perde assim que o mesmo número existe em dois
lugares.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """Valores imutáveis usados para construir as janelas de telemetria."""

    # --- origem ---------------------------------------------------------
    dataset_name: str = "Automotive OBD-II Dataset"
    dataset_doi: str = "10.35097/1130"
    dataset_license: str = "CC BY 4.0"
    dataset_url: str = (
        "https://www.radar-service.eu/radar-backend/archives/"
        "bCtGxdTklQlfQcAq/versions/1/content"
    )
    # O endereço entrega um pacote BagIt: um tar que embrulha o zip com os CSV.
    bagit_inner_zip: str = "10.35097-1130/data/dataset/OBD-II-Dataset.zip"
    # MD5 declarado em manifest-md5.txt dentro do próprio pacote.
    bagit_inner_md5: str = "4aece6c7b16f59a69be4fafec7ad54d9"

    data_root: Path = SRC / "data"
    archive_name: str = "obd-ii-dataset.tar"
    csv_dirname: str = "OBD-II-Dataset"
    cache_name: str = "janelas.npz"

    # --- canais ---------------------------------------------------------
    # Selecionados por taxa real de mudança acima de 0,3 Hz e sem redundância.
    # Ficaram de fora: pedal_e (correlação de 0,99 com pedal_d, é o par
    # redundante do pedal), acelerador_abs, temp_arref e temp_ambiente (todos
    # abaixo de 0,04 Hz de mudança real). Ver PLANO-EXPERIMENTAL.md.
    channels: tuple[str, ...] = (
        "rpm",
        "velocidade",
        "fluxo_ar",
        "pressao_adm",
        "temp_adm",
        "pedal_d",
    )

    # --- janelamento ----------------------------------------------------
    # A taxa de linha do CSV é de ~11,3 Hz, mas cada linha repete o último
    # valor conhecido: o canal mais rápido muda a 1,11 Hz. Reamostrar para 1 Hz
    # evita que o autoencoder aprenda a copiar valor retido.
    resample_hz: int = 1
    window_s: int = 60
    stride_s: int = 10
    # Lacuna acima deste valor encerra o segmento contínuo. A lacuna é operação
    # normal (buffer despejado ao voltar à cobertura), não anomalia, e não pode
    # atravessar uma janela.
    gap_s: float = 2.0

    # --- sessões --------------------------------------------------------
    # Sufixos que descrevem condição de direção ou erro de medição. Ficam fora
    # do conjunto de treino; a Messfehler é usada como verificação qualitativa.
    special_sessions: tuple[str, ...] = (
        "Messfehler",
        "Vollbremsung",
        "Glatteis",
        "Beschleunigung",
    )

    # --- divisão --------------------------------------------------------
    # O split é por sessão, nunca aleatório por janela: janelas vizinhas se
    # sobrepõem no tempo e um sorteio por janela vazaria treino para teste.
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    # O teste recebe o restante.

    # O split usa uma semente PRÓPRIA e fixa, independente das seeds de
    # treinamento. Assim a variação entre as três execuções mede variação de
    # treinamento, e não troca das sessões avaliadas — mesmo princípio dos
    # splits oficiais fixos adotados no Artigo 1.
    split_seed: int = 20261009

    canonical_seeds: tuple[int, ...] = (42, 1337, 2026)

    def __post_init__(self) -> None:
        if self.window_s <= 0 or self.stride_s <= 0:
            raise ValueError("Janela e passo devem ser positivos.")
        if self.stride_s > self.window_s:
            raise ValueError("O passo não pode ser maior que a janela.")
        if self.gap_s <= 0:
            raise ValueError("O limiar de lacuna deve ser positivo.")
        if self.resample_hz != 1:
            raise ValueError(
                "O protocolo fixa 1 Hz; outra taxa exige rever o PLANO-EXPERIMENTAL."
            )
        if not 0 < self.train_fraction < 1:
            raise ValueError("A fração de treino deve ficar entre 0 e 1.")
        if not 0 < self.val_fraction < 1:
            raise ValueError("A fração de validação deve ficar entre 0 e 1.")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("Treino e validação não deixam sessões para o teste.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Há canal repetido na seleção.")

    # --- derivados ------------------------------------------------------

    @property
    def archive_path(self) -> Path:
        return self.data_root / self.archive_name

    @property
    def csv_dir(self) -> Path:
        return self.data_root / self.csv_dirname

    @property
    def cache_path(self) -> Path:
        return self.data_root / self.cache_name

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.val_fraction

    @property
    def window_shape(self) -> tuple[int, int]:
        """Formato de uma janela: (amostras, canais)."""

        return (self.window_s * self.resample_hz, self.n_channels)

    def as_dict(self) -> dict[str, Any]:
        """Converte para um objeto serializável, para entrar no JSON de saída."""

        bruto = asdict(self)
        return {
            chave: str(valor) if isinstance(valor, Path) else valor
            for chave, valor in bruto.items()
        }


DATA_CONFIG = DataConfig()


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros do ajuste do autoencoder recorrente.

    Os detectores estatístico e PCA possuem ajuste fechado e ignoram estes
    valores, mas atravessam o mesmo ``train_eval`` para calibração e avaliação.
    """

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    max_train_batches: int = 0
    max_validation_batches: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("O tamanho do lote deve ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")
        if self.weight_decay < 0:
            raise ValueError("O weight decay não pode ser negativo.")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("Épocas e paciência devem ser positivas.")
        if self.min_delta < 0:
            raise ValueError("A melhora mínima não pode ser negativa.")
        if self.max_train_batches < 0 or self.max_validation_batches < 0:
            raise ValueError("Limites de lotes não podem ser negativos.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def for_quick_run(self) -> "TrainingConfig":
        """Mantém o protocolo, reduzindo apenas o trabalho para smoke tests."""

        return TrainingConfig(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=2,
            patience=1,
            min_delta=self.min_delta,
            max_train_batches=10,
            max_validation_batches=5,
        )


TRAINING_CONFIG = TrainingConfig()


@dataclass(frozen=True)
class RepresentationConfig:
    """Bottleneck compartilhado pelo PCA e pelo autoencoder LSTM."""

    latent_dim: int = 16

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("A dimensão latente deve ser positiva.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


REPRESENTATION_CONFIG = RepresentationConfig()


@dataclass(frozen=True)
class FaultConfig:
    """Parâmetros das cinco injeções de falha.

    Nenhuma falha é inventada: cada classe corresponde a um modo de falha que a
    documentação do `baja-telemetry-api` descreve como real. A procedência de
    cada uma vira texto da Metodologia.
    """

    # --- ganho / escala errada (docs/05) --------------------------------
    # O firmware troca RPM_ESCALA de 0,25 para 0,125 e o backend passa a gravar
    # o dobro do valor real. É a falha central do artigo: permanece dentro da
    # faixa válida e nenhuma checagem a detecta.
    gain_factor: float = 2.0

    # --- sensor travado -------------------------------------------------
    # A leitura congela no último valor e assim permanece até o fim da janela.

    # --- deriva ---------------------------------------------------------
    # Viés que cresce linearmente, típico de deriva térmica. O valor final é
    # dado em múltiplos do desvio padrão do canal, para a severidade não
    # depender da unidade.
    drift_final_std: float = 3.0

    # --- pico / outlier (ruído elétrico no barramento CAN) --------------
    spike_magnitude_std: float = 6.0
    spike_count: int = 3

    # --- lacuna (docs/06) -----------------------------------------------
    # Perda curta de pacotes, que a reamostragem com last-value-carried-forward
    # transforma em valores repetidos. Lacunas acima de `DataConfig.gap_s`
    # encerram o segmento e por construção nunca aparecem dentro de uma janela;
    # por isso a classe injetada cobre apenas a perda curta.
    gap_min_s: int = 3
    gap_max_s: int = 8

    # A falha começa em algum ponto da primeira metade da janela, para sempre
    # sobrar trecho afetado suficiente para a latência de detecção fazer sentido.
    onset_max_fraction: float = 0.5

    # Fração de janelas corrompidas no conjunto de avaliação. Mantida baixa de
    # propósito: é o desbalanceamento que justifica o AUC-PR como métrica
    # principal, e inflá-lo deixaria o problema artificialmente fácil.
    anomaly_fraction: float = 0.20

    # Canal padrão de cada classe, escolhido pela plausibilidade física do modo
    # de falha. O ganho vai no RPM porque é literalmente o caso do RPM_ESCALA;
    # a deriva vai na temperatura porque deriva térmica é o caso documentado.
    default_channel: tuple[tuple[str, str], ...] = (
        ("gain", "rpm"),
        ("stuck", "velocidade"),
        ("drift", "temp_adm"),
        ("spike", "pressao_adm"),
        ("gap", "fluxo_ar"),
    )

    # Faixa que um `if` de firmware consideraria válida para cada canal. NÃO são
    # os extremos observados nos dados: são limites de plausibilidade física,
    # que é o que uma validação de faixa real checaria.
    physical_range: tuple[tuple[str, float, float], ...] = (
        ("rpm", 0.0, 8000.0),
        ("velocidade", 0.0, 300.0),
        ("fluxo_ar", 0.0, 400.0),
        ("pressao_adm", 0.0, 300.0),
        ("temp_adm", -40.0, 200.0),
        ("pedal_d", 0.0, 100.0),
    )

    def __post_init__(self) -> None:
        if self.gain_factor == 1.0:
            raise ValueError("Um ganho de 1,0 não altera nada.")
        if self.spike_count <= 0:
            raise ValueError("O número de picos deve ser positivo.")
        if self.gap_min_s <= 0 or self.gap_max_s < self.gap_min_s:
            raise ValueError("A duração da lacuna está mal definida.")
        if not 0 < self.onset_max_fraction < 1:
            raise ValueError("O início da falha deve cair dentro da janela.")
        if not 0 < self.anomaly_fraction < 1:
            raise ValueError("A fração de anomalias deve ficar entre 0 e 1.")

    def channel_for(self, fault: str) -> str:
        """Canal padrão da classe de falha."""

        return dict(self.default_channel)[fault]

    def range_for(self, channel: str) -> tuple[float, float]:
        """Faixa válida declarada para o canal."""

        for nome, minimo, maximo in self.physical_range:
            if nome == channel:
                return minimo, maximo
        raise KeyError(f"Canal sem faixa declarada: {channel}")

    def as_dict(self) -> dict[str, Any]:
        """Converte para um objeto serializável, para entrar no JSON de saída."""

        return asdict(self)


FAULT_CONFIG = FaultConfig()

# Nomes das classes de falha, na ordem em que aparecem na tabela do artigo.
FAULT_NAMES: tuple[str, ...] = ("gain", "stuck", "drift", "spike", "gap")

# Rótulo em português de cada classe, para figuras e tabelas.
FAULT_LABELS_PT: dict[str, str] = {
    "gain": "ganho",
    "stuck": "travado",
    "drift": "deriva",
    "spike": "pico",
    "gap": "lacuna",
}


@dataclass(frozen=True)
class DetectionConfig:
    """Protocolo comum de pontuação, idêntico para os três detectores.

    A comparação do artigo só vale se os modelos forem avaliados exatamente da
    mesma forma. Qualquer diferença de agregação ou de calibração entre eles
    invalida a tabela.
    """

    # Cada detector produz um escore por amostra; o escore da janela é a
    # agregação desses valores. O máximo é usado porque detecção é sobre o pior
    # instante, e porque a latência — que é inerentemente por amostra — fica
    # consistente com o escore de janela.
    window_aggregation: str = "max"

    # Ponto de operação nominal da linha de base: o `if` que um firmware faria
    # dispara quando a leitura se afasta mais de 3 desvios da média de operação
    # normal. Não é o limiar usado na avaliação — esse é calibrado na validação
    # normal (#30) —, mas é a regra que dá nome ao modelo.
    threshold_sigmas: float = 3.0

    # Multiplicador da dispersão dos escores normais de validação. É separado
    # dos 3σ nominais da baseline: a calibração usa μ + λσ sobre o escore de
    # janela de cada detector, qualquer que seja a unidade desse escore.
    calibration_sigmas: float = 3.0

    # Piso para o desvio por canal, evitando divisão por zero em canal constante.
    min_std: float = 1e-8

    def __post_init__(self) -> None:
        if self.window_aggregation not in {"max", "mean"}:
            raise ValueError("Agregação deve ser 'max' ou 'mean'.")
        if self.threshold_sigmas <= 0:
            raise ValueError("O limiar em desvios deve ser positivo.")
        if self.calibration_sigmas <= 0:
            raise ValueError("O multiplicador da calibração deve ser positivo.")
        if self.min_std <= 0:
            raise ValueError("O piso do desvio deve ser positivo.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DETECTION_CONFIG = DetectionConfig()

# Nome curto de cada detector, usado no nome do JSON e na tabela do artigo.
MODEL_NAMES: tuple[str, ...] = ("limiar", "pca", "autoencoder")

MODEL_LABELS_PT: dict[str, str] = {
    "limiar": "Limiar 3σ",
    "pca": "PCA",
    "autoencoder": "Autoencoder LSTM",
}


# Nome curto de cada coluna do CSV original, sem a unidade entre colchetes.
# O dicionário cobre os dez canais do arquivo; `DataConfig.channels` escolhe
# quais entram no experimento.
COLUMN_NAMES: dict[str, str] = {
    "Engine Coolant Temperature": "temp_arref",
    "Intake Manifold Absolute Pressure": "pressao_adm",
    "Engine RPM": "rpm",
    "Vehicle Speed Sensor": "velocidade",
    "Intake Air Temperature": "temp_adm",
    "Air Flow Rate from Mass Flow Sensor": "fluxo_ar",
    "Absolute Throttle Position": "acelerador_abs",
    "Ambient Air Temperature": "temp_ambiente",
    "Accelerator Pedal Position D": "pedal_d",
    "Accelerator Pedal Position E": "pedal_e",
}


def set_seed(seed: int) -> np.random.Generator:
    """Fixa a semente de `random`, NumPy e, se disponível, PyTorch.

    Devolve o gerador NumPy da execução. Todo sorteio do pipeline deve usar
    esse gerador, nunca o estado global de `np.random`, para que duas execuções
    com a mesma semente produzam exatamente o mesmo resultado.
    """

    if seed < 0:
        raise ValueError("A semente não pode ser negativa.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # PyTorch só é necessário para o autoencoder recorrente.
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return np.random.default_rng(seed)
