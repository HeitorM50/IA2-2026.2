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
