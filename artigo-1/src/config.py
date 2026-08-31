"""Configuração única do pipeline de dados do Artigo 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class DataConfig:
    """Valores imutáveis usados para carregar e transformar o BloodMNIST."""

    dataset_flag: str = "bloodmnist"
    dataset_size: int = 64
    dataset_doi: str = "10.5281/zenodo.10519652"
    data_root: Path = Path(__file__).resolve().parent / "data"
    summary_path: Path = Path(__file__).resolve().parent / "dataset-summary.json"

    train_size: int = 11_959
    val_size: int = 1_712
    test_size: int = 3_421
    channels: int = 3
    num_classes: int = 8

    batch_size: int = 64
    stats_batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    train_drop_last: bool = False

    rotation_degrees: float = 15.0
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5

    canonical_seeds: tuple[int, ...] = (42, 1337, 2026)
    class_names_pt: tuple[str, ...] = (
        "basófilo",
        "eosinófilo",
        "eritroblasto",
        "granulócito imaturo",
        "linfócito",
        "monócito",
        "neutrófilo",
        "plaqueta",
    )

    @property
    def expected_split_sizes(self) -> dict[str, int]:
        """Retorna os tamanhos oficiais em uma estrutura conveniente."""

        return {
            "train": self.train_size,
            "val": self.val_size,
            "test": self.test_size,
        }


DATA_CONFIG = DataConfig()


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros de uma família de modelos sob o protocolo comum."""

    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    head_learning_rate: float = 1e-3
    head_parameter_prefix: str = ""
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    max_train_batches: int = 0
    max_eval_batches: int = 0

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("O protocolo comum suporta somente o otimizador AdamW.")
        if self.learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("As taxas de aprendizado devem ser positivas.")
        if self.weight_decay < 0:
            raise ValueError("O weight decay não pode ser negativo.")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("Épocas máximas e paciência devem ser positivas.")
        if self.min_delta < 0:
            raise ValueError("O min_delta não pode ser negativo.")
        if self.max_train_batches < 0 or self.max_eval_batches < 0:
            raise ValueError("Limites de lotes não podem ser negativos.")

    def as_dict(self) -> dict[str, Any]:
        """Converte a configuração para um objeto diretamente serializável."""

        return asdict(self)

    def for_quick_run(self) -> TrainingConfig:
        """Limita uma configuração para uma verificação local não canônica."""

        return replace(
            self,
            max_epochs=2,
            patience=1,
            max_train_batches=10,
            max_eval_batches=5,
        )


@dataclass(frozen=True)
class ModelSpec:
    """Localização do builder e configuração usada pelo orquestrador."""

    module: str
    training_config: str


TRAINING_CONFIGS: Mapping[str, TrainingConfig] = MappingProxyType(
    {
        "toy": TrainingConfig(
            learning_rate=1e-2,
            head_learning_rate=1e-2,
            max_epochs=5,
            patience=2,
        ),
        "logreg": TrainingConfig(
            learning_rate=1e-3,
            head_learning_rate=1e-3,
        ),
        "cnn": TrainingConfig(
            learning_rate=1e-3,
            head_learning_rate=1e-3,
        ),
        "resnet18": TrainingConfig(
            learning_rate=1e-4,
            head_learning_rate=1e-3,
            head_parameter_prefix="fc.",
        ),
    }
)


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "logreg": ModelSpec("src.models.baseline", "logreg"),
        "cnn": ModelSpec("src.models.cnn", "cnn"),
        "resnet18": ModelSpec("src.models.transfer", "resnet18"),
    }
)
