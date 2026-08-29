"""Configuração única do pipeline de dados do Artigo 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

