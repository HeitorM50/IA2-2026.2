"""Regressão logística multinomial usada como linha de base."""

from __future__ import annotations

from torch import nn

from ..config import DATA_CONFIG


def build() -> nn.Module:
    """Cria a fronteira linear sobre todos os pixels normalizados."""

    input_features = (
        DATA_CONFIG.channels * DATA_CONFIG.dataset_size * DATA_CONFIG.dataset_size
    )
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(input_features, DATA_CONFIG.num_classes),
    )
