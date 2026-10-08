"""Modelo B — PCA usado como autoencoder linear.

Cada janela é achatada, projetada no mesmo espaço latente que será usado pelo
autoencoder LSTM e reconstruída. O erro quadrático médio entre canais produz um
escore por amostra, preservando a interface necessária à latência de detecção.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.config import (
    DETECTION_CONFIG,
    REPRESENTATION_CONFIG,
    DetectionConfig,
    RepresentationConfig,
)
from src.models.base import Detector


class PCADetector(Detector):
    """Autoencoder linear com ajuste fechado por decomposição em componentes."""

    name = "pca"
    fit_mode = "closed_form"

    def __init__(
        self,
        cfg: DetectionConfig = DETECTION_CONFIG,
        representation_cfg: RepresentationConfig = REPRESENTATION_CONFIG,
    ) -> None:
        super().__init__(cfg)
        self.representation_cfg = representation_cfg
        self.pca_: PCA | None = None
        self.window_shape_: tuple[int, int] | None = None

    @property
    def latent_dim(self) -> int:
        return self.representation_cfg.latent_dim

    def fit(self, X: np.ndarray) -> "PCADetector":
        """Ajusta o subespaço usando somente as janelas normais recebidas."""

        windows = self._check_shape(X)
        if len(windows) == 0:
            raise ValueError("O conjunto de treino não pode estar vazio.")
        flattened = windows.reshape(len(windows), -1).astype(np.float64)
        maximum_components = min(flattened.shape)
        if self.latent_dim >= maximum_components:
            raise ValueError(
                "A dimensão latente deve ser menor que min(janelas, atributos); "
                f"recebido {self.latent_dim} para limite {maximum_components}."
            )
        if float(np.var(flattened, axis=0).sum()) <= 0.0:
            raise ValueError("O PCA não pode ser ajustado em dados constantes.")

        self.pca_ = PCA(n_components=self.latent_dim, svd_solver="full")
        self.pca_.fit(flattened)
        self.window_shape_ = (windows.shape[1], windows.shape[2])
        self._fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Retorna o MSE de reconstrução de cada instante da janela."""

        self._check_fitted()
        windows = self._check_shape(X)
        current_shape = (windows.shape[1], windows.shape[2])
        if current_shape != self.window_shape_:
            raise ValueError(
                "O formato das janelas diverge do usado no ajuste: "
                f"esperado {self.window_shape_}, recebido {current_shape}."
            )

        flattened = windows.reshape(len(windows), -1).astype(np.float64)
        reconstructed = self.pca_.inverse_transform(self.pca_.transform(flattened))
        reconstructed = reconstructed.reshape((len(windows), *self.window_shape_))
        return np.mean(
            np.square(windows.astype(np.float64) - reconstructed),
            axis=2,
        )

    def n_parameters(self) -> int:
        """Componentes e média necessários para projetar e reconstruir."""

        self._check_fitted()
        return int(self.pca_.components_.size + self.pca_.mean_.size)

    def model_details(self) -> dict[str, Any]:
        """Registra compressão e variância explicada para a Discussão."""

        self._check_fitted()
        ratios = np.asarray(self.pca_.explained_variance_ratio_, dtype=np.float64)
        if not np.isfinite(ratios).all():
            raise RuntimeError("O PCA produziu variância explicada não finita.")
        return {
            "latent_dim": self.latent_dim,
            "input_dim": int(self.pca_.n_features_in_),
            "explained_variance_ratio": ratios.tolist(),
            "explained_variance_total": float(ratios.sum()),
        }


def build(
    cfg: DetectionConfig = DETECTION_CONFIG,
    representation_cfg: RepresentationConfig = REPRESENTATION_CONFIG,
) -> PCADetector:
    """Construtor usado pelo orquestrador comum."""

    return PCADetector(cfg, representation_cfg)
