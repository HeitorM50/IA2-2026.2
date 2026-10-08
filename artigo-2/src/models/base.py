"""Interface comum aos três detectores do Artigo 2.

A contribuição do artigo é comparativa, e ela só vale se os três modelos
passarem exatamente pelo mesmo protocolo. Esta classe existe para que a
diferença entre eles seja apenas o mecanismo de pontuação — nunca a forma de
treinar, agregar ou decidir.

Todo detector precisa produzir **escore por amostra**, e não só por janela: a
latência de detecção mede quantas amostras se passam entre o início da falha e o
instante em que o escore cruza o limiar, e isso é impossível a partir de um único
número por janela.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.config import DETECTION_CONFIG, DetectionConfig


class Detector(ABC):
    """Contrato comum. O escore é sempre "quanto isto destoa do normal"."""

    name: str = "detector"
    fit_mode: str = "closed_form"

    def __init__(self, cfg: DetectionConfig = DETECTION_CONFIG) -> None:
        self.cfg = cfg
        self._fitted = False

    # --- a implementar por cada modelo ----------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Detector":
        """Ajusta o detector. Recebe SOMENTE janelas normais de treino.

        Que o conjunto seja de fato normal é garantido por asserção em
        `train.py`, não por confiança.
        """

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Escore por amostra: `(n, window_s)`. Maior significa mais anômalo."""

    @abstractmethod
    def n_parameters(self) -> int:
        """Quantidade de parâmetros aprendidos, para a coluna de custo da tabela."""

    # --- comum a todos --------------------------------------------------

    def score_windows(self, X: np.ndarray) -> np.ndarray:
        """Agrega o escore por amostra num escore por janela.

        A agregação vem do `DetectionConfig` e é idêntica para os três modelos.
        """

        return self.aggregate_scores(self.score_samples(X))

    def aggregate_scores(self, sample_scores: np.ndarray) -> np.ndarray:
        """Agrega escores já calculados sem consultar o conjunto novamente."""

        por_amostra = np.asarray(sample_scores, dtype=np.float64)
        if por_amostra.ndim != 2 or por_amostra.shape[1] == 0:
            raise ValueError(
                "Esperados escores com formato (janelas, amostras); "
                f"recebido {por_amostra.shape}."
            )
        if not np.isfinite(por_amostra).all():
            raise ValueError("Os escores por amostra devem ser finitos.")
        if self.cfg.window_aggregation == "max":
            return por_amostra.max(axis=1)
        return por_amostra.mean(axis=1)

    def _mark_fitted(self) -> None:
        """Marca como ajustado um detector treinado pelo loop MSE comum."""

        self._fitted = True

    def model_details(self) -> dict[str, Any]:
        """Metadados específicos do modelo que devem acompanhar o JSON."""

        self._check_fitted()
        return {}

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Decisão binária por janela no limiar dado.

        Sem limiar explícito usa o ponto de operação nominal em desvios. Na
        avaliação do artigo o limiar vem da calibração na validação normal, não
        daqui.
        """

        corte = self.cfg.threshold_sigmas if threshold is None else threshold
        return (self.score_windows(X) > corte).astype(np.int8)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"O detector '{self.name}' precisa ser ajustado antes de pontuar."
            )

    @staticmethod
    def _check_shape(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                f"Esperado um tensor (n, amostras, canais); recebido {X.shape}."
            )
        return np.asarray(X, dtype=np.float32)
