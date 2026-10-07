"""Modelo A — limiar de 3 desvios por canal.

É a linha de base ingênua do artigo: literalmente o `if` que alguém escreveria no
firmware. Para cada canal, aprende média e desvio na operação normal de treino e
mede o quanto cada amostra se afasta disso.

O papel dela não é ganhar. É dar significado ao número do autoencoder: "0,83"
não diz nada sozinho; "0,83 contra 0,61 do limiar, sob o mesmo protocolo" é
resultado.

A previsão registrada no `PLANO-EXPERIMENTAL.md` é que esta linha de base vença
nos picos — outlier é exatamente aquilo para que um limiar foi feito — e perca
feio no ganho e no sensor travado, que nunca se afastam o bastante da média.
"""

from __future__ import annotations

import numpy as np

from src.config import DETECTION_CONFIG, DetectionConfig
from src.models.base import Detector


class ThresholdDetector(Detector):
    """Escore de uma amostra: o maior afastamento padronizado entre os canais."""

    name = "limiar"

    def __init__(self, cfg: DetectionConfig = DETECTION_CONFIG) -> None:
        super().__init__(cfg)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ThresholdDetector":
        """Calcula média e desvio por canal sobre as janelas normais de treino."""

        X = self._check_shape(X)
        self.mean_ = X.mean(axis=(0, 1))
        desvio = X.std(axis=(0, 1))
        # Canal constante tem desvio zero; o piso evita divisão por zero sem
        # mudar o resultado, que fica em zero de qualquer forma.
        self.std_ = np.maximum(desvio, self.cfg.min_std)
        self._fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Para cada instante, o maior |z| entre os seis canais.

        Toma-se o máximo entre canais, e não a soma, porque a regra que esta
        linha de base imita dispara quando **algum** sensor sai da faixa — não
        quando o conjunto deles, somado, destoa.
        """

        self._check_fitted()
        X = self._check_shape(X)
        z = np.abs(X - self.mean_) / self.std_
        return z.max(axis=2)

    def n_parameters(self) -> int:
        """Uma média e um desvio por canal."""

        self._check_fitted()
        return int(self.mean_.size + self.std_.size)


def build(cfg: DetectionConfig = DETECTION_CONFIG) -> ThresholdDetector:
    """Construtor usado pelo orquestrador, espelhando o padrão do Artigo 1."""

    return ThresholdDetector(cfg)
