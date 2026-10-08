"""Modelo C — autoencoder LSTM sequência-a-sequência.

O codificador lê a janela na ordem temporal e a reduz a um único vetor
latente. O decodificador recebe somente esse vetor repetido ao longo da janela,
de modo que nenhuma amostra original contorne o gargalo. O erro quadrático
médio entre canais em cada instante é o escore usado pelo protocolo comum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.config import (
    AUTOENCODER_CONFIG,
    DATA_CONFIG,
    DETECTION_CONFIG,
    REPRESENTATION_CONFIG,
    AutoencoderConfig,
    DataConfig,
    DetectionConfig,
    RepresentationConfig,
)
from src.models.base import Detector


class LSTMAutoencoder(Detector, nn.Module):
    """Reconstrói uma janela completa a partir de um gargalo recorrente."""

    name = "autoencoder"
    fit_mode = "torch_mse"

    def __init__(
        self,
        cfg: DetectionConfig = DETECTION_CONFIG,
        representation_cfg: RepresentationConfig = REPRESENTATION_CONFIG,
        autoencoder_cfg: AutoencoderConfig = AUTOENCODER_CONFIG,
        data_cfg: DataConfig = DATA_CONFIG,
    ) -> None:
        nn.Module.__init__(self)
        Detector.__init__(self, cfg)
        self.representation_cfg = representation_cfg
        self.autoencoder_cfg = autoencoder_cfg
        self.data_cfg = data_cfg

        recurrent_dropout = (
            autoencoder_cfg.dropout if autoencoder_cfg.num_layers > 1 else 0.0
        )
        self.encoder = nn.LSTM(
            input_size=data_cfg.n_channels,
            hidden_size=autoencoder_cfg.hidden_size,
            num_layers=autoencoder_cfg.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.to_latent = nn.Linear(
            autoencoder_cfg.hidden_size,
            representation_cfg.latent_dim,
        )
        self.decoder = nn.LSTM(
            input_size=representation_cfg.latent_dim,
            hidden_size=autoencoder_cfg.hidden_size,
            num_layers=autoencoder_cfg.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.to_output = nn.Linear(
            autoencoder_cfg.hidden_size,
            data_cfg.n_channels,
        )

    @property
    def latent_dim(self) -> int:
        return self.representation_cfg.latent_dim

    def fit(self, X: np.ndarray) -> "LSTMAutoencoder":
        """Impede um ajuste sem a validação exigida pela parada antecipada."""

        raise RuntimeError(
            "O autoencoder deve ser ajustado por train_eval, que recebe a "
            "validação normal e restaura a melhor época."
        )

    def reset_parameters(self) -> None:
        """Reinicializa todos os pesos depois que o loop fixa a seed."""

        self.encoder.reset_parameters()
        self.to_latent.reset_parameters()
        self.decoder.reset_parameters()
        self.to_output.reset_parameters()
        self._fitted = False

    def forward(self, X: Tensor) -> Tensor:
        """Reconstrói ``X`` usando somente a representação latente."""

        if X.ndim != 3:
            raise ValueError(
                "Esperado um tensor (n, amostras, canais); "
                f"recebido {tuple(X.shape)}."
            )
        if X.shape[2] != self.data_cfg.n_channels:
            raise ValueError(
                f"Esperados {self.data_cfg.n_channels} canais; "
                f"recebidos {X.shape[2]}."
            )

        _, (hidden, _) = self.encoder(X)
        latent = self.to_latent(hidden[-1])
        decoder_input = latent.unsqueeze(1).expand(-1, X.shape[1], -1)
        decoded, _ = self.decoder(decoder_input)
        return self.to_output(decoded)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Retorna o MSE entre canais para cada amostra de cada janela."""

        self._check_fitted()
        windows = self._check_shape(X)
        expected_shape = self.data_cfg.window_shape
        if tuple(windows.shape[1:]) != expected_shape:
            raise ValueError(
                "O formato das janelas diverge da configuração: "
                f"esperado {expected_shape}, recebido {tuple(windows.shape[1:])}."
            )
        if not np.isfinite(windows).all():
            raise ValueError("As janelas devem conter somente valores finitos.")
        if len(windows) == 0:
            return np.empty((0, expected_shape[0]), dtype=np.float32)

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        scores: list[Tensor] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(windows), self.autoencoder_cfg.score_batch_size):
                    batch = torch.from_numpy(
                        windows[start : start + self.autoencoder_cfg.score_batch_size]
                    ).to(device)
                    error = (self(batch) - batch).square().mean(dim=2)
                    scores.append(error.cpu())
        finally:
            self.train(was_training)
        return torch.cat(scores).numpy()

    def n_parameters(self) -> int:
        """Conta todos os escalares aprendidos da arquitetura."""

        return sum(int(parameter.numel()) for parameter in self.parameters())

    def model_details(self) -> dict[str, Any]:
        """Expõe no JSON a arquitetura necessária à reprodução."""

        self._check_fitted()
        return {
            "architecture": "lstm_sequence_autoencoder",
            "input_channels": self.data_cfg.n_channels,
            "window_samples": self.data_cfg.window_shape[0],
            "latent_dim": self.latent_dim,
            "hidden_size": self.autoencoder_cfg.hidden_size,
            "num_layers": self.autoencoder_cfg.num_layers,
            "dropout": self.autoencoder_cfg.dropout,
            "decoder_input": "repeated_latent",
        }


def build(
    cfg: DetectionConfig = DETECTION_CONFIG,
    representation_cfg: RepresentationConfig = REPRESENTATION_CONFIG,
    autoencoder_cfg: AutoencoderConfig = AUTOENCODER_CONFIG,
    data_cfg: DataConfig = DATA_CONFIG,
) -> LSTMAutoencoder:
    """Construtor usado pelo orquestrador comum."""

    return LSTMAutoencoder(
        cfg,
        representation_cfg,
        autoencoder_cfg,
        data_cfg,
    )
