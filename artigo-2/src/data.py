"""Carregamento e janelamento da telemetria do Artigo 2.

O módulo transforma os CSV do Automotive OBD-II Dataset (KIT, DOI 10.35097/1130)
no tensor de janelas que todos os modelos consomem.

Contrato publicado na issue #27:

    X       float32  (n_janelas, 60, 6)
    session           (n_janelas,)  nome da sessão de origem

Os canais seguem a ordem de `DataConfig.channels`.

Toda a saída é NumPy: nada aqui depende de PyTorch. Os modelos A (limiar) e B
(PCA) não precisam de torch, e manter a dependência fora daqui deixa o pipeline
testável sem instalar o framework.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMN_NAMES, DATA_CONFIG, DataConfig


@dataclass(frozen=True)
class Split:
    """Um conjunto de janelas e a sessão de origem de cada uma."""

    X: np.ndarray
    session: np.ndarray

    def __len__(self) -> int:
        return len(self.X)

    @property
    def sessions(self) -> list[str]:
        """Nomes distintos das sessões presentes, em ordem."""

        return sorted(set(self.session.tolist()))


@dataclass(frozen=True)
class Dataset:
    """Os três splits normalizados e as estatísticas usadas na normalização."""

    train: Split
    val: Split
    test: Split
    mean: np.ndarray
    std: np.ndarray

    def summary(self) -> dict[str, object]:
        """Resumo serializável, para entrar no JSON de cada execução."""

        return {
            "janelas": {
                "treino": len(self.train),
                "validacao": len(self.val),
                "teste": len(self.test),
            },
            "sessoes": {
                "treino": len(self.train.sessions),
                "validacao": len(self.val.sessions),
                "teste": len(self.test.sessions),
            },
            "media_treino": self.mean.tolist(),
            "desvio_treino": self.std.tolist(),
        }


# --------------------------------------------------------------------------
# Obtenção do arquivo
# --------------------------------------------------------------------------


def download(cfg: DataConfig = DATA_CONFIG) -> Path:
    """Baixa e extrai o dataset, se ainda não estiver em disco.

    O endereço entrega um pacote BagIt (tar) contendo um zip com os CSV e um
    manifesto MD5. A integridade do zip interno é conferida contra esse
    manifesto antes da extração.
    """

    if cfg.csv_dir.exists():
        return cfg.csv_dir

    cfg.data_root.mkdir(parents=True, exist_ok=True)
    if not cfg.archive_path.exists():
        urllib.request.urlretrieve(cfg.dataset_url, cfg.archive_path)

    with tarfile.open(cfg.archive_path) as tar:
        tar.extractall(cfg.data_root, filter="data")

    interno = cfg.data_root / cfg.bagit_inner_zip
    md5 = hashlib.md5(interno.read_bytes()).hexdigest()
    if md5 != cfg.bagit_inner_md5:
        raise RuntimeError(
            "MD5 do zip interno não confere com o manifesto BagIt.\n"
            f"  esperado: {cfg.bagit_inner_md5}\n"
            f"  obtido  : {md5}"
        )

    with zipfile.ZipFile(interno) as zf:
        zf.extractall(cfg.data_root)

    return cfg.csv_dir


def _short_name(column: str) -> str:
    """Remove a unidade entre colchetes e aplica o nome curto do canal."""

    base = re.sub(r"\s*\[.*\]\s*$", "", column).strip()
    return COLUMN_NAMES.get(base, base)


def load_session(path: Path) -> pd.DataFrame:
    """Lê uma sessão e converte a coluna de tempo para timedelta."""

    frame = pd.read_csv(path, encoding="latin-1")
    frame.columns = [_short_name(c) for c in frame.columns]
    frame["Time"] = pd.to_timedelta(frame["Time"])
    return frame


def session_paths(cfg: DataConfig = DATA_CONFIG) -> list[Path]:
    """Sessões utilizáveis, já sem as quatro de nome especial.

    As especiais descrevem condição de direção (`Vollbremsung`, `Glatteis`,
    `Beschleunigung`) ou erro de medição (`Messfehler`). Ficam fora do conjunto
    principal por precaução; a `Messfehler` é retomada na Discussão como
    verificação qualitativa.
    """

    download(cfg)
    todas = sorted(cfg.csv_dir.glob("*.csv"))
    return [p for p in todas if p.stem.split("_")[-1] not in cfg.special_sessions]


# --------------------------------------------------------------------------
# Segmentação e janelamento
# --------------------------------------------------------------------------


def segment_frame(frame: pd.DataFrame, cfg: DataConfig = DATA_CONFIG) -> list[np.ndarray]:
    """Corta a sessão nas lacunas e reamostra cada trecho contínuo para 1 Hz.

    A taxa de linha do CSV é de ~11,3 Hz, mas cada linha repete o último valor
    conhecido de cada canal: o canal mais rápido muda a 1,11 Hz. Reamostrar para
    1 Hz evita que o autoencoder aprenda a copiar valor retido, o que produziria
    erro de reconstrução artificialmente baixo.

    Segmentos menores que uma janela são descartados.
    """

    intervalo = frame["Time"].diff().dt.total_seconds()
    marcado = frame.assign(_segmento=(intervalo > cfg.gap_s).cumsum())

    regra = f"{1 / cfg.resample_hz:g}s"
    segmentos: list[np.ndarray] = []
    for _, bloco in marcado.groupby("_segmento", sort=True):
        reamostrado = (
            bloco.set_index("Time")[list(cfg.channels)]
            .resample(regra)
            .last()
            .ffill()
            .dropna()
        )
        if len(reamostrado) >= cfg.window_s * cfg.resample_hz:
            segmentos.append(reamostrado.to_numpy(dtype=np.float32))
    return segmentos


def make_windows(segment: np.ndarray, cfg: DataConfig = DATA_CONFIG) -> np.ndarray:
    """Recorta um segmento contínuo em janelas deslizantes.

    Devolve um tensor `(n_janelas, window_s, n_canais)`. As janelas se sobrepõem
    quando o passo é menor que a janela — motivo pelo qual o split precisa ser
    por sessão, e nunca por janela.
    """

    comprimento = cfg.window_s * cfg.resample_hz
    if len(segment) < comprimento:
        return np.empty((0, *cfg.window_shape), dtype=np.float32)

    inicios = range(0, len(segment) - comprimento + 1, cfg.stride_s * cfg.resample_hz)
    janelas = np.stack([segment[i : i + comprimento] for i in inicios])
    return janelas.astype(np.float32, copy=False)


def build_windows(cfg: DataConfig = DATA_CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Percorre todas as sessões utilizáveis e devolve as janelas sem normalizar.

    A normalização fica de fora de propósito: ela precisa ser ajustada depois do
    split, usando apenas o treino.
    """

    blocos: list[np.ndarray] = []
    origens: list[np.ndarray] = []

    for caminho in session_paths(cfg):
        nome = caminho.stem
        for segmento in segment_frame(load_session(caminho), cfg):
            janelas = make_windows(segmento, cfg)
            if len(janelas) == 0:
                continue
            blocos.append(janelas)
            origens.append(np.full(len(janelas), nome))

    if not blocos:
        raise RuntimeError("Nenhuma janela foi construída; verifique o download.")

    return np.concatenate(blocos), np.concatenate(origens)


# --------------------------------------------------------------------------
# Divisão por sessão
# --------------------------------------------------------------------------


def split_sessions(
    sessions: np.ndarray, cfg: DataConfig = DATA_CONFIG
) -> dict[str, np.ndarray]:
    """Reparte os nomes de sessão entre treino, validação e teste.

    A divisão é **por sessão**, nunca por janela: janelas vizinhas se sobrepõem
    no tempo, e um sorteio por janela colocaria trechos quase idênticos nos dois
    lados da avaliação.

    O sorteio usa `cfg.split_seed`, uma semente própria e fixa, independente das
    seeds de treinamento. Assim a variação entre as três execuções canônicas
    mede variação de treinamento, e não troca das sessões avaliadas.
    """

    distintas = np.array(sorted(set(sessions.tolist())))
    minimo = 3
    if len(distintas) < minimo:
        raise ValueError(
            f"São necessárias ao menos {minimo} sessões para dividir em três "
            f"conjuntos; foram encontradas {len(distintas)}."
        )

    embaralhadas = np.random.default_rng(cfg.split_seed).permutation(distintas)

    n = len(embaralhadas)
    n_treino = max(1, round(n * cfg.train_fraction))
    n_validacao = max(1, round(n * cfg.val_fraction))
    # O teste leva o restante, e precisa sobrar ao menos uma sessão para ele.
    if n_treino + n_validacao >= n:
        n_treino = n - 2
        n_validacao = 1

    return {
        "train": embaralhadas[:n_treino],
        "val": embaralhadas[n_treino : n_treino + n_validacao],
        "test": embaralhadas[n_treino + n_validacao :],
    }


# --------------------------------------------------------------------------
# Normalização
# --------------------------------------------------------------------------


def fit_normalizer(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula média e desvio por canal. Recebe SOMENTE janelas de treino.

    Ajustar com validação ou teste é vazamento de dados, e é o erro que mais
    custa nota neste tipo de trabalho.
    """

    media = windows.mean(axis=(0, 1))
    desvio = windows.std(axis=(0, 1))
    # Canal constante tem desvio zero; trocar por 1 evita divisão por zero sem
    # alterar o valor normalizado, que fica em zero de qualquer forma.
    desvio = np.where(desvio > 0, desvio, 1.0)
    return media.astype(np.float32), desvio.astype(np.float32)


def apply_normalizer(
    windows: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Aplica as estatísticas do treino a qualquer conjunto."""

    return ((windows - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------
# Ponto de entrada
# --------------------------------------------------------------------------


def load_dataset(
    cfg: DataConfig = DATA_CONFIG,
    *,
    use_cache: bool = True,
    normalize: bool = True,
) -> Dataset:
    """Monta os três splits.

    A ordem importa e é a regra mais importante do pipeline: **dividir primeiro,
    normalizar depois**, com as estatísticas vindas apenas do treino.

    Com `normalize=False` os splits saem em unidades físicas (RPM, km/h, kPa),
    e `mean`/`std` continuam sendo devolvidos, ajustados no treino. É a forma
    usada pela injeção de falhas: uma troca de constante de escala no firmware
    multiplica o RPM real, e esse efeito só faz sentido antes da normalização.
    O consumidor normaliza depois com `apply_normalizer`.
    """

    janelas, sessoes = _windows_cached(cfg) if use_cache else build_windows(cfg)
    divisao = split_sessions(sessoes, cfg)

    mascaras = {
        chave: np.isin(sessoes, nomes) for chave, nomes in divisao.items()
    }

    media, desvio = fit_normalizer(janelas[mascaras["train"]])

    def monta(mascara: np.ndarray) -> Split:
        bruto = janelas[mascara]
        return Split(
            X=apply_normalizer(bruto, media, desvio) if normalize else bruto,
            session=sessoes[mascara],
        )

    partes = {chave: monta(mascara) for chave, mascara in mascaras.items()}

    return Dataset(
        train=partes["train"],
        val=partes["val"],
        test=partes["test"],
        mean=media,
        std=desvio,
    )


def _windows_cached(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Guarda as janelas cruas em disco; construí-las custa cerca de um minuto.

    O cache fica em `src/data/`, que está no `.gitignore`.
    """

    if cfg.cache_path.exists():
        with np.load(cfg.cache_path, allow_pickle=False) as arquivo:
            return arquivo["X"], arquivo["session"]

    janelas, sessoes = build_windows(cfg)
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.cache_path, X=janelas, session=sessoes)
    return janelas, sessoes
