"""Verificação do Automotive OBD-II Dataset (KIT, DOI 10.35097/1130).

Reproduz os números registrados em PLANO-EXPERIMENTAL.md, seção
"Verificação prática". Rodar de dentro de artigo-2/:

    python src/verifica_dataset.py

Baixa o arquivo para src/data/ se ele ainda não existir. O diretório src/data/
está no .gitignore — o dado bruto nunca entra no Git.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://www.radar-service.eu/radar-backend/archives/bCtGxdTklQlfQcAq/versions/1/content"
DOI = "10.35097/1130"

RAIZ = Path(__file__).resolve().parent
DIR_DADOS = RAIZ / "data"
ARQUIVO = DIR_DADOS / "obd-ii-dataset.tar"
DIR_CSV = DIR_DADOS / "OBD-II-Dataset"

# O download é um pacote BagIt: um tar que embrulha o zip com os CSV.
ZIP_INTERNO = "10.35097-1130/data/dataset/OBD-II-Dataset.zip"
# MD5 declarado em manifest-md5.txt dentro do próprio pacote.
MD5_ESPERADO = "4aece6c7b16f59a69be4fafec7ad54d9"

# Nome curto de cada canal, sem a unidade entre colchetes.
CANAIS = {
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

# Canais selecionados: taxa real de mudança acima de 0,3 Hz e não redundantes.
NUCLEO = ["rpm", "velocidade", "fluxo_ar", "pressao_adm", "temp_adm", "pedal_d"]

# Sufixos de sessão que descrevem condição de direção ou erro de medição.
# Ficam fora do conjunto de treino.
ESPECIAIS = {"Messfehler", "Vollbremsung", "Glatteis", "Beschleunigung"}

LACUNA_S = 2.0
JANELA_S = 60
PASSO_S = 10


def baixa() -> None:
    """Baixa e extrai o dataset, se ainda não estiver em disco.

    O endpoint entrega um pacote BagIt (tar) contendo um zip com os CSV e um
    manifesto MD5. A integridade do zip interno é conferida contra esse
    manifesto antes da extração.
    """
    if DIR_CSV.exists():
        return

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    if not ARQUIVO.exists():
        print(f"baixando de {URL} ...")
        urllib.request.urlretrieve(URL, ARQUIVO)

    print("extraindo o pacote BagIt ...")
    with tarfile.open(ARQUIVO) as tar:
        tar.extractall(DIR_DADOS, filter="data")

    interno = DIR_DADOS / ZIP_INTERNO
    md5 = hashlib.md5(interno.read_bytes()).hexdigest()
    if md5 != MD5_ESPERADO:
        raise RuntimeError(
            f"MD5 do zip interno não confere.\n"
            f"  esperado: {MD5_ESPERADO}\n"
            f"  obtido  : {md5}"
        )
    print(f"integridade conferida (MD5 {md5})")

    with zipfile.ZipFile(interno) as zf:
        zf.extractall(DIR_DADOS)


def encurta(coluna: str) -> str:
    base = re.sub(r"\s*\[.*\]\s*$", "", coluna).strip()
    return CANAIS.get(base, base)


def carrega(caminho: Path) -> pd.DataFrame:
    """Lê uma sessão e converte a coluna de tempo para timedelta."""
    quadro = pd.read_csv(caminho, encoding="latin-1")
    quadro.columns = [encurta(c) for c in quadro.columns]
    quadro["Time"] = pd.to_timedelta(quadro["Time"])
    return quadro


def segmentos(quadro: pd.DataFrame) -> list[pd.DataFrame]:
    """Corta a sessão nas lacunas e reamostra cada trecho contínuo para 1 Hz.

    A lacuna é operação normal (buffer despejado ao voltar à cobertura), não
    anomalia, e por isso não pode atravessar uma janela.
    """
    intervalo = quadro["Time"].diff().dt.total_seconds()
    quadro = quadro.assign(segmento=(intervalo > LACUNA_S).cumsum())

    saida = []
    for _, bloco in quadro.groupby("segmento"):
        reamostrado = (
            bloco.set_index("Time")[NUCLEO].resample("1s").last().ffill().dropna()
        )
        if len(reamostrado) >= JANELA_S:
            saida.append(reamostrado)
    return saida


def main() -> None:
    baixa()
    arquivos = sorted(DIR_CSV.glob("*.csv"))

    usadas = [c for c in arquivos if c.stem.split("_")[-1] not in ESPECIAIS]
    total_segmentos = total_amostras = total_janelas = 0

    for caminho in usadas:
        for trecho in segmentos(carrega(caminho)):
            total_segmentos += 1
            total_amostras += len(trecho)
            total_janelas += (len(trecho) - JANELA_S) // PASSO_S + 1

    print(f"\nAutomotive OBD-II Dataset — DOI {DOI}")
    print("=" * 60)
    print(f"sessões no arquivo       : {len(arquivos)}")
    print(f"sessões usadas           : {len(usadas)} ({len(ESPECIAIS)} especiais fora)")
    print(f"canais selecionados      : {len(NUCLEO)} -> {', '.join(NUCLEO)}")
    print(f"segmentos contínuos      : {total_segmentos}")
    print(f"amostras a 1 Hz          : {total_amostras:,} ({total_amostras / 3600:.1f} h)")
    print(f"janelas de {JANELA_S}s (passo {PASSO_S}s): {total_janelas:,}")
    print(f"tensor                   : {total_janelas} x {JANELA_S} x {len(NUCLEO)} "
          f"= {total_janelas * JANELA_S * len(NUCLEO) * 4 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
