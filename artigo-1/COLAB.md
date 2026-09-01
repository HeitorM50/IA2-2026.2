# Execução canônica da Issue #7 no Google Colab

Este roteiro usa o Colab somente como executor do código versionado. Os resultados
são gravados no Google Drive depois de cada par modelo × seed, portanto uma
desconexão não apaga execuções já concluídas.

## 1. Criar o runtime

No Colab, selecione **Ambiente de execução → Alterar tipo de ambiente de
execução → GPU**. Depois execute:

```python
from google.colab import drive

drive.mount("/content/drive")
```

## 2. Fixar o código

Cole no valor de `RUN_COMMIT` o SHA informado na entrega da branch. Não use o nome
da branch: o SHA impede que uma atualização posterior altere o experimento no
meio da grade.

```python
import os
import subprocess
from pathlib import Path

RUN_COMMIT = "COLE_AQUI_O_SHA_CONGELADO"
RUN_ID = "canonical-1"
REPO_DIR = Path("/content/IA2-2026.2")

if REPO_DIR.exists():
    subprocess.run(["git", "fetch", "origin"], cwd=REPO_DIR, check=True)
else:
    subprocess.run(
        ["git", "clone", "https://github.com/HeitorM50/IA2-2026.2.git", str(REPO_DIR)],
        check=True,
    )
subprocess.run(["git", "checkout", "--detach", RUN_COMMIT], cwd=REPO_DIR, check=True)

RUN_ROOT = (
    Path("/content/drive/MyDrive/IA2-2026.2/issue-7")
    / RUN_COMMIT
    / RUN_ID
)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
(RUN_ROOT / "results").mkdir(exist_ok=True)

os.environ["REPO_DIR"] = str(REPO_DIR)
os.environ["RUN_ROOT"] = str(RUN_ROOT)
print("Código:", RUN_COMMIT)
print("Resultados:", RUN_ROOT)
```

## 3. Instalar e testar

```python
import os
import subprocess

article_dir = os.path.join(os.environ["REPO_DIR"], "artigo-1")
subprocess.run(
    ["python", "-m", "pip", "install", "-r", "requirements.txt"],
    cwd=article_dir,
    check=True,
)
subprocess.run(["python", "-m", "pytest"], cwd=article_dir, check=True)

status = subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=os.environ["REPO_DIR"],
    check=True,
    capture_output=True,
    text=True,
).stdout
assert not status, f"O checkout precisa estar limpo antes do treino:\n{status}"
```

## 4. Conferir e fixar o ambiente

Esta célula cria `environment.json` na primeira sessão. Em uma retomada, ela
aborta se o Colab entregar outra GPU ou versões diferentes. Se isso acontecer,
não misture os resultados: altere `RUN_ID` para `canonical-2` e refaça as nove
execuções no novo diretório.

```python
import json
import os
import platform
from pathlib import Path

import torch
import torchvision

assert torch.cuda.is_available(), "Ative um runtime com GPU antes de continuar."
signature = {
    "device_name": torch.cuda.get_device_name(0),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "cuda_version": torch.version.cuda,
}

manifest = Path(os.environ["RUN_ROOT"]) / "environment.json"
if manifest.exists():
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    assert previous == signature, (
        f"Ambiente diferente:\nantes={previous}\nagora={signature}"
    )
else:
    manifest.write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
print(signature)
```

## 5. Preparar o BloodMNIST

O cache fica no Drive para evitar um novo download após desconexões. Esta célula
não altera arquivos versionados.

```python
import os
import shutil
from pathlib import Path

repo = Path(os.environ["REPO_DIR"])
run_root = Path(os.environ["RUN_ROOT"])
data_path = repo / "artigo-1/src/data/bloodmnist_64.npz"
cache_path = run_root.parent / "cache/bloodmnist_64.npz"
data_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.parent.mkdir(parents=True, exist_ok=True)

if cache_path.exists():
    shutil.copy2(cache_path, data_path)
else:
    os.chdir(repo / "artigo-1")
    from src.data import compute_train_stats

    print("Estatísticas do treino:", compute_train_stats())
    shutil.copy2(data_path, cache_path)
print("Dataset pronto:", data_path)
```

## 6. Executar os nove pares

Execute a célula abaixo. O comando percorre `logreg`, `cnn` e `resnet18`, nessa
ordem, com as seeds 42, 1337 e 2026. Não acrescente `--quick` nem `--overwrite`.

```bash
%%bash
set -euo pipefail
cd "$REPO_DIR/artigo-1"
python -m src.run \
  --models logreg cnn resnet18 \
  --seeds 42 1337 2026 \
  --output-dir "$RUN_ROOT/results" \
  --resume 2>&1 | tee -a "$RUN_ROOT/training.log"
```

Se o Colab desconectar, repita desde a etapa 1 e use o mesmo `RUN_COMMIT`. Se a
etapa 4 aceitar o ambiente, execute novamente esta célula: `--resume` pula JSONs
válidos e refaz somente o par que estava incompleto.

## 7. Validar e empacotar

```bash
%%bash
set -euo pipefail
cd "$REPO_DIR/artigo-1"
python -m src.report \
  --results-dir "$RUN_ROOT/results" \
  --summary "$RUN_ROOT/resumo.csv" \
  --figure "$RUN_ROOT/confusao-melhor-modelo.pdf"
```

```python
import os
import shutil
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
archive = shutil.make_archive(
    str(run_root / "resultados-issue-7"),
    "zip",
    root_dir=run_root,
    base_dir="results",
)
print("Envie este arquivo para consolidação:", archive)
```

O ZIP contém somente os nove JSONs. `resumo.csv` e a figura serão regenerados no
repositório a partir deles, garantindo que os artefatos versionados sejam
reproduzíveis.
