# `artigo-2/src/` — guia de implementação

Ordem sugerida. Cada arquivo tem uma responsabilidade só.
Antes de escrever qualquer coisa aqui, acionar a skill `experimento-pytorch`.

| Arquivo | Responsabilidade | Status |
| ------- | ---------------- | ------ |
| `verifica_dataset.py` | baixa, confere o MD5 oficial e reproduz os números da verificação registrada no `PLANO-EXPERIMENTAL.md` | **pronto** |
| `config.py` | fonte **única** de seeds, janela, passo, hiperparâmetros e caminhos. Nenhuma constante duplicada em outro arquivo | **pronto** (#27) |
| `data.py` | download do dataset, reamostragem para 1 Hz, segmentação nas lacunas, janelamento e split por sessão. Saída em NumPy, sem depender de PyTorch | **pronto** (#27) |
| `faults.py` | as cinco injeções de falha em unidades físicas, mais `violates_range` (o `if` do firmware) e `build_evaluation_set` | **pronto** (#28) |
| `models/base.py` | interface comum: `fit`, `score_samples`, `score_windows`, `predict`, `n_parameters` | **pronto** (#31) |
| `models/baseline.py` | limiar 3σ por canal | **pronto** (#31) |
| `models/pca.py` | PCA como autoencoder linear, erro de reconstrução | **pronto** (#32) |
| `models/autoencoder.py` | autoencoder LSTM sequência-a-sequência | **pronto** (#33) |
| `train.py` | loop comum de treino (MSE de reconstrução) e calibração do limiar na validação normal | **pronto** (#30) |
| `metrics.py` | AUC-PR, F1/precisão/revocação no limiar, latência de detecção, desagregação por classe de falha | **pronto** (#29) |
| `run.py` | orquestra modelo × seed e grava um JSON por execução em `results/` | **pronto** (#30) |
| `report.py` | consolida os JSON em `resumo.csv`, emite as macros LaTeX e a figura de duas agregações | **pronto** (#35) |

## Ambiente

Cada artigo tem o próprio venv, como no Artigo 1. Rodar de dentro de `artigo-2/`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Os testes precisam ser invocados com `python -m pytest` a partir de `artigo-2/`,
e não com o executável `pytest`: é o `-m` que coloca o diretório atual no
`sys.path` e torna `from src import ...` importável.

```sh
.venv/bin/python -m pytest tests -m "not integration"   # rápido, sem dataset
.venv/bin/python -m pytest tests                         # exige o dataset baixado
```

## Contrato do tensor (fechado na #27)

```
X        float32  (n_janelas, 60, 6)
session           (n_janelas,)  nome da sessão de origem
canais em ordem: rpm, velocidade, fluxo_ar, pressao_adm, temp_adm, pedal_d
```

`data.load_dataset()` devolve um `Dataset` com `train`, `val`, `test` (cada um um
`Split` com `.X` e `.session`) mais `mean` e `std` ajustados **somente no treino**.

## Invariantes que precisam virar teste, não confiança

1. Nenhum modelo vê janela anômala durante o treino — asserção explícita.
2. A normalização usa estatísticas calculadas somente no split de treino.
3. O limiar é calibrado só na validação normal; o teste é consultado uma vez.
4. O janelamento usa `frame_time`, nunca `received_at`.
5. Rodar duas vezes com a mesma seed produz o mesmo JSON.
6. A falha de ganho injetada **não** dispara nenhuma checagem de faixa — é o que
   sustenta o argumento central do artigo.

## Saída

Um JSON por par modelo × seed em `results/`, com o mesmo formato do Artigo 1:
identificação do modelo, seed, métricas globais, métricas por classe de falha,
tempo de treino, contagem de parâmetros e ambiente de execução.
