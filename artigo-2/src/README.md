# `artigo-2/src/` — guia de implementação

Ordem sugerida. Cada arquivo tem uma responsabilidade só.
Antes de escrever qualquer coisa aqui, acionar a skill `experimento-pytorch`.

| Arquivo | Responsabilidade | Status |
| ------- | ---------------- | ------ |
| `verifica_dataset.py` | baixa, confere o MD5 oficial e reproduz os números da verificação registrada no `PLANO-EXPERIMENTAL.md` | **pronto** |
| `config.py` | fonte **única** de seeds, janela, passo, hiperparâmetros e caminhos. Nenhuma constante duplicada em outro arquivo | a criar |
| `data.py` | download do dataset, *loader* para o formato canônico `(tempo, sinal, valor)`, reamostragem, janelamento por `frame_time`, split por sessão | a criar |
| `faults.py` | as cinco injeções de falha: ganho, travado, deriva, pico, lacuna. Cada função recebe janela normal e devolve janela estragada + máscara do que foi estragado | a criar |
| `models/baseline.py` | limiar 3σ por canal | a criar |
| `models/pca.py` | PCA como autoencoder linear, erro de reconstrução | a criar |
| `models/autoencoder.py` | autoencoder LSTM sequência-a-sequência | a criar |
| `train.py` | loop comum de treino (MSE de reconstrução) e calibração do limiar na validação normal | adaptar de `artigo-1/src/train.py` |
| `metrics.py` | AUC-PR, F1/precisão/revocação no limiar, latência de detecção, desagregação por classe de falha | adaptar de `artigo-1/src/metrics.py` |
| `run.py` | orquestra modelo × seed e grava um JSON por execução em `results/` | adaptar de `artigo-1/src/run.py` |
| `report.py` | consolida os JSON em `resumo.csv`, gera a tabela LaTeX e a figura | adaptar de `artigo-1/src/report.py` |

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
