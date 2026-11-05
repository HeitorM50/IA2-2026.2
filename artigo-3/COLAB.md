# Execução no Colab — Artigo 3

O notebook só chama código versionado. Nenhum parâmetro experimental é definido
no notebook: tudo vem de `src/config.py`. Runtime: **GPU T4**.

## Pré-requisitos (antes de abrir o Colab)

- `src/config.py` sem nenhum `PENDENTE`, **inclusive `dataset.doi`**. O DOI entra
  no hash do protocolo; preenchê-lo depois da seleção de `k` invalida a seleção.
- Commit com o config final mergeado na `main`; anote o SHA.
- No Drive: os PDFs em `IA2/art3/`. O dataset (`data/perguntas.jsonl`) já vem
  com o clone; o manifesto do `run.py` grava o hash dele.

## Notebook

O mesmo roteiro está em `colab.ipynb`, pronto para abrir no Colab.

```python
# 1. Código a partir de um SHA fixo, nunca da branch (#53)
!git clone https://github.com/HeitorM50/IA2-2026.2.git
%cd IA2-2026.2
!git checkout <SHA-DO-COMMIT-CONGELADO>
%cd artigo-3

# 2. Ambiente: faixas do requirements.txt sobre o torch do Colab.
#    A versão exata usada fica gravada em src/results/raw/<split>/pip-freeze.txt.
!pip install -q -r requirements.txt

# 3. Corpus a partir do Drive (os PDFs não estão no repositório)
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['IA2_CORPUS_DIR'] = '/content/drive/MyDrive/IA2/art3'

# 4. Verificações antes de qualquer execução
!python -m pytest -q                                 # com faiss: nenhum teste pulado
!python -m src.corpus
!git diff --exit-code data/manifesto-corpus.json     # o corpus é o mesmo do commit?

# 5. Recuperação no dev e seleção de k (só dev; o test não é tocado aqui).
#    Constrói os 7 índices em cache/ (fora do Git) e grava os rankings.
!python -m src.avaliacao_recuperacao avaliar --split dev
!python -m src.avaliacao_recuperacao selecionar-k --avaliacao-dir src/results/retrieval/dev

# 6. Ensaio curto de geração no dev: conferir se o 3B cita [n] e se abstém.
!python -m src.run --split dev --limite 3 --saida src/results/raw/ensaio \
    --selecao-top-k src/results/retrieval/dev/top-k-selection.json

# 7. Recuperação no test e grade canônica (7 configs × 45 perguntas = 315 pares)
SEL = 'src/results/retrieval/dev/top-k-selection.json'
!python -m src.avaliacao_recuperacao avaliar --split test --selecao-top-k {SEL}
!python -m src.run --split test --selecao-top-k {SEL}
# Se o runtime cair: !python -m src.run --split test --selecao-top-k {SEL} --resume

# 8. Métricas automáticas da geração e amostra para a avaliação humana
!python -m src.analise_geracao automaticas --split test
!python -m src.analise_geracao amostra --split test --n 10
```

O passo 6 não exige `PENDENTE` resolvido nem Git limpo: é o lugar de ver
problema de prompt antes de gastar a grade. Se o ensaio mudar código, um commit
novo precisa ser feito e o notebook reiniciado a partir do passo 1 com o SHA novo.

## Depois

Baixar `src/results/` inteiro e guardar no Drive: `environment.json`,
`pip-freeze.txt`, `run.log`, `run-manifest.json`, `summary.csv`,
`top-k-selection.json`, os JSON brutos e `data/avaliacao/`. Os JSON por pergunta
e o `rankings.jsonl` ficam fora do Git porque contêm texto dos slides.

O `run.py` recusa a grade `test` se: algum campo do `config.py` estiver `PENDENTE`,
o Git estiver sujo, o HEAD não estiver destacado, o gerador diferir do congelado,
faltar `--selecao-top-k` produzida no dev ou o texto extraído não bater com
`data/manifesto-corpus.json`.
