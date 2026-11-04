# Execução no Colab — Artigo 3

O notebook só chama código versionado. Nenhum parâmetro experimental é definido
no notebook: tudo vem de `src/config.py`.

```python
# 1. Código a partir de um SHA fixo, nunca da branch (#53)
!git clone https://github.com/HeitorM50/IA2-2026.2.git
%cd IA2-2026.2
!git checkout <SHA-DO-COMMIT-CONGELADO>
%cd artigo-3

# 2. Ambiente exato
!pip install -q -r requirements-lock.txt

# 3. Corpus a partir do Drive (os PDFs não estão no repositório)
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['IA2_CORPUS_DIR'] = '/content/drive/MyDrive/IA2/art3'

# 4. Verificações antes da grade
!python -m pytest -q
!python -m src.corpus
!git diff --exit-code data/manifesto-corpus.json   # o corpus é o mesmo do commit?

# 5. Ensaio curto no dev, depois a grade canônica (6 configs × 45 perguntas)
!python -m src.run --split dev --limite 3 --saida src/results/raw/ensaio
!python -m src.run --split test
# Se o runtime cair: !python -m src.run --split test --resume
```

Ao final, baixar `src/results/` inteiro, incluindo `environment.json`,
`pip-freeze.txt` e `run.log`. Os JSON por pergunta ficam fora do Git
(contêm texto dos slides) até a decisão de licença; guardar no Drive.

O `run.py` recusa a grade `test` se: algum campo do `config.py` estiver `PENDENTE`,
o Git estiver sujo, o HEAD não estiver destacado, o gerador diferir do congelado
ou o texto extraído não bater com `data/manifesto-corpus.json`.
