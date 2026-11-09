# Recuperação e revisão dos resultados do Artigo 3

Os agregados versionados não bastam para refazer denominadores, diferenças
pareadas e avaliação humana. O pacote necessário deve conter
`retrieval/test/rankings.jsonl`, os 315 JSONs em `raw/test/`, a amostra, a chave
cega e as avaliações individuais. Esses arquivos contêm texto protegido dos
slides e continuam fora do Git.

## 1. Recuperar e validar

No Colab, monte o Drive e procure `MyDrive/IA2/art3-resultados`. Antes de copiar
ou analisar qualquer arquivo, execute o validador somente leitura da revisão:

```bash
python -m src.validar_artefatos \
  /content/drive/MyDrive/IA2/art3-resultados
```

O comando exige o commit experimental
`4f347d62c167bf152ff32c6e5c54f570dc705c38`, sete configurações, 45 perguntas,
315 pares e `top-k = 5`; também confere os hashes encadeados nos manifestos.

Copie o pacote validado para uma pasta de trabalho nova. Não altere nem
sobrescreva `art3-resultados`.

## 2. Reconstruir se o pacote não existir

Faça a reconstrução a partir do commit experimental e grave em uma pasta com
data, por exemplo `MyDrive/IA2/art3-resultados-reconstruidos-AAAA-MM-DD`. Use uma
T4, as revisões congeladas do E5 e do Qwen e os comandos das etapas 1 a 8 de
`COLAB.md`. Ao final, copie `src/results/` e `data/avaliacao/` para a nova pasta
e rode o validador acima. Essa execução não substitui a original: o manifesto
deve permitir distingui-las.

## 3. Avaliação humana cega

Regere deterministicamente a amostra de dez perguntas e sete configurações:

```bash
python -m src.analise_geracao amostra --split test --n 10 \
  --resultados CAMINHO/results/raw --destino CAMINHO/avaliacao
```

Gustavo e Heitor preenchem cópias independentes de `planilha-cega.csv`, sem
abrir `chave.csv`. Calcule a concordância antes da reunião:

```bash
python -m src.avaliacao_humana concordancia \
  CAMINHO/avaliacao/gustavo.csv CAMINHO/avaliacao/heitor.csv
python -m src.avaliacao_humana preparar-consenso \
  CAMINHO/avaliacao/planilha-cega.csv \
  CAMINHO/avaliacao/gustavo.csv CAMINHO/avaliacao/heitor.csv \
  CAMINHO/avaliacao/consenso.csv
```

Preencha apenas as colunas `consenso_*` das divergências, ainda sem abrir a
chave. Depois gere os agregados, a tabela e as figuras selecionando
explicitamente a avaliação humana:

```bash
python -m src.report \
  --resultados-dir CAMINHO/results \
  --avaliacao-dir CAMINHO/avaliacao \
  --avaliacao-principal humana \
  --avaliador CAMINHO/avaliacao/gustavo.csv \
  --avaliador CAMINHO/avaliacao/heitor.csv \
  --consenso CAMINHO/avaliacao/consenso.csv
```

O relatório mantém `f1_tokens_v1` e a variante lexical apenas no CSV de apoio.
O PDF usa correção, completude, fidelidade e afirmações não sustentadas do
consenso humano. Rankings, respostas, contextos e planilhas não devem ser
adicionados ao Git.
