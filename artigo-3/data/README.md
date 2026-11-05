# Dados do Artigo 3

O corpus integral continua fora do Git. Metadados e contratos sem conteúdo dos
slides são versionados para tornar a anotação auditável.

| Caminho | Conteúdo | Versionado |
| --- | --- | --- |
| `corpus/` | os dez PDFs das aulas 14–23 (ou defina `IA2_CORPUS_DIR`) | não — direitos autorais em aberto (#46) |
| `extraido/aula-NN.jsonl` | texto por página com origem e SHA-256 do PDF | não — é o texto integral |
| `manifesto-corpus.json` | hashes, número de páginas e páginas suspeitas | sim |
| `dataset-schema.json` | contrato dos dez campos públicos | sim |
| `planejamento-dataset.jsonl` | IDs, splits, categorias e revisão cruzada | sim |
| `perguntas.jsonl` | perguntas, respostas e evidências (#49) | privado até a publicação |

Para gerar a extração:

```sh
cd artigo-3
IA2_CORPUS_DIR=/caminho/para/contexto/art3 .venv/bin/python -m src.corpus
```

Validar o planejamento e, depois da anotação, o dataset:

```sh
cd artigo-3
python -m src.planejamento_dataset
python -m src.dataset --dataset data/perguntas.jsonl
```
