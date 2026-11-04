# Dados do Artigo 3

Nada aqui é versionado além deste arquivo e do `manifesto-corpus.json`.

| Caminho | Conteúdo | Versionado |
| --- | --- | --- |
| `corpus/` | os dez PDFs das aulas 14–23 (ou defina `IA2_CORPUS_DIR`) | não — direitos autorais em aberto (#46) |
| `extraido/aula-NN.jsonl` | texto por página com origem e SHA-256 do PDF | não — é o texto integral |
| `manifesto-corpus.json` | hashes, número de páginas e páginas suspeitas | sim |
| `perguntas.jsonl` | dataset de avaliação (#49) | só depois de publicado com DOI |

Para gerar a extração:

```sh
cd artigo-3
IA2_CORPUS_DIR=/caminho/para/contexto/art3 .venv/bin/python -m src.corpus
```
