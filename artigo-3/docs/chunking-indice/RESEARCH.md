# Pesquisa — chunking e índice vetorial

## Problema

As seis janelas fixas devem variar somente tamanho e sobreposição, recebendo o
mesmo fluxo de páginas. A baseline `slide` permanece como sétima configuração.
Cada chunk precisa conservar sua origem e nunca atravessar aulas.

## Decisões

- tokenização com o tokenizador do próprio modelo de embeddings;
- `add_special_tokens=False` durante a segmentação;
- sobreposição inteira calculada por `round(tamanho * proporcao)`;
- `intfloat/multilingual-e5-base` congelado por commit do Hugging Face;
- prefixos `passage: ` e `query: `, conforme o treinamento do E5;
- vetores `float32` normalizados em L2 antes de adicionar e consultar;
- `faiss.IndexFlatIP`, que produz busca exata por cosseno após normalização;
- cache identificado pelo corpus, entrada canônica, modelo, revisão e chunker.

## Riscos

- pesos do modelo têm aproximadamente 1,1 GB e não devem ser baixados pela
  suíte unitária;
- os textos extraídos não são versionados por direitos autorais;
- artefatos FAISS podem variar entre versões da biblioteca, portanto o ambiente
  canônico ainda deve atualizar `requirements-lock.txt` após a instalação.

## Referências

- documentação oficial do FAISS sobre produto interno e cosseno;
- documentação oficial do Sentence Transformers sobre embeddings normalizados;
- card oficial do modelo `intfloat/multilingual-e5-base`.
