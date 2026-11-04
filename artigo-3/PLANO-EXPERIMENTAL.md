# Plano experimental — Artigo 3

Fonte única dos valores: `src/config.py`. Este documento explica o porquê;
se divergir do código, vale o código e este arquivo deve ser corrigido.

## Pergunta

Como o tamanho e a sobreposição dos chunks influenciam (i) a recuperação da
evidência correta, (ii) a precisão factual e (iii) a ocorrência de afirmações
não sustentadas em um sistema RAG sobre as aulas 14–23?

## Variáveis

- **Independente:** estratégia de chunking, em duas famílias:
  - estrutural — `slide`: um chunk por slide, a segmentação natural do material
    (linha de base sem parâmetro de tamanho);
  - janela fixa — `c128-o0`, `c128-o20`, `c256-o0`, `c256-o20`, `c512-o0`,
    `c512-o20`: tamanho e sobreposição em tokens do tokenizador do embedding.
- **Fixas entre configurações:** corpus extraído (mesmo manifesto), modelo e
  revisão de embedding, índice FAISS `IndexFlatIP` com vetores normalizados,
  um único `top-k` selecionado em desenvolvimento, gerador, prompt, temperatura
  0 e conjunto de perguntas. Os candidatos de `top-k` são 1, 3 e 5.

O embedding foi congelado como `intfloat/multilingual-e5-base`, revisão
`d128750597153bb5987e10b1c3493a34e5a4502a`. A segmentação usa o tokenizador da
mesma revisão, sem tokens especiais, e o E5 recebe os prefixos `passage: ` para
chunks e `query: ` para perguntas.

## Pipeline

1. `src/corpus.py` — extrai o texto por página com origem e SHA-256, recompõe
   acentos soltos do LaTeX e remove o rodapé repetido do Beamer (#48).
2. Chunking por aula, sem atravessar a fronteira entre aulas (#50).
3. Indexação e busca (#50).
4. Coleta dos rankings de desenvolvimento até a posição 5 e métricas de
   recuperação contra `evidencias_validas` (#51).
5. Seleção de um `top-k` global pelo maior Recall médio em `dev`, com desempate
   pelo menor `k`; o artefato dessa decisão é obrigatório antes de abrir `test`.
6. Geração restrita ao contexto, com abstenção obrigatória (#52).
7. `src/run.py` orquestra a grade e grava um JSON por configuração × pergunta (#53).
8. `src/report.py` consolida e emite as macros do artigo (#54).

Os índices são cacheados por hash do corpus, do fluxo canônico de páginas, do
modelo, da revisão e da configuração. Cada cache registra número de chunks,
tamanho do índice e tempos de chunking, embedding e construção FAISS.

## Dataset de avaliação (#49)

60 itens: 30 factuais, 15 contextuais, 10 multitrecho, 5 não respondíveis.
Split fixado **antes** de qualquer experimento: 15 `dev`, 45 `test`.
O `dev` contém 8/4/2/1 itens por categoria e o `test`, 22/11/8/4.
Cada aula recebe seis itens e o `dev` cobre as dez aulas; a alocação imutável
está em `data/planejamento-dataset.jsonl`.
O `dev` serve para depurar o pipeline e o prompt; o `test` é lido uma única vez,
na execução canônica (7 × 45 = 315 pares).
O contrato está em `data/dataset-schema.json`; as regras de evidência, afirmação
não sustentada, abstenção e revisão cruzada estão em
`docs/dataset/GUIA-ANOTACAO.md`.

## Métricas

| Nível | Métrica | Papel | Justificativa |
| --- | --- | --- | --- |
| Recuperação | Recall@k selecionado | principal | fração das páginas-evidência distintas presentes no contexto do gerador |
| Recuperação | MRR@5, Precision@k (k = 1, 3, 5) | secundárias | primeira posição relevante e proporção de chunks relevantes |
| Recuperação | redundância e latência | diagnóstico | IoU médio dos intervalos de tokens; média, mediana e p95 da busca |
| Resposta | correção, completude, fidelidade | principais da geração | rubrica humana cega à configuração |
| Resposta | afirmações não sustentadas | alucinação | só conta afirmação sem apoio no contexto; paráfrase correta não é alucinação |
| Resposta | abstenção correta | perguntas não respondíveis | distingue "não sabe" de "inventa" |

Falha de recuperação (evidência fora do top-k) e falha de geração (evidência
recuperada, resposta errada) são reportadas separadamente.

Em itens multitrecho, Recall@k é a fração das evidências obrigatórias cobertas;
o relatório também indica quando todas foram recuperadas. MRR@5 vale zero se
nenhuma evidência aparece nas cinco posições. Precision@k usa `k` como
denominador e considera cada chunk relevante individualmente. Perguntas não
respondíveis ficam fora dessas quatro métricas e da seleção de `top-k`, mas
permanecem nas medidas de latência e redundância. Todos os agregados são
produzidos por configuração, no conjunto geral e nas Partes V, VI e VII.

Fluxo operacional:

```sh
python -m src.avaliacao_recuperacao avaliar --split dev
python -m src.avaliacao_recuperacao selecionar-k --avaliacao-dir src/results/retrieval/dev
python -m src.avaliacao_recuperacao avaliar --split test --selecao-top-k src/results/retrieval/dev/top-k-selection.json
python -m src.run --split test --selecao-top-k src/results/retrieval/dev/top-k-selection.json
```

## Reprodutibilidade

- Execução canônica a partir de um SHA destacado, Git limpo, manifestos de corpus
  e dataset conferidos (`run.py` aborta se divergirem).
- `test` exige `top-k-selection.json`, ligado por hash ao split `dev`, ao
  protocolo, ao corpus, aos sete índices e aos rankings que originaram a escolha.
- `config.validar_congelamento()` impede a grade enquanto houver campo `PENDENTE`.
- Ambiente congelado em `requirements-lock.txt`; Colab em `COLAB.md`.
