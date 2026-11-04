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
  `top_k = 5`, gerador, prompt, temperatura 0, conjunto de perguntas.

## Pipeline

1. `src/corpus.py` — extrai o texto por página com origem e SHA-256, recompõe
   acentos soltos do LaTeX e remove o rodapé repetido do Beamer (#48).
2. Chunking por aula, sem atravessar a fronteira entre aulas (#50).
3. Indexação e busca (#50).
4. Métricas de recuperação contra `evidencias_validas` (#51).
5. Geração restrita ao contexto, com abstenção obrigatória (#52).
6. `src/run.py` orquestra a grade e grava um JSON por configuração × pergunta (#53).
7. `src/report.py` consolida e emite as macros do artigo (#54).

## Dataset de avaliação (#49)

60 itens: 30 factuais, 15 contextuais, 10 multitrecho, 5 não respondíveis.
Split fixado **antes** de qualquer experimento: 15 `dev`, 45 `test`.
O `dev` serve para depurar o pipeline e o prompt; o `test` é lido uma única vez,
na execução canônica (7 × 45 = 315 pares).

## Métricas

| Nível | Métrica | Papel | Justificativa |
| --- | --- | --- | --- |
| Recuperação | Recall@5 | principal | mede se a evidência chega ao gerador; é o que limita a resposta |
| Recuperação | MRR, Precision@k (k = 1, 3, 5) | secundárias | ordem do acerto e diluição do contexto por chunks grandes |
| Resposta | correção, completude, fidelidade | principais da geração | rubrica humana cega à configuração |
| Resposta | afirmações não sustentadas | alucinação | só conta afirmação sem apoio no contexto; paráfrase correta não é alucinação |
| Resposta | abstenção correta | perguntas não respondíveis | distingue "não sabe" de "inventa" |

Falha de recuperação (evidência fora do top-k) e falha de geração (evidência
recuperada, resposta errada) são reportadas separadamente.

## Reprodutibilidade

- Execução canônica a partir de um SHA destacado, Git limpo, manifestos de corpus
  e dataset conferidos (`run.py` aborta se divergirem).
- `config.validar_congelamento()` impede a grade enquanto houver campo `PENDENTE`.
- Ambiente congelado em `requirements-lock.txt`; Colab em `COLAB.md`.
