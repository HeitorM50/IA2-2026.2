# Artigo 3 — notas e decisões

Prazo: **05/11/2026**. Entrega pelo Teams, cada autor na pasta da própria matrícula.

## Tema

Comparação de estratégias de *chunking* em um sistema RAG sobre as aulas 14–23
(Partes V, VI e VII) de Inteligência Artificial II.

> Como o tamanho e a sobreposição dos chunks influenciam a recuperação de
> evidências, a precisão factual e a ocorrência de alucinações em um sistema RAG
> aplicado aos conteúdos das Partes V, VI e VII de Inteligência Artificial II?

## Divisão de trabalho

| Issue | Escopo | Responsável |
| --- | --- | --- |
| #46 | validar tema, rota, corpus e protocolo | Heitor |
| #47 | esqueleto reproduzível | Heitor |
| #48 | extração do corpus | Heitor |
| #49 | dataset de perguntas e evidências | Gustavo |
| #50 | tokenização, chunking, FAISS | Gustavo |
| #51 | métricas de recuperação | Gustavo |
| #52 | geração, fidelidade e alucinação | Heitor |
| #53 | `run.py` e execução canônica | Heitor |
| #54 | resultados, macros, tabela e figura | Gustavo |
| #55 | escrita e revisão do artigo | Gustavo |

## Decisões (#46)

Status em 04/11/2026. "Proposto" = decidido pela dupla, aguardando confirmação.

| Decisão | Status | Valor |
| --- | --- | --- |
| Rota temática | **aberta — confirmar com o professor** | Rota I (comparação de 6 configurações) ou adaptação para Rota III |
| Temática distinta dos artigos 1 e 2 | ok | A1: classificação de imagens; A2: detecção de anomalia em séries; A3: recuperação de texto |
| Corpus | congelado | aulas 14–23, Partes V–VII, dez PDFs |
| Redistribuição dos PDFs | **aberta — confirmar com o professor** | até lá, PDFs e texto extraído fora do Git |
| Configurações | congelado | 128/256/512 tokens × 0%/20% de sobreposição (`src/config.py`) |
| Métrica principal de recuperação | proposto | Recall@5 (a evidência certa precisa chegar ao gerador); MRR e Precision@k secundárias |
| Métricas da resposta | proposto | correção, completude, fidelidade, afirmações não sustentadas, abstenção correta |
| Gerador | **aberta** | modelo aberto local (reprodutível) vs. API versionada |
| Publicação do dataset | proposto | Zenodo com DOI, licença CC BY 4.0, só perguntas/respostas/referência à página — sem texto integral dos slides |

### Riscos

- **Direitos autorais:** os slides são material do professor. Publicar só
  identificadores (aula, página) e trechos curtos de evidência, nunca o texto
  integral extraído, até haver autorização.
- **Rota:** o plano da disciplina prevê Rota III ou IV para o Artigo 3. Se o
  professor não aceitar Rota I, a adaptação mais barata é tratar a análise de
  atribuição do RAG (qual trecho sustentou cada afirmação) como explicabilidade.
- **Prazo:** com entrega em 05/11, a avaliação humana cega (#52) pode precisar
  ser reduzida a uma amostra; se acontecer, vira limitação declarada.

## Registro

- 04/11/2026 — esqueleto do `artigo-3/` criado; extração do corpus implementada
  e testada com PDFs sintéticos. Os PDFs reais (`contexto/art3/`) ainda não estão
  na máquina do Heitor; o catálogo aula→(parte, título) em `src/config.py` está vazio.
