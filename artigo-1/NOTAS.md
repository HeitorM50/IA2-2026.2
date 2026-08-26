# Artigo 1 — decisões e andamento

**Prazo: 03/09/2026** · vale 20% da nota final · entrega pelo Teams.

## Dupla

Divisão em fatia vertical: cada um leva modelos **e** seções, para que ninguém
fique bloqueado esperando o outro e os dois conheçam o artigo inteiro na
apresentação.

| Integrante | GitHub | Matrícula | Responsabilidade |
| ---------- | ------ | --------- | ---------------- |
| Heitor Macedo | `HeitorM50` | — | pipeline de dados, linha de base, transfer learning · Metodologia, Resumo, Conclusão |
| Gustavo | `guxvr` | — | loop de treino e métricas, CNN do zero, figuras · Introdução, Trabalhos Relacionados, Resultados, `refs.bib` |

A quatro mãos: decisão do tema, Discussão e Limitações, revisão final e entrega.

O andamento fica nas issues do GitHub, no milestone `Artigo 1 — 03/09/2026`.

## Decisões em aberto

- [x] **Rota temática** — **I, comparação entre algoritmos** (decidido em 26/08/2026).
      O escopo é fechado (N modelos, um protocolo) e o resultado sempre existe,
      mesmo que seja "a arquitetura simples ganhou".
- [ ] **Dataset público** — precisa ser público, citável e pequeno o bastante
      para treinar várias vezes até 03/09.
- [ ] **Modelos a comparar** — se Rota I, no mínimo três, incluindo uma linha de
      base simples que não seja rede neural.
- [ ] **Métrica principal** — e a justificativa da escolha, que é critério de correção.
- [ ] **Onde treinar** — máquina local ou Google Colab. Se o dataset for de imagens,
      Colab com GPU economiza muito tempo.

## Registro de decisões

Anotar aqui, com data, cada decisão fechada. Isso vira material da seção de
Metodologia depois.

- **26/08/2026** — repositório e esqueleto LaTeX criados.
- **26/08/2026** — rota temática fechada: **Rota I**, comparação entre algoritmos.
- **26/08/2026** — divisão do trabalho em fatia vertical (tabela acima), com as
  issues abertas no GitHub e milestone vencendo 02/09, um dia antes do prazo real.
- **26/08/2026** — repositório publicado em `github.com/HeitorM50/IA2-2026.2`
  (público), com `guxvr` como colaborador.
- **[pendente]** — tema, conjunto de dados, modelos comparados e métrica principal:
  ver a issue #1. É o bloqueio de tudo; precisa fechar até 27/08.

## Checklist antes de entregar

- [ ] Cabe em 4 páginas com as referências dentro
- [ ] Nenhum texto-guia ou placeholder `[entre colchetes]` sobrou no PDF
- [ ] Dataset citado formalmente em `refs.bib`
- [ ] Todas as citações do texto aparecem nas referências (sem `[?]` no PDF)
- [ ] Métrica justificada explicitamente
- [ ] Seção de limitações escrita a sério, não uma frase protocolar
- [ ] Nomes, matrículas e e-mails dos dois integrantes corretos no `\author{}`
- [ ] Os **dois** subiram o PDF no Teams, cada um na pasta da própria matrícula
