# IA2-2026.2 — Inteligência Artificial II (FGA0261)

Repositório dos artigos científicos da disciplina **FGA0261 — Tópicos Especiais em
Eletrônica (Inteligência Artificial II)**, UnB/FCTE, turma T02, 2º/2026.
Professor: Fabiano Araujo Soares.
Site da disciplina: <https://www.fabianosoares.eng.br/fga0261-inteligência-artificial-ii>

Trabalho em **dupla**. Cada artigo é escrito em LaTeX no formato de congresso do IEEE.

---

## 1. Como a disciplina é avaliada

A nota é 100% composta por artigos e pela apresentação final.

| Entrega       | Data       | Peso |
| ------------- | ---------- | ---- |
| Artigo 1      | 03/09/2026 | 20%  |
| Artigo 2      | 09/10/2026 | 25%  |
| Artigo 3      | 05/11/2026 | 25%  |
| Apresentações | a partir de 08/12/2026 | 30% |

Frequência mínima de 75%, sem compensação de falta.

## 2. Restrições duras de cada artigo

Estas vêm do item 6 do plano de ensino (`docs/plano-de-ensino.pdf`). Furar qualquer
uma custa nota antes de o conteúdo ser lido.

- **Máximo 4 páginas**, formato de artigo de congresso IEEE (duas colunas,
  classe `IEEEtran` com opção `conference`). O limite inclui resumo, figuras,
  tabelas e referências.
- **Temáticas distintas** entre os três artigos. A exigência é de tema distinto,
  não de dataset distinto — reaproveitar um modelo já treinado sob outra pergunta
  de pesquisa é legítimo.
- **Repositório público de dados** obrigatório: de pesquisa, didático ou gerado
  artificialmente. O dataset precisa ser citado formalmente nas referências.
- O artigo precisa conter: proposta de projeto a partir dos dados, **metodologia**,
  **resultados** e **discussão crítica incluindo as limitações**.
- **Entrega apenas pelo Microsoft Teams**, na pasta `Avaliações` -> subpasta da
  atividade -> uma pasta com a matrícula. **Cada aluno da dupla cria a própria
  pasta e sobe o material** — não basta um entregar pelos dois.

## 3. Critérios de correção declarados

1. Uso adequado de linguagem científica.
2. Clareza na descrição da metodologia (tem que ser reproduzível pela leitura).
3. Adequação das métricas ao problema proposto (justificar a escolha; acurácia em
   base desbalanceada é o erro clássico).
4. Profundidade da análise na discussão e conclusão.

## 4. As quatro rotas temáticas aceitas

| Rota | Tipo de artigo | Seção que carrega o peso |
| ---- | -------------- | ------------------------ |
| I    | Comparação entre algoritmos para o mesmo problema | Resultados (tabela comparativa, protocolo idêntico entre modelos) |
| II   | Aplicação de algoritmos a um problema proposto | Metodologia (justificar arquitetura, perda, pré-processamento; exige baseline) |
| III  | Análise de explicabilidade (XAI) de um modelo treinado | Discussão (LIME, SHAP, Grad-CAM; o achado forte é o modelo acertar pelo motivo errado) |
| IV   | Estudo de viés e equidade (fairness) | Resultados desagregados por subgrupo + métricas de fairness e alguma mitigação |

Trilha planejada, casada com a ordem da ementa:

- **Artigo 1 (03/09)** — conteúdo disponível: fundamentos de deep learning, MLP,
  regularização, CNNs, transfer learning. Rota I ou II.
- **Artigo 2 (09/10)** — RNN/LSTM, autoencoders, VAE/GAN, Transformers, LLMs, PLN.
  Rota II ou I.
- **Artigo 3 (05/11)** — deep RL, XAI, fairness. Rota III ou IV, idealmente
  reaproveitando o modelo treinado no Artigo 1 ou 2.

## 5. Orçamento de páginas (IEEEtran conference, 4 páginas)

Quatro páginas em duas colunas dão cerca de 3.500-4.000 palavras **com** figuras e
referências dentro. Alvo por seção:

| Seção | Espaço | Conteúdo mínimo |
| ----- | ------ | --------------- |
| Title / Authors / Abstract / Index Terms | ~0,3 pág. | Abstract de 150-200 palavras já com o número principal do resultado |
| I. Introduction | ~0,6 pág. | Contexto, lacuna, pergunta de pesquisa, frase explícita de contribuição |
| II. Related Work | ~0,5 pág. | 5-8 trabalhos agrupados por abordagem, fechando com o diferencial deste |
| III. Methodology | ~1,0 pág. | Dataset e origem, pré-processamento, arquitetura, hiperparâmetros, split, seed, métricas e justificativa |
| IV. Results | ~0,8 pág. | Uma tabela principal e uma figura que responda a uma pergunta |
| V. Discussion & Limitations | ~0,5 pág. | Interpretação + limitações honestas (cobrado nominalmente pelo plano) |
| VI. Conclusion | ~0,2 pág. | 3-4 frases, sem resultado novo |
| References | ~0,3 pág. | 10-15 itens em estilo IEEE, incluindo o dataset |

Corta espaço sem doer: teoria de livro-texto (cite Goodfellow em vez de derivar),
prosa que descreve tabela, figura decorativa, parágrafo sobre biblioteca.

## 6. Estrutura do repositório

```
IA2-2026.2/
├── CLAUDE.md              este arquivo — contexto da disciplina
├── docs/
│   ├── plano-de-ensino.pdf
│   └── guia-artigos.md    guia detalhado (rotas, datasets, checklist)
├── template-ieee/         originais do professor, NÃO editar
└── artigo-1/
    ├── NOTAS.md           decisões do artigo: tema, dataset, divisão de trabalho
    ├── paper/             o artigo em LaTeX
    │   ├── main.tex
    │   ├── refs.bib
    │   ├── figs/          figuras geradas pelos experimentos entram aqui
    │   ├── IEEEtran.cls   classe oficial IEEE, NÃO editar
    │   └── IEEEtran.bst   estilo de bibliografia IEEE, NÃO editar
    └── src/               código dos experimentos (PyTorch)
```

Os artigos 2 e 3 vão ganhar pastas irmãs (`artigo-2/`, `artigo-3/`) com a mesma forma.

## 7. Convenções obrigatórias neste repositório

### LaTeX

- **Uma frase por linha** no `.tex`. Isso não muda nada no PDF (o LaTeX junta as
  linhas) e faz o `git diff` e o merge ficarem legíveis quando os dois estiverem
  editando o mesmo parágrafo. É a convenção mais importante aqui.
- Não quebrar linha no meio de uma frase para "alinhar" o código.
- Não editar `IEEEtran.cls` nem `IEEEtran.bst`.
- Nada de `\vspace` negativo ou `\fontsize` para caber nas 4 páginas — corte texto.
- Figuras em `paper/figs/`, sempre com `\label` e referenciadas por `\ref` no texto.

### Build

Rodar de dentro de `artigo-1/paper/`:

```sh
make          # latexmk -pdf: compila main.pdf resolvendo citações e referências
make watch    # recompila a cada salvamento
make clean    # remove os arquivos auxiliares
```

Dependências no Arch:

```sh
sudo pacman -S texlive-basic texlive-latex texlive-latexrecommended \
               texlive-fontsrecommended texlive-bibtexextra texlive-binextra
```

No VS Code, a extensão `james-yu.latex-workshop` compila com o mesmo `latexmk`
(já configurada em `.vscode/settings.json`).

### Git

- Trabalho em dupla: cada um em sua branch (`feat/metodologia`, `feat/resultados`),
  merge por pull request. Nunca os dois na `main` ao mesmo tempo.
- Commits em português, no imperativo: `escreve secao de metodologia`,
  `adiciona tabela comparativa`.
- Artefatos de build (`*.aux`, `*.log`, `*.bbl`, `main.pdf`) estão no `.gitignore`.
- **Nunca adicionar Claude como co-autor nos commits.**

### Escrita

- Texto em português, voz impessoal ("foram avaliados", não "avaliamos").
- Todo termo técnico definido na primeira ocorrência, com a sigla entre parênteses.
- Nenhuma afirmação sem referência ou sem dado no próprio artigo que a sustente.
- Números de resultado sempre com a métrica nomeada e, quando houver mais de uma
  execução, com desvio padrão.

## 8. Estado atual

- [x] Repositório e esqueleto LaTeX do Artigo 1 montados
- [x] Definir rota temática do Artigo 1 (I ou II)
- [x] Escolher e citar o dataset público
- [ ] Implementar os experimentos em `artigo-1/src/`
- [ ] Escrever o artigo
- [ ] Revisar contra os 4 critérios de correção e o limite de 4 páginas
- [ ] Subir no Teams (os **dois** integrantes, cada um na pasta da própria matrícula)

Decisões em aberto estão registradas em `artigo-1/NOTAS.md`.
