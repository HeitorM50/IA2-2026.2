# IA2 2026.2 — Artigos de Inteligência Artificial II

Artigos científicos da disciplina FGA0261 (UnB/FCTE, turma T02), em formato de
congresso IEEE, escritos em dupla.

## Começar

```sh
# dependências LaTeX (Arch)
sudo pacman -S texlive-basic texlive-latex texlive-latexrecommended \
               texlive-fontsrecommended texlive-bibtexextra texlive-binextra

# compilar o artigo 1
cd artigo-1/paper
make
```

O PDF sai em `artigo-1/paper/main.pdf`.

No VS Code, abra a pasta raiz e instale as extensões recomendadas — a compilação
passa a acontecer a cada salvamento, com preview lado a lado.

## Onde está o quê

- `CLAUDE.md` — contexto completo da disciplina, regras de avaliação, convenções.
  É o arquivo que o Claude Code lê ao abrir o projeto.
- `docs/guia-artigos.md` — ideias por rota temática, fontes de dados, erros comuns.
- `docs/plano-de-ensino.pdf` — o documento original do professor.
- `template-ieee/` — os arquivos originais do template IEEE, para consulta.
- `artigo-1/paper/` — o artigo em LaTeX.
- `artigo-1/src/` — código dos experimentos.
- `artigo-1/NOTAS.md` — decisões da dupla e checklist de entrega.
- `artigo-1/PLANO-EXPERIMENTAL.md` — pergunta de pesquisa, protocolo e fluxo de
  desenvolvimento local com execução final no Colab.

## Prazos

| Entrega | Data | Peso |
| ------- | ---- | ---- |
| Artigo 1 | 03/09/2026 | 20% |
| Artigo 2 | 09/10/2026 | 25% |
| Artigo 3 | 05/11/2026 | 25% |
| Apresentação | a partir de 08/12/2026 | 30% |
