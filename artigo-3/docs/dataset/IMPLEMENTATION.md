# Implementação do dataset de avaliação

## Pré-requisitos

- tema e protocolo validados na issue #46;
- corpus extraído e manifesto congelado pela issue #48;
- acesso local aos JSONL extraídos, que não são versionados por direitos autorais;
- duas pessoas disponíveis para autoria e revisão cruzada.

## Fase 1 — Contrato e congelamento prévio

- [x] Fixar 60 IDs e splits antes dos experimentos.
- [x] Fixar 30/15/10/5 categorias e a composição 8/4/2/1 no `dev`.
- [x] Distribuir seis itens por aula e cobrir as três partes.
- [x] Publicar JSON Schema do registro.
- [x] Validar IDs, contagens, páginas, evidências e consistência semântica básica.
- [x] Registrar autor, revisor e estado fora do dataset público.

Critério: `python -m src.planejamento_dataset` termina com código zero.

## Fase 2 — Anotação e revisão

- [ ] Redigir as 60 perguntas consultando o corpus extraído.
- [ ] Registrar resposta e todas as evidências válidas dos 55 itens respondíveis.
- [ ] Fazer busca negativa documentada para os cinco itens não respondíveis.
- [ ] Executar revisão cruzada cega à resposta do autor.
- [ ] Corrigir divergências e marcar cada item como `aprovado`.

Critério: `python -m src.dataset --dataset data/perguntas.jsonl` termina com código zero e todos os itens estão aprovados.

## Fase 3 — Congelamento experimental

- [ ] Gerar SHA-256 do dataset e do planejamento.
- [ ] Copiar o `dev` para o ambiente de desenvolvimento.
- [ ] Manter respostas e evidências de `test` fora do ambiente de ajuste.
- [ ] Registrar versão candidata, commit e hashes no manifesto experimental.
- [ ] Executar o teste uma única vez a partir de SHA destacado e Git limpo.

## Fase 4 — Publicação

- [ ] Remover qualquer texto protegido que exceda referência bibliográfica/localização.
- [ ] Confirmar licença CC BY 4.0 somente para as anotações autorais.
- [ ] Criar release imutável e tag `dataset-v1.0.0`.
- [ ] Arquivar no Zenodo e obter DOI.
- [ ] Substituir a citação provisória pelo DOI em `CITATION.cff` e `CITATION.bib`.
- [ ] Publicar hashes e relatório final de validação.

## Arquivos centrais

- `data/planejamento-dataset.jsonl`: IDs e splits congelados;
- `data/dataset-schema.json`: contrato público;
- `data/perguntas.jsonl`: anotações privadas até a publicação;
- `src/planejamento_dataset.py`: validação da matriz;
- `src/dataset.py`: validação do dataset;
- `docs/dataset/GUIA-ANOTACAO.md`: regras humanas;
- `dataset/`: metadados da futura release.
