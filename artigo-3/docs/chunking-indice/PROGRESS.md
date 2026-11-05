# Progresso — chunking e índice vetorial

## Estado: implementação concluída

Data de referência: 4 de novembro de 2026.

## Decisões congeladas

- baseline estrutural `slide` mais seis janelas fixas;
- modelo e tokenizador `intfloat/multilingual-e5-base`;
- revisão `d128750597153bb5987e10b1c3493a34e5a4502a`;
- prefixos E5 para consulta e passagem;
- arredondamento da sobreposição para o inteiro mais próximo;
- cache fora do Git em `cache/indices/`.

## Resultado

- 80 testes aprovados em 20,86 segundos;
- FAISS, Sentence Transformers e Transformers importados nas versões congeladas;
- `pip check` sem dependências quebradas;
- código compilado e `git diff --check` sem erros.

## Sessão

- Fase 1 concluída: `src/chunking.py` e 10 testes aprovados.
- Fases 2 e 3 concluídas: normalização, FAISS, cache e manifesto implementados.
- FAISS real validado com persistência e reconstrução do mesmo ranking.
- Fase 4 concluída: integração, dependências e suíte completa verificadas.
