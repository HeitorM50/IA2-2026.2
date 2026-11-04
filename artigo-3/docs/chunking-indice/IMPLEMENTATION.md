# Implementação — chunking e índice vetorial

## Fase 1 — Contrato e tokenização

- [x] congelar tokenizador, modelo, revisão e prefixos;
- [x] criar chunks determinísticos por tokens;
- [x] preservar páginas, aula, parte, título e origem;
- [x] testar limites, sobreposição e fronteiras entre aulas.

## Fase 2 — Embeddings e busca

- [x] normalizar documentos e consultas;
- [x] construir `IndexFlatIP`;
- [x] implementar o contrato `Recuperador`;
- [x] testar conteúdo, scores, metadados e determinismo.

## Fase 3 — Cache e manifesto

- [x] derivar chave completa do cache;
- [x] persistir índice, embeddings, chunks e manifesto atomicamente;
- [x] registrar contagens, tamanhos e tempos;
- [x] testar reutilização, invalidação e corrupção.

## Fase 4 — Integração

- [x] expor `construir_recuperador` para `src.run`;
- [x] atualizar dependências e documentação;
- [x] executar a suíte completa.
