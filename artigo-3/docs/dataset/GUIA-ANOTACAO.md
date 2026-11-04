# Guia de anotação do dataset de avaliação

## Escopo e fonte

O dataset avalia sistemas RAG sobre as aulas 14–23 de Inteligência Artificial II.
Somente o corpus congelado em `data/manifesto-corpus.json` pode sustentar uma resposta.
Conhecimento externo, ainda que correto, não torna uma pergunta respondível.

Cada item deve preservar o `id`, o `split`, a parte, a aula e a categoria já definidos em `data/planejamento-dataset.jsonl`.
Esses metadados foram fixados antes dos experimentos e não podem ser ajustados em resposta ao desempenho do sistema.

## Categorias

### Factual

A resposta é uma informação explícita recuperável em uma única página.
Uma paráfrase fiel é aceitável, mas não deve exigir inferência externa.

### Contextual

A resposta exige interpretar uma afirmação no contexto da aula, relacionar definição e consequência ou desambiguar o sentido de um conceito.
Ainda deve existir ao menos uma página suficiente para sustentar a resposta.

### Multitrecho

A resposta completa exige combinar pelo menos duas evidências distintas.
Cada evidência isolada deve ser insuficiente para a resposta completa.
As evidências podem estar em páginas ou aulas diferentes e todas devem ser listadas.

### Não respondível

O corpus não fornece informação suficiente para responder.
A pergunta deve permanecer relacionada aos temas das aulas e parecer plausível, sem ser absurda ou facilmente identificável apenas pelo estilo.
`resposta_referencia` deve ser `null` e `evidencias_validas` deve ser uma lista vazia.

## Redação

Uma pergunta é aceita quando:

- possui uma única interpretação razoável;
- pode ser compreendida sem apontadores como “no trecho acima”;
- não revela a resposta na própria formulação;
- não depende de figura ilegível, conhecimento externo ou opinião;
- usa os termos das aulas sem copiar longas passagens protegidas;
- tem granularidade compatível com uma resposta curta e verificável.

A resposta de referência deve conter somente as proposições necessárias.
Exemplos ou justificativas adicionais só entram quando também forem sustentados pelas evidências.

## Evidências

Cada evidência é o par `aula` e `pagina`, com numeração iniciada em 1 conforme o leitor de PDF.
O primeiro par deve coincidir com os campos principais `aula` e `pagina` do item.
Quando o mesmo conceito aparece em mais de um local, todas as alternativas suficientes podem ser registradas.

Em itens multitrecho, a lista representa evidências que devem ser combinadas, não alternativas independentes.
A revisão deve registrar em sua ata quando duas combinações diferentes também forem válidas.

Não se publica o texto integral dos slides.
Trechos literais usados durante a revisão devem ficar em área privada e ser limitados ao mínimo necessário.

## Regra de afirmação não sustentada

Uma afirmação é uma proposição verificável da resposta.
Ela é não sustentada quando não é implicada pelo contexto recuperado, ainda que seja verdadeira por conhecimento externo.
Detalhes, números, causas, exemplos e conclusões adicionais contam como afirmações separadas.
Paráfrases semanticamente equivalentes e conectivos puramente linguísticos não contam como novas afirmações.

Se uma resposta contém uma proposição correta sustentada e outra relevante sem suporte, ela deve ser marcada como contendo afirmação não sustentada.

## Regra de abstenção

A forma canônica é: “Não há informação suficiente nas aulas para responder.”

- Em item não respondível, essa recusa sem acréscimos factuais é `abstencao_correta`.
- Em item respondível, a recusa é `abstencao_incorreta`.
- Inventar uma justificativa para a recusa conta como afirmação não sustentada.
- Dar uma resposta parcial sem reconhecer a insuficiência não conta como abstenção correta.

## Revisão cruzada

O autor preenche pergunta, resposta e evidências.
O revisor tenta responder consultando apenas o corpus, sem ler antes a resposta do autor, e depois verifica:

1. clareza e unicidade da interpretação;
2. categoria atribuída;
3. correção e suficiência da resposta de referência;
4. existência e suficiência de cada evidência;
5. presença de outras evidências válidas;
6. respondibilidade contra o corpus completo.

Autor e revisor devem ser pessoas diferentes.
O estado percorre `planejado`, `redigido`, `revisado`, `corrigido`, `aprovado` e `congelado`.
Somente itens `aprovado` ou `congelado` entram na versão candidata.

Para uma pergunta não respondível, o revisor deve documentar buscas pelo termo principal, sinônimos, siglas e conceitos relacionados nas dez aulas.
A confirmação exige ausência de evidência suficiente, não a mera ausência da mesma formulação textual.

## Resolução de divergências

Autor e revisor registram a divergência e chegam a uma redação consensual.
Se a categoria ou o split precisasse mudar, o item deve ser reescrito dentro da célula originalmente planejada; o split não pode ser trocado.
Persistindo desacordo sobre respondibilidade, o item é descartado e substituído por outro com o mesmo ID e metadados planejados antes do congelamento final.
