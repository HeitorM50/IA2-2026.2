# Auditoria do artigo 3 segundo as avaliações dos artigos 1 e 2

Data: 09/11/2026. Versão examinada: `e39fb57`, branch `origin/feat/artigo-3-report`, e PDF `artigo-3/artigo3-IA2-chunking-RAG.pdf`.

**Parecer:** o artigo atende à estrutura exigida e apresenta um experimento pertinente, mas ainda contém inconsistências de relato e inferências que podem provocar descontos semelhantes aos dos artigos anteriores. As maiores oportunidades estão em resultados, discussão e reprodutibilidade. Não há base para prometer nota máxima ou prever a nota exata do professor.

A `main` local (`d86ece1`) e o `main.tex` consultado na branch padrão do GitHub ainda contêm o esqueleto. Esta auditoria usa a versão completa encontrada em `feat/artigo-3-report`. Durante a revisão passou a estar disponível o PDF `artigo-3/artigo3-IA2-chunking-RAG.pdf`: suas quatro páginas foram lidas e confirmam o texto, a tabela e os problemas descritos abaixo. Não foram modificados o artigo, o código experimental ou seus resultados.

## 1. O padrão de correção observado

As imagens fornecidas mostram 91/100 no artigo 1 e 89/100 no artigo 2. Os cinco critérios valem 20 pontos cada.

| Critério | Evidência das correções anteriores | Situação do artigo 3 |
| --- | --- | --- |
| I. Apresentação do problema | Delimitar pergunta, domínio e desfecho; não ampliar a lacuna além da revisão | Forte: pergunta explícita, domínio delimitado e baseline estrutural. Moderar afirmações amplas no resumo e na conclusão. |
| II. Metodologia proposta | Descrever escolhas operacionais, reprodução e medição de tempo | Parcial: modelos, divisão, grade e bootstrap descritos; faltam detalhes dos juízes, denominadores e cronometragem. |
| III. Resultados obtidos | Apresentar todas as métricas anunciadas, dispersão e análises desagregadas | Maior risco: completude e afirmações não sustentadas são anunciadas, mas não apresentadas; há inconsistências de denominadores e uma legenda contradita pelos dados. |
| IV. Discussão e conclusões | Não transformar associação em causa, ausência de significância em equivalência, nem medida parcial em custo completo | Maior risco: a conclusão sobre o gargalo do gerador depende de uma classificação que aceita evidência parcial como suficiente. |
| V. Bibliografia | Referências pertinentes, apoio exato à afirmação e documentação da configuração usada | Boa cobertura temática; melhorar citações das implementações e restringir afirmações sobre juízes e posição da evidência ao que as fontes demonstram. |

No artigo 2, o desconto mais expressivo foi em resultados (15/20), especialmente por métricas anunciadas e não apresentadas. No artigo 3, essa mesma fragilidade reaparece. O artigo 1 também perdeu pontos por detalhes de tempo e pelo uso de uma medida parcial para sustentar vantagem econômica; o artigo 3 deve delimitar seu custo a tokens de entrada e latência medida.

## 2. Verificações realizadas e seus limites

- Leitura integral do texto completo, referências, tabela, macros e CSV do commit `e39fb57`.
- Inspeção do código de métricas, avaliação dos juízes, classificação de falhas, geração e registros de execução.
- Compilação de uma cópia isolada: **4 páginas**, sem avisos LaTeX de citações/referências indefinidas, `Overfull` ou `Underfull` no log final. Inspeção visual da página de resultados.
- **39 testes passaram** nos módulos `test_report`, `test_analise_geracao`, `test_avaliacao_humana` e `test_avaliacao_recuperacao`. Isso verifica comportamentos implementados; não comprova a validade científica das métricas.
- Dois exemplos sintéticos confirmaram problemas de interpretação: recuperação de só uma entre duas evidências obrigatórias pode ser classificada como falha de geração; uma resposta com negação pode obter F1 lexical 1 contra a afirmação oposta.
- Os rankings individuais, os 315 JSON de geração e as planilhas dos juízes não estão nesta cópia nem na árvore versionada examinada. Portanto, **não foi possível recalcular os intervalos nem auditar as respostas e notas individuais**. As notas do projeto indicam que os brutos estão no Drive dos autores.
- O DOI do dataset está no artigo, na bibliografia e nos registros do projeto. As tentativas de abrir o DOI e o registro Zenodo nesta auditoria não tiveram sucesso; isso não demonstra que o DOI seja inválido, mas sua disponibilidade pública não foi confirmada.

O plano de ensino local, item 6, exige temas distintos, IEEE e até quatro páginas, mas não exige que o terceiro artigo seja especificamente XAI ou fairness. RAG é compatível com os temas do curso e distinto dos trabalhos de classificação de imagens e detecção de falhas. O planejamento antigo em `CLAUDE.md` não constitui restrição adicional do professor.

## 3. Correções prioritárias

### A. Rever a atribuição das falhas ao gerador — prioridade alta

**Evidência:** [`recuperou_evidencia`](https://github.com/HeitorM50/IA2-2026.2/blob/e39fb57/artigo-3/src/analise_geracao.py#L57) retorna verdadeiro se qualquer trecho contiver qualquer página anotada. Para perguntas multitrecho, o guia exige combinar evidências distintas. Uma resposta errada que recebeu somente parte da evidência já pode ser contada como `falha_geracao`.

Exemplo sintético confirmado: evidências obrigatórias nas páginas 1 e 2, contexto contendo somente a página 1, correção 0 → o código retorna `falha_geracao`. Não foi possível determinar quantos casos reais foram afetados sem os brutos.

Além disso, pertencer à página anotada não garante conter a passagem suficiente: uma janela pode incluir só uma fração daquela página. A classificação atual é um diagnóstico por correspondência de páginas, não uma identificação causal do componente responsável.

**Correção recomendada:** distinguir evidência ausente, parcial e suficiente; em itens multitrecho exigir a combinação necessária, respeitando alternativas válidas. Verificar o texto efetivamente entregue ao gerador. Recalcular os desfechos com os resultados já produzidos, preservando também os originais.

**Redação enquanto isso:** “Pela classificação baseada em presença de páginas-evidência, houve mais respostas incorretas com alguma evidência recuperada do que sem evidência; esse diagnóstico não isola a causa do erro.” Substituir no resumo e na conclusão a afirmação de que “o gargalo está no gerador”.

Isso responde diretamente à crítica feita no artigo 2 sobre atribuir a causa dos ganhos sem isolá-la experimentalmente.

### B. Apresentar as métricas de qualidade anunciadas — prioridade alta

**Evidência:** a metodologia anuncia correção, completude, fidelidade e contagem de afirmações não sustentadas ([texto, linha 128](https://github.com/HeitorM50/IA2-2026.2/blob/e39fb57/artigo-3/paper/main.tex#L128)). Os resultados mostram correção e fidelidade, mas não os resultados de completude nem de afirmações não sustentadas.

O gerador de relatórios já agrega afirmações não sustentadas, mas não agrega a média de completude. Existem macros e CSV para a primeira métrica:

| Configuração | Média do escore de afirmações não sustentadas |
| --- | ---: |
| Slide | 0,45 |
| 128 / 0% | 0,55 |
| 128 / 20% | 0,30 |
| 256 / 0% | 0,20 |
| 256 / 20% | 0,40 |
| 512 / 0% | 0,50 |
| 512 / 20% | 0,50 |

Esses valores foram lidos dos agregados, não recalculados das planilhas. O escore usa 3 para “três ou mais”; sua média não é uma contagem exata irrestrita.

**Correção recomendada:** publicar as duas métricas por configuração e explicar a escala, os denominadores e a incerteza. Se faltar espaço, simplificar uma figura ou transferir detalhes de implementação ao suplemento. Acrescentar a agregação de completude ao `report.py`, mantendo a regra de números gerados por código. Não inventar notas faltantes.

Também é útil reportar a abstenção indevida: há configurações que recusam perguntas respondíveis, embora todas acertem as quatro perguntas não respondíveis. O sucesso 4/4 não demonstra ausência geral de alucinações.

### C. Corrigir denominadores e o significado da precisão de citação — prioridade alta

**Evidência:** a frase “duas métricas automáticas cobrem todas as 315 respostas” não corresponde ao [`report.py`](https://github.com/HeitorM50/IA2-2026.2/blob/e39fb57/artigo-3/src/report.py#L170):

- F1 usa 41 perguntas respondíveis por configuração: 287 respostas ao todo.
- Precisão de citação exclui abstenções e respostas sem citação válida: 254 respostas ao todo, distribuídas de forma desigual.
- `Cit.` é a fração com citação entre as respostas não abstidas, não entre todas as 45 perguntas.
- A diferença pareada de precisão usa somente a interseção das perguntas elegíveis nas duas configurações. As médias marginais da tabela e a diferença pareada podem, portanto, se referir a subconjuntos diferentes.

As contagens abaixo foram inferidas dos agregados, que permitem recuperar os inteiros, mas não identificar os pares específicos:

| Configuração | Respostas com citação / respostas não abstidas |
| --- | ---: |
| Slide | 38/40 |
| 128 / 0% | 36/38 |
| 128 / 20% | 37/39 |
| 256 / 0% | 38/41 |
| 256 / 20% | 39/40 |
| 512 / 0% | 35/39 |
| 512 / 20% | 31/38 |

A métrica é a média, por resposta, da fração de IDs citados distintos que intersectam uma página-evidência. Não verifica se cada afirmação é sustentada pela passagem citada, nem corresponde necessariamente à razão global entre todas as citações relevantes e todas as citações.

**Correção recomendada:** explicitar o cálculo e os subconjuntos na metodologia/legenda, fornecer o número de pares de cada comparação e chamar a medida de relevância das citações por página. Apresentar cobertura de citação junto da precisão. Uma análise suplementar em todas as perguntas respondíveis, com tratamento explícito da ausência de citação, ajuda a verificar sensibilidade à seleção; não substituir silenciosamente a métrica original.

### D. Remover contradições e conclusões excessivas — prioridade alta, execução simples

1. **Figura 3:** a legenda diz que as falhas de geração predominam em todas as configurações. Em `c256-o20`, os dados registram uma falha de geração e uma de recuperação. Corrigir para “predominam em seis configurações, com empate em uma”.
2. **“Menor precisão”:** a discussão não identifica qual precisão diminui. P@5 é 0,22 para slide e 0,26/0,33 para as janelas de 512 tokens, portanto não diminui em relação ao baseline. Há redução em relação às janelas de 128 tokens; já a precisão de citação diminui em relação ao slide. Especificar métrica e comparador.
3. **“Tão bem quanto”:** intervalos das diferenças que incluem zero não demonstram equivalência. Preferir “obteve o maior MRR observado, sem evidência conclusiva de diferença nas comparações pareadas”.
4. **“A segmentação estrutural basta”:** restringir ao corpus, orçamento e métricas examinados; a correção média dos juízes não sustenta suficiência prática geral.
5. **Exceção de precisão de citação:** o IC pareado de `512 / 0%` contra slide é [-0,50; -0,12], conforme as macros. Essa é uma evidência exploratória dentro de várias comparações; o relatório usa ICs individuais de 95%, sem ajuste de multiplicidade. Qualificar a conclusão ou fazer análise de sensibilidade antes de tratá-la como confirmação definitiva.

Esses ajustes retomam exatamente o cuidado com inferências e conclusões cobrado nos dois artigos anteriores.

### E. Tornar a avaliação por juízes auditável — prioridade alta

O texto declara corretamente que os juízes são LLMs, mas “Claude Opus” e “Claude Sonnet” não identificam versões reprodutíveis. Faltam os identificadores completos, data, configuração, prompt integral, forma de apresentação dos itens, política de repetição/retentativa e tratamento de saídas inválidas.

**Correção recomendada:** anexar esses metadados e as notas por item. Usar “dois modelos avaliados separadamente” se a independência operacional estiver documentada; não inferir independência estatística dos erros por serem modelos diferentes. Uma avaliação humana cega da mesma amostra, se realizada de fato, permitiria verificar a validade dos julgamentos automáticos.

Os 70 itens correspondem às mesmas 10 perguntas em sete configurações. Os agregados mostram duas perguntas não respondíveis por configuração; os 56 itens respondíveis são, assim, **oito perguntas repetidas sete vezes**, não 56 perguntas independentes. Explicitar isso ao interpretar os 35 acertos, 16 falhas de geração e cinco de recuperação. Os kappas são concordância entre juízes, não prova de correção das notas.

### F. Delimitar custo e detalhar a cronometragem — prioridade média/alta

[`responder`](https://github.com/HeitorM50/IA2-2026.2/blob/e39fb57/artigo-3/src/geracao.py#L108) mede a latência de geração entre o fim da busca e o retorno do gerador. Isso inclui construção do prompt, tokenização, geração e decodificação; não inclui a busca anterior nem o carregamento inicial dos modelos. O backend contabiliza os tokens do prompt completo após o chat template, não somente os trechos recuperados.

**Correção recomendada:** renomear “tokens de contexto” para “tokens de entrada do gerador”, definir o intervalo de medição e limitar a conclusão a essa etapa. Informar ordem de execução, aquecimento, processamento por pergunta e tokens de saída; acrescentar dispersão/mediana/p95 se os brutos permitirem. Descrever o que efetivamente foi feito, sem acrescentar retrospectivamente aquecimento ou sincronização inexistentes.

O estudo não mede custo monetário, energia, memória ou custo total de implantação. As referências à GPU T4 e à precisão já ajudam, mas não substituem a definição operacional da medida.

### G. Responder diretamente sobre sobreposição e explicitar as métricas de evidência — prioridade média/alta

O título promete investigar tamanho **e sobreposição**, mas as comparações inferenciais concentram-se em cada configuração contra `slide`. Acrescentar comparações pareadas de 20% contra 0% dentro de cada tamanho, declaradas exploratórias.

Os agregados já indicam que o efeito não é uniforme: Recall@5 passa de 0,89 para 0,91 em 128 tokens, de 0,89 para 0,93 em 256 e de 0,94 para 0,89 em 512. Esses são valores descritivos arredondados; os ICs das diferenças precisam dos rankings individuais.

O Recall implementado mede cobertura de **todas as páginas anotadas**. Em cinco perguntas factuais do teste (`q003`, `q019`, `q050`, `q056`, `q057`) há múltiplas páginas alternativas, embora uma só possa bastar para responder. Portanto, Recall menor que 1 não significa necessariamente evidência insuficiente. Manter a definição de cobertura de páginas e acrescentar/explicitar sucesso de recuperação por suficiência: alguma alternativa nas factuais/contextuais, combinação exigida nas multitrecho.

O orçamento fixo de cinco chunks permite comparar os sistemas nessas condições, mas não isola tamanho de trecho de quantidade total de texto. O aumento de tokens é parte do efeito observado. Uma comparação futura com orçamento fixo de tokens avaliaria outra pergunta; não é obrigatório ampliar agora a grade para reconhecer essa limitação.

### H. Corrigir ou limitar o F1 lexical — prioridade média

O normalizador remove `nao` e `sem`. Um exemplo sintético executado na auditoria retornou F1 = 1 para “O modelo não usa atenção” contra “O modelo usa atenção”. Isso confirma que a métrica pode ignorar uma contradição, além de penalizar paráfrases.

**Correção recomendada:** chamá-la explicitamente de sobreposição lexical, sem atribuir-lhe validação factual; preservar negações em uma variante e recalcular com transparência, mantendo a versão original como análise de sensibilidade. Documentar a normalização específica e citar a origem da métrica, caso se mantenha a analogia com SQuAD no código.

### I. Ajustar bibliografia e pacote de reprodução — prioridade média/alta

- Incluir referências diretas aos cartões de [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) e [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), vinculadas às revisões realmente usadas. É o equivalente ao desconto do artigo 1 pela falta de documentação específica dos pesos ImageNet.
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) relata desempenho frequentemente melhor no começo **ou no fim** do contexto. Não sustenta a simplificação de que tudo que não está no início é pior, nem prova que esse mecanismo causou os resultados deste Qwen.
- [Panickssery et al.](https://arxiv.org/abs/2404.13076) investigam reconhecimento e preferência pelas próprias gerações. A fonte justifica cautela com vieses, mas não estabelece, por si, a afirmação geral de que dois juízes LLM concordam mais entre si do que humanos. Reformular essa afirmação ou fornecer evidência diretamente correspondente.
- A referência [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/) é pertinente à granularidade; completar suas páginas (15159–15177) e DOI. A referência [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) também corresponde ao assunto discutido.
- O link do repositório já aparece no artigo, corrigindo uma deficiência do artigo 1. Entretanto, a branch padrão examinada não contém o texto final. Preparar uma versão/release imutável que vincule artigo, execução `4f347d6` e análise `e39fb57` ou sucessora.
- Disponibilizar rankings, respostas, notas dos juízes e metadados de análise em local estável, respeitando as restrições de redistribuição dos slides. Documentar o caminho de acesso ao corpus e confirmar o download do dataset pelo DOI. Ter hashes não substitui ter acesso aos arquivos correspondentes.

## 4. Ordem de trabalho para buscar a maior nota

| Ordem | Ação | Exige nova geração pelo Qwen? |
| --- | --- | --- |
| 1 | Recuperar rankings, respostas e planilhas da execução original | Não |
| 2 | Auditar suficiência das evidências e recalcular a classificação de falhas | Não; exige revisão das evidências |
| 3 | Completar as métricas anunciadas e corrigir denominadores | Não |
| 4 | Corrigir legenda, comparadores, equivalência e conclusões causais | Não |
| 5 | Detalhar juízes, tempo, tokens e fontes específicas | Não; depende dos registros existentes |
| 6 | Acrescentar diferenças pareadas de sobreposição e análises de sensibilidade | Não, com os brutos |
| 7 | Validar a mesma amostra com avaliadores humanos cegos | Não; exige avaliação humana real |
| 8 | Regenerar tabelas/figuras/macros e recompilar, preservando quatro páginas | Não |

Não é necessário começar por um gerador maior ou por refazer a grade experimental. O ganho mais direto está em alinhar pergunta, implementação, métricas apresentadas e força das conclusões. Qualquer mudança posterior na análise deve ser registrada como tal, sem apresentar decisões feitas após os resultados como escolhas prévias ao teste.

## 5. Critério de encerramento da revisão

- Todas as métricas anunciadas aparecem com definição, escala e denominador.
- Nenhuma conclusão de causa ou equivalência excede o desenho e a evidência.
- Todas as legendas e comparações são compatíveis com o CSV.
- A pergunta sobre sobreposição recebe resposta explícita.
- Juízes, modelos e execução têm identificação reproduzível.
- O pacote permite reconstruir as tabelas a partir dos registros ou documenta claramente o acesso necessário.
- O PDF final conserva quatro páginas, referências resolvidas e apresentação legível.

Até essas condições serem verificadas, a versão é um trabalho com boa base experimental, mas ainda não uma versão revisada para minimizar os descontos já conhecidos do professor.
