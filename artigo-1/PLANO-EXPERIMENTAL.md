# Artigo 1 — plano experimental e uso dos ambientes

**Decisões consolidadas em 29/08/2026.**

Este documento registra a ideia científica do Artigo 1, o protocolo comum aos
modelos e a divisão de uso entre a máquina local e o Google Colab.
As decisões resumidas também devem permanecer em `NOTAS.md`, enquanto este arquivo
serve como referência operacional para implementar e executar os experimentos.

## Ideia do artigo

O Artigo 1 seguirá a **Rota I — comparação entre algoritmos**.
Será investigado quanto o aumento de complexidade do modelo melhora a classificação
de oito tipos de células sanguíneas em imagens microscópicas.

A comparação representa três níveis de capacidade:

1. uma fronteira de decisão linear sobre os pixels;
2. características visuais aprendidas do zero por uma rede convolucional;
3. características transferidas de uma rede previamente treinada em uma base ampla.

A contribuição pretendida não é somente identificar o maior valor de desempenho.
O artigo deve quantificar o ganho de cada nível de complexidade, verificar sua
estabilidade entre execuções, identificar as classes confundidas e discutir se o
ganho compensa o custo computacional.

## Pergunta de pesquisa

> Em classificação multiclasse de tipos celulares no BloodMNIST, em que medida
> uma CNN compacta treinada do zero e uma ResNet18 pré-treinada no ImageNet-1K
> superam uma regressão logística em F1 macro, sob o mesmo protocolo experimental?

A Introdução apresentará essa pergunta explicitamente e a Conclusão deverá
respondê-la com os resultados observados, sem introduzir novos resultados.

## Conjunto de dados

Será usado o **BloodMNIST 64 × 64**, subconjunto RGB da coleção MedMNIST+.
Ele contém 17.092 imagens microscópicas distribuídas entre oito tipos celulares:
basófilo, eosinófilo, eritroblasto, granulócito imaturo, linfócito, monócito,
neutrófilo e plaqueta.

Serão preservados os splits oficiais:

| Split | Imagens |
| ----- | ------: |
| Treino | 11.959 |
| Validação | 1.712 |
| Teste | 3.421 |

A distribuição oficial está no Zenodo sob o DOI
[`10.5281/zenodo.10519652`](https://doi.org/10.5281/zenodo.10519652).
O download deverá ser feito pelo código, nunca por uma etapa manual não documentada.

Os splits permanecerão fixos em todos os modelos e seeds.
Assim, a variação observada será atribuída ao treinamento, e não à troca das
amostras de teste.

## Modelos comparados

### Modelo A — regressão logística multinomial

Será a linha de base simples e não neural.
Cada imagem normalizada será achatada em um vetor, sem camada oculta, e uma camada
linear produzirá os oito logits de classificação.
Esse modelo responderá se representações não lineares são realmente necessárias.

### Modelo B — CNN compacta treinada do zero

Uma rede neural convolucional (CNN) própria aprenderá características locais de
forma, textura e coloração diretamente no BloodMNIST.
Todos os pesos serão inicializados aleatoriamente em cada seed.
A arquitetura exata, a regularização e os hiperparâmetros serão registrados em
`config.py` durante as issues de implementação.

### Modelo C — ResNet18 com transferência de aprendizado

A ResNet18 será inicializada com pesos do ImageNet-1K.
A camada classificadora será substituída por uma saída de oito classes e toda a
rede será submetida a ajuste fino no BloodMNIST.
Serão reportados separadamente os números de parâmetros totais e treináveis.

## Protocolo experimental comum

Os três modelos receberão imagens RGB 64 × 64 e usarão os mesmos splits.
A normalização terá média e desvio calculados somente sobre o treino.
Qualquer aumento de dados será aplicado exclusivamente no treino.
Validação e teste terão apenas transformações determinísticas.

Serão usadas as seeds `42`, `1337` e `2026`.
Elas controlarão a inicialização, a ordem dos lotes e a aleatoriedade do aumento
de dados, mas não modificarão os índices dos splits oficiais.

A seleção de hiperparâmetros, a parada antecipada e a escolha da melhor época
observarão apenas a validação.
O teste será avaliado uma vez ao final de cada execução.
Todos os valores deverão vir de uma única configuração versionada, sem constantes
duplicadas entre scripts ou células de notebook.

## Métricas e resultados

A métrica principal será o **F1 macro**.
Ela calcula o F1 de cada classe separadamente e atribui o mesmo peso aos oito tipos
celulares, reduzindo a influência das diferenças de frequência entre classes e
considerando simultaneamente precisão e revocação.

Serão métricas secundárias:

- acurácia balanceada;
- acurácia;
- tempo de treinamento;
- número total de parâmetros;
- número de parâmetros treináveis;
- matriz de confusão e desempenho por classe.

A tabela principal apresentará média e desvio padrão das três seeds, com F1 macro
na primeira coluna e ao menos uma medida de custo.
A figura principal será a matriz de confusão de teste do modelo com maior F1 macro
médio. As matrizes das três seeds serão somadas e normalizadas por classe
verdadeira, mostrando quais tipos celulares ainda são confundidos sem repetir a
comparação global já apresentada pela tabela.

Cada execução produziu um JSON identificado por modelo e seed.
Os nove JSONs alimentam automaticamente `resumo.csv` e a figura principal.
Na Issue #10, a tabela LaTeX deverá ser gerada ou importada a partir de
`resumo.csv`; se houver transcrição, todos os valores deverão ser conferidos
mecanicamente contra os JSONs, sem usar números lembrados ou recalculados à mão.

## Uso da máquina local

A máquina local de referência possui Intel Core i5-1135G7, 8 GB de RAM e GPU
integrada Intel Iris Xe, sem CUDA.
Ela consegue armazenar o BloodMNIST e executar os três modelos em CPU, mas a
ResNet18 é lenta para as nove execuções completas e compete por uma quantidade
limitada de memória.

O ambiente local será usado para:

1. escrever e revisar o código;
2. baixar e inspecionar o conjunto de dados;
3. testar splits, normalização e reprodutibilidade;
4. executar testes unitários e modelos de brinquedo;
5. fazer execuções de fumaça com poucas amostras ou uma época;
6. validar o formato dos JSONs, CSVs, tabelas e figuras;
7. compilar o artigo com `make`.

A regressão logística e a CNN compacta podem ser executadas completamente em CPU
durante o desenvolvimento, se isso ajudar a encontrar erros.
Essas execuções locais, entretanto, não serão misturadas aos resultados finais.

## Quando usar o Google Colab

O Colab será usado somente depois que o pipeline passar localmente de ponta a
ponta.
O objetivo é reservar a GPU para a execução final, sem transformar o notebook em
uma segunda implementação do projeto.

Antes de abrir a sessão de GPU, devem estar confirmados:

- download automático do BloodMNIST;
- splits oficiais e distribuição de classes conferidos;
- uma execução curta de cada modelo concluída;
- métricas e JSONs gerados sem campos ausentes;
- seed aplicada a `random`, NumPy e PyTorch;
- teste isolado das decisões de treinamento;
- commit exato da execução identificado.

No Colab, o procedimento será:

1. selecionar um runtime com GPU e registrar o modelo da GPU;
2. obter o repositório no commit que passou nos testes locais;
3. instalar as dependências declaradas pelo projeto;
4. executar o mesmo `run.py` usado localmente, sem copiar o loop para células;
5. rodar os três modelos com as seeds `42`, `1337` e `2026`;
6. salvar cada JSON assim que a execução terminar, evitando perda por desconexão;
7. gerar `resumo.csv` e as figuras pelo código versionado;
8. trazer apenas resultados e figuras para a branch de resultados.

Apesar de somente os modelos neurais precisarem efetivamente da GPU, as **nove
execuções finais serão feitas no mesmo ambiente Colab**.
Isso é necessário para tornar comparáveis as medições de tempo de treinamento.
Se o modelo de GPU mudar entre sessões, o hardware usado deverá constar nos JSONs;
tempos obtidos em GPUs diferentes não serão comparados diretamente.

O notebook, se existir, será apenas um lançador documentado.
O carregamento dos dados, os modelos, o treinamento e as métricas continuarão em
`artigo-1/src/`, que será a única fonte da implementação.

## Artefatos versionados

Devem entrar no Git:

- código, configurações e lista de dependências;
- um JSON por par modelo × seed;
- `resumo.csv` consolidado;
- scripts de geração das figuras;
- figuras finais usadas no artigo.

Não devem entrar no Git:

- cópia do conjunto de dados;
- checkpoints grandes, salvo decisão explícita em contrário;
- arquivos temporários do Colab;
- artefatos auxiliares da compilação LaTeX.

## Decisões consolidadas na implementação

As Issues #1 a #7 concluíram a fase experimental com estas decisões:

- CNN compacta com três blocos de 32, 64 e 128 canais, duas convoluções `3 × 3`
  por bloco, normalização em lote, ReLU, max pooling, pooling médio global e
  dropout de 0,30;
- aumento geométrico somente no treino, com rotações de até 15 graus e reflexões
  horizontal e vertical com probabilidade de 0,5;
- lotes de 64 imagens, no máximo 30 épocas, `min_delta=1e-4` e parada antecipada
  após cinco épocas sem melhora no F1 macro de validação;
- AdamW com `weight_decay=1e-4`, taxa `1e-3` para regressão logística e CNN e,
  na ResNet18, `1e-4` no backbone e `1e-3` na cabeça;
- matriz de confusão de teste da ResNet18 agregada entre as três seeds e
  normalizada por classe verdadeira como figura principal.

As nove execuções canônicas foram realizadas no mesmo runtime Google Colab, com
GPU Tesla T4, e produziram todos os JSONs, `resumo.csv` e a figura versionados.
Não serão adicionados outros datasets, arquiteturas ou estudos de ablação durante
a escrita. Essa contenção mantém o artigo centrado na pergunta de pesquisa.

## Limite de interpretação

O experimento compara algoritmos em um benchmark de imagens pequenas e não valida
um sistema de diagnóstico clínico.
As conclusões deverão permanecer restritas ao BloodMNIST, ao protocolo adotado e
às três famílias de modelos avaliadas.
