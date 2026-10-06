# Artigo 2 — plano experimental

**Consolidado em 06/10/2026, com 3 dias de prazo.**

Este documento é a referência operacional para implementar e executar os
experimentos. As decisões resumidas também ficam em `NOTAS.md`.
A versão em linguagem corrente da ideia está em `EXPLICACAO-TELEMETRIA.md`
(não versionado).

## Ideia do artigo

**Rota II — aplicação de algoritmos a um problema proposto.**

> Treinar um autoencoder apenas em telemetria **normal** e usar o erro de
> reconstrução como escore de anomalia, para detectar falha de sensor sem precisar
> de histórico de falhas rotuladas.

A situação é real: a equipe UnBaja SAE não possui — e não terá tão cedo — um
histórico rotulado de falhas. Qualquer abordagem supervisionada está descartada
por construção do problema, não por conveniência.

## Pergunta de pesquisa

> Em telemetria veicular multivariada de baixa taxa, quanto um autoencoder
> recorrente treinado apenas em operação normal supera detectores estatísticos e
> lineares na identificação de falhas de sensor injetadas, e a que custo
> computacional?

A Introdução apresenta essa pergunta explicitamente e a Conclusão a responde com
os resultados observados, sem introduzir resultado novo.

## Conjunto de dados

**Automotive OBD-II Dataset (KIT)** — DOI `10.35097/1130`, CC BY 4.0, 11,6 MB,
Weber, M. (2023). Cobre rotação, temperatura do líquido de arrefecimento e
velocidade.

O download é feito pelo código, nunca por etapa manual não documentada.
A cópia do dataset não entra no Git (`artigo-*/src/data/` está no `.gitignore`).

### Verificação prática — feita em 06/10/2026

O dataset foi baixado e inspecionado antes de qualquer decisão de modelagem.
**Resultado: aprovado, com uma ressalva que muda o pré-processamento.**

| Medida | Valor observado |
| ------ | --------------- |
| Sessões | 81 arquivos CSV, todos do mesmo veículo (Seat Leon), 2017–2018 |
| Duração total | 70,0 h · mediana de 50,1 min por sessão · mínimo 12,2 min |
| Canais | 10 nominais, presentes em 81/81 sessões |
| Taxa de linha | ~11,3 Hz |
| Lacunas > 2 s | 11 sessões afetadas, 45 lacunas, a maior de 603,9 s |

#### A ressalva: a taxa de 11 Hz é falsa

Cada linha repete o último valor conhecido de cada canal. A taxa de **atualização
real** foi medida contando quantas vezes o valor de fato muda:

| Canal | Taxa real | Decisão |
| ----- | --------- | ------- |
| `fluxo_ar` | 1,11 Hz | **usar** |
| `rpm` | 1,07 Hz | **usar** |
| `pressao_adm` | 0,86 Hz | **usar** |
| `velocidade` | 0,61 Hz | **usar** |
| `temp_adm` | 0,44 Hz | **usar** |
| `pedal_d` | 0,33 Hz | **usar** |
| `pedal_e` | 0,39 Hz | descartar — correlação de 0,99 com `pedal_d` |
| `acelerador_abs` | 0,03 Hz | descartar — quase constante |
| `temp_arref` | 0,03 Hz | descartar — quase constante |
| `temp_ambiente` | 0,01 Hz | descartar — muda a cada ~100 s |

**Consequência metodológica:** o pipeline reamostra para **1 Hz**, não 11 Hz.
Treinar a 11 Hz faria o autoencoder aprender a copiar valores retidos e produziria
um erro de reconstrução artificialmente baixo — resultado inflado e falso. Essa
decisão precisa estar escrita na Metodologia, porque é exatamente o tipo de
armadilha que o leitor atento procura.

`pedal_d` e `pedal_e` são o par redundante do pedal do acelerador (dois sensores
no mesmo pedal, padrão de segurança automotiva). Fica apenas `pedal_d` na tarefa
principal. A redundância merece uma frase na Discussão: uma falha em só um dos dois
produz uma divergência fisicamente impossível, e é justamente por isso que o carro
carrega dois.

#### Segmentação e janelas

Cada sessão é cortada nas lacunas maiores que 2 s, e cada trecho contínuo vira um
segmento independente. Isso é o análogo direto do buffer do `baja-telemetry-api`:
a lacuna é operação normal, não anomalia, e não pode atravessar uma janela.

Com janela de 60 s e passo de 10 s, sobre **6 canais a 1 Hz**:

| | |
| - | - |
| Sessões aproveitadas | 77 (as 4 especiais ficam de fora) |
| Segmentos contínuos | 85 · 19 descartados por terem menos de 60 s |
| Amostras a 1 Hz | 235.340 (65,4 h) |
| **Janelas** | **23.073** |
| Tensor completo | 23.073 × 60 × 6 em float32 = 33,2 MB |

Folga confortável. O treino roda em CPU, sem necessidade de Colab.

#### As quatro sessões especiais

Quatro arquivos fogem do padrão `Normal`/`Stau`/`Frei`: `Messfehler` (erro de
medição), `Vollbremsung` (frenagem total), `Glatteis` (gelo) e `Beschleunigung`
(aceleração). As três últimas descrevem **condição de direção**, não falha de
sensor. Nenhuma apresenta valor globalmente fora da faixa da sessão de referência.

Todas ficam **fora do conjunto de treino**, por precaução. A sessão `Messfehler`
pode ser usada como verificação qualitativa na Discussão: um detector treinado só
em operação normal deveria marcá-la mais do que marca uma sessão normal. É barato e
é o único contato do artigo com uma anomalia que não foi injetada por nós.

## Representação canônica

Formato interno único: `(tempo, sinal, valor)`, reamostrado para matriz larga.
É exatamente o formato da tabela `signal_point` do `baja-telemetry-api`.

Cada dataset ganha só um *loader* que emite esse formato; janelamento, injeção de
falhas, treino, avaliação e relatório são compartilhados. De brinde, o formato de
entrada do detector passa a ser o formato de saída da API.

### A sutileza do tempo

`frame_time` (quando o dado foi gerado) e `received_at` (quando chegou no servidor)
podem estar separados por dez minutos, porque o carro acumula frames em buffer
quando sai do alcance do WiFi e despeja tudo ao voltar ao box.

**Isso é operação normal, não é anomalia.** O janelamento usa `frame_time`. Usar
`received_at` faria o detector aprender a marcar todo retorno ao box como falha.

## Protocolo de injeção de falhas

As falhas não são inventadas: são os modos de falha que a documentação do
`baja-telemetry-api` descreve como reais.

| Falha | Origem documentada | Por que importa |
| ----- | ------------------ | --------------- |
| **Ganho/escala errada** | `docs/05` — firmware muda `RPM_ESCALA` de 0,25 para 0,125 e o backend grava o dobro do RPM real | **Silenciosa**: passa por qualquer checagem de faixa. É o melhor argumento do artigo |
| **Sensor travado** | leitura para de variar sem sair da faixa válida | também silenciosa |
| **Deriva** | deriva térmica de sensor, típica no canal de temperatura | cresce devagar; limiar demora a pegar |
| **Pico/outlier** | ruído elétrico no barramento CAN | o caso em que o 3σ deve ganhar |
| **Lacuna** | `docs/06` — carro sai do alcance do WiFi | distinguir de buffer legítimo |

A falha de ganho é o argumento central: nenhuma validação de faixa a pega, e é
exatamente o caso em que um detector aprendido tem vantagem real sobre uma regra.

### Quanto cada falha escapa da checagem de faixa — medido em 06/10

`faults.violates_range` implementa o `if` que um firmware faria: para cada canal,
uma faixa de plausibilidade física (RPM entre 0 e 8.000, velocidade entre 0 e 300
km/h, e assim por diante). Essas faixas **não** são os extremos observados nos
dados; são os limites que uma validação real checaria.

Medindo sobre 400 janelas do split de teste por classe:

| Falha | Canal | Passa pela checagem |
| ----- | ----- | ------------------: |
| **Ganho** | `rpm` | **100,0 %** |
| **Travado** | `velocidade` | **100,0 %** |
| **Lacuna** | `fluxo_ar` | **100,0 %** |
| Deriva | `temp_adm` | 67,5 % |
| Pico | `pressao_adm` | 3,8 % |

E nenhuma das 3.191 janelas normais do teste viola a faixa — a regra não produz
falso positivo, ela simplesmente é cega para três das cinco classes.

Esses números sustentam o artigo inteiro e devem aparecer na Metodologia ou nos
Resultados: três das cinco falhas são **completamente invisíveis** para uma regra
de faixa, e o pico, que é o caso em que a regra funciona, é justamente onde o
limiar estatístico deve vencer o autoencoder.

## Modelos comparados

Todos treinados **apenas em janelas normais**, avaliados sob o mesmo protocolo.

| | Modelo | Papel |
| - | ------ | ----- |
| A | Limiar 3σ por canal sobre o sinal cru | linha de base ingênua — é o que um `if` no firmware faria |
| B | PCA (autoencoder linear), erro de reconstrução | responde se a não linearidade é necessária |
| C | **Autoencoder LSTM** sequência-a-sequência | o modelo proposto |

O autoencoder convolucional 1D do plano original foi **cortado** pelo prazo.

## Protocolo comum

- Split por sessão/veículo, nunca aleatório por janela — janelas vizinhas se
  sobrepõem e um split aleatório vaza.
- Normalização com estatísticas calculadas **somente no treino**.
- Limiar calibrado na distribuição de erro da **validação normal** (`μ + λσ`),
  nunca no teste.
- Seeds `42`, `1337`, `2026`, controlando inicialização, ordem dos lotes e a
  aleatoriedade da injeção de falhas.
- Nenhum modelo vê janela anômala em treino. Isso é uma **asserção no código**, não
  uma confiança.
- Cada execução produz um JSON identificado por modelo e seed em `src/results/`.

## Métricas

**Principal: AUC-PR.** Janelas anômalas são uma fração pequena do total. Nessa
situação a acurácia é inútil e a AUC-ROC engana — ela parece boa mesmo com o
detector ruim. Justificar isso explicitamente cobre o critério de correção nº 3.

Secundárias:

- F1, precisão e revocação no limiar calibrado;
- **latência de detecção** — quantas amostras até o alarme disparar. Um detector
  que acerta só depois do motor fundir não serve de alarme em prova;
- número de parâmetros e tempo de treino.

**Resultado desagregado por classe de falha.** A previsão é que o 3σ ganhe nos
picos e perca feio no ganho e na deriva. Esse contraste é o que carrega a nota na
Discussão.

## Tabela e figura

- **Tabela principal:** três modelos × AUC-PR, com média e desvio das três seeds,
  mais ao menos uma medida de custo.
- **Figura principal:** AUC-PR por classe de falha, mostrando que cada detector
  enxerga um tipo de problema diferente.

Os valores vêm de `src/results/` por `report.py`. **Nenhum número digitado à mão.**

## Reaproveitamento do Artigo 1

Copiar a espinha, não reescrever. Já existe e está testado em `artigo-1/src/`:

| Arquivo do Artigo 1 | Como entra aqui |
| ------------------- | --------------- |
| `config.py` | mesma ideia de fonte única de hiperparâmetros |
| `train.py` (`train_eval`) | loop comum; trocar a perda para reconstrução (MSE) |
| `metrics.py` | acrescentar AUC-PR e latência; a estrutura de saída fica |
| `run.py` | orquestração modelo × seed, já pronta |
| `report.py` | JSON → `resumo.csv` → tabela LaTeX → figura |
| `tests/`, `pytest.ini` | padrão de teste |
| `paper/` | esqueleto LaTeX, `Makefile`, `build.ps1` |

As quatro skills em `.claude/skills/` valem sem alteração.

## Cronograma — 3 dias

| Até | Entrega |
| --- | ------- |
| **06/10, fim do dia** | dataset baixado e verificado; `PLANO-EXPERIMENTAL.md` fechado contra o que os dados realmente permitem |
| **07/10** | pipeline completo: loader, formato canônico, janelamento, cinco falhas, split, e os três modelos |
| **08/10** | execução canônica (3 modelos × 3 seeds), tabela, figura, e o artigo escrito |
| **09/10** | revisão contra os 4 critérios, 4 páginas, entrega no Teams pelos **dois** |

**Se atrasar, a ordem de corte é:** primeiro o PCA, depois a métrica de latência,
depois uma seed (ficando duas, com a limitação declarada no texto).
Nunca cai: a linha de base 3σ e o resultado por classe de falha.

## Limite de interpretação

O experimento usa falhas **injetadas** em dado público de outro veículo, não falhas
reais do carro da equipe. As conclusões ficam restritas a esse protocolo, a esse
dataset e às três famílias de detectores avaliadas. O artigo não valida um sistema
de alarme embarcado em prova.

## Ponte para o Artigo 3

O Artigo 3 (05/11) pede Rota III (XAI) ou IV (fairness), idealmente reaproveitando
um modelo já treinado. Este detector entrega isso: *por que* o autoencoder marcou
esta janela como anômala? Atribuição do erro de reconstrução por canal, SHAP sobre
a janela, e verificação de se o modelo aponta o sensor que de fato falhou.
