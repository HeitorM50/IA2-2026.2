# Artigo 2 — decisões e andamento

**Prazo: 09/10/2026** · vale 25% da nota final · entrega pelo Teams.

> ⚠️ **Este artigo começou em 06/10, com 3 dias de prazo.** O escopo abaixo já é a
> versão enxuta. Qualquer ampliação precisa ser decidida contra o relógio, não
> contra o que seria ideal.

## Dupla

Divisão em fatia vertical, como no Artigo 1: cada um leva código **e** seções,
para que ninguém fique bloqueado esperando o outro e os dois conheçam o artigo
inteiro na apresentação.

| Integrante | GitHub | Matrícula | Eixo |
| ---------- | ------ | --------- | ---- |
| Heitor Macêdo Ricardo | `HeitorM50` | 241039073 | **dados e falhas** — pipeline, injeção, execução · Metodologia, Discussão, Resumo |
| Gustavo Xavier Evangelista | `guxvr` | 241025247 | **modelos e avaliação** — métricas, treino, modelos, relatório · Introdução, Resultados, `refs.bib` |

| | Heitor | Gustavo |
| - | ------ | ------- |
| Código | #27 `config`/`data` · #28 `faults` · #31 Modelo A (3σ) · #34 `run` e execução | #29 `metrics` · #30 `train` · #32 Modelo B (PCA) · #33 Modelo C (LSTM) · #35 `report` |
| Escrita | #37 Metodologia · #39 Discussão e Limitações · #41 Resumo e Conclusão | #36 Introdução e Trabalhos Relacionados · #38 Resultados · #40 `refs.bib` |
| A quatro mãos | #42 revisão final · #43 entrega | |

O andamento fica nas issues do GitHub, no milestone `Artigo 2 — 09/10/2026`,
que vence em 08/10 — um dia antes do prazo real.

### O caminho crítico, que é o que importa com 3 dias

A **#27 é bloqueante**: todo o resto consome o tensor que ela produz. Por isso ela
sai primeiro e, antes de estar pronta, o contrato de interface precisa estar
publicado como comentário na própria issue:

```
X: float32 (n_janelas, 60, 6)
canais: rpm, velocidade, fluxo_ar, pressao_adm, temp_adm, pedal_d
sessao: (n_janelas,) — id da sessão de origem, para o split
```

Com esse contrato fechado, as issues #29, #30, #32 e #33 podem ser escritas contra
um tensor sintético, sem esperar o pipeline real. **Quem não fizer isso vai ficar
parado metade do prazo.**

As issues que não dependem de nada e podem começar imediatamente: **#29**
(métricas), **#36** (Introdução e Trabalhos Relacionados), **#40** (`refs.bib`) e
**#37** (Metodologia). Escrever em paralelo com o código é obrigatório aqui — não
há tempo para escrever só depois que os resultados saírem.

## Decisões do experimento

- [x] **Rota temática** — **II, aplicação de algoritmos a um problema proposto**.
      A contribuição é a solução; o peso cai na Metodologia.
- [x] **Tema distinto do Artigo 1** — o Artigo 1 foi classificação supervisionada
      de imagens (BloodMNIST, Rota I). Este é série temporal multivariada com
      detecção **não supervisionada**. Exigência do item 6 do plano de ensino
      satisfeita com folga.
- [x] **Janela de ementa** — até 09/10 o curso cobriu RNN/LSTM, autoencoders,
      VAE/GAN e Transformers. O autoencoder recorrente cai no centro.
- [x] **Conjunto de dados** — Automotive OBD-II (KIT), DOI `10.35097/1130`,
      CC BY 4.0, 11,6 MB. **Verificado em 06/10**: 81 sessões, 70 h, 10 canais.
      Aprovado, com reamostragem obrigatória para 1 Hz (ver `PLANO-EXPERIMENTAL.md`).
- [x] **Modelos** — limiar 3σ, PCA e autoencoder LSTM.
- [x] **Métrica principal** — AUC-PR, pelo desbalanceamento extremo entre janelas
      normais e anômalas. Secundárias: F1 no limiar calibrado, precisão, revocação
      e **latência de detecção**.
- [x] **Seeds** — `42`, `1337`, `2026`, as mesmas do Artigo 1.
- [x] **Onde treinar** — local. O tensor completo tem 33,2 MB e o modelo é pequeno;
      o Colab não é necessário.

## Pergunta de pesquisa

> Em telemetria veicular multivariada de baixa taxa, quanto um autoencoder
> recorrente treinado apenas em operação normal supera detectores estatísticos e
> lineares na identificação de falhas de sensor injetadas, e a que custo
> computacional?

## Registro de decisões

Anotar aqui, com data, cada decisão fechada. Isso vira material da Metodologia.

- **15/09/2026** — ideia do artigo definida: detecção não supervisionada de
  anomalia em telemetria veicular, servindo à equipe UnBaja SAE. Plano completo
  redigido com cronograma de 24 dias.
- **06/10/2026** — retomada do tema após avaliar alternativas (conectoma do
  *Drosophila*: neurotransmissor por autoencoder, RNN restrita pelo conectoma,
  dimorfismo sexual, predição de conexão). Todas descartadas para o Artigo 2 por
  exigirem preparação de dados incompatível com 3 dias de prazo. A ideia da **RNN
  restrita pelo conectoma** fica registrada como forte candidata ao Artigo 3.
- **06/10/2026** — escopo cortado para a versão enxuta: **um** dataset (KIT),
  **três** modelos (3σ, PCA, autoencoder LSTM), as cinco classes de falha e três
  seeds. Cortados em relação ao plano original: o autoencoder convolucional 1D, o
  segundo dataset (VED) e o teste de generalização cruzada entre datasets.
- **06/10/2026** — esqueleto do `artigo-2/` criado espelhando o Artigo 1, com o
  LaTeX compilando limpo desde o primeiro commit.

- **06/10/2026 — dataset verificado:** 81 sessões do mesmo veículo (Seat Leon,
  2017–2018), 70,0 h, 10 canais presentes em todas as sessões. Baixado pelo
  endpoint do RADAR/KIT que o DOI resolve.
- **06/10/2026 — a taxa de 11 Hz é falsa:** cada linha repete o último valor
  conhecido. Medida a taxa real de mudança por canal, o topo fica em 1,11 Hz
  (`fluxo_ar`) e 1,07 Hz (`rpm`). **O pipeline reamostra para 1 Hz.** Treinar a
  11 Hz faria o autoencoder aprender a copiar valor retido e inflaria o resultado.
- **06/10/2026 — seleção de canais:** ficam `rpm`, `velocidade`, `fluxo_ar`,
  `pressao_adm`, `temp_adm` e `pedal_d`. Saem `pedal_e` (correlação de 0,99 com
  `pedal_d`, é o par redundante do pedal), `acelerador_abs`, `temp_arref` e
  `temp_ambiente` (todos abaixo de 0,04 Hz de mudança real).
- **06/10/2026 — segmentação:** sessões cortadas em lacunas maiores que 2 s;
  11 sessões têm lacuna, a maior de 603,9 s. Resultam 85 segmentos contínuos e
  **23.073 janelas** de 60 s com passo de 10 s. Tensor de 33,2 MB — roda em CPU.
- **06/10/2026 — sessões especiais:** `Messfehler`, `Vollbremsung`, `Glatteis` e
  `Beschleunigung` ficam fora do treino. As três últimas são condição de direção,
  não falha. A `Messfehler` vira verificação qualitativa na Discussão.

- **06/10/2026 — contrato do tensor (#27):** fechado em
  `X float32 (n_janelas, 60, 6)` mais `session (n_janelas,)` com o nome da sessão
  de origem, canais na ordem `rpm, velocidade, fluxo_ar, pressao_adm, temp_adm,
  pedal_d`. É o que as issues de modelo consomem.
- **06/10/2026 — o split é fixo entre as seeds.** `data.split_sessions` usa
  `DataConfig.split_seed`, uma semente própria e constante, independente das seeds
  de treinamento. Assim a variação entre as três execuções canônicas mede variação
  de **treinamento**, e não troca das sessões avaliadas — mesmo princípio dos
  splits oficiais fixos do Artigo 1.
- **06/10/2026 — `data.py` não depende de PyTorch.** A saída é NumPy. Os modelos A
  (limiar) e B (PCA) não precisam de torch, e manter o framework fora do pipeline
  deixa o carregamento testável sem instalá-lo. O empacotamento em tensores fica
  com quem precisa deles.
- **06/10/2026 — ambiente:** `artigo-2/.venv`, espelhando `artigo-1/.venv`. Os
  testes rodam com `python -m pytest` de dentro de `artigo-2/`; é o `-m` que põe o
  diretório no `sys.path` e torna `from src import ...` importável.

- **06/10/2026 — injeção de falhas (#28):** as cinco classes implementadas em
  `src/faults.py`, operando em **unidades físicas**. Para isso `load_dataset`
  ganhou a opção `normalize=False`; o contrato publicado na #27 fica intacto,
  porque o padrão continua devolvendo os splits normalizados.
- **06/10/2026 — o argumento central está medido, não afirmado:** ganho, sensor
  travado e lacuna passam pela checagem de faixa em **100 %** das janelas; a
  deriva passa em 67,5 % e o pico em apenas 3,8 %. Nenhuma das 3.191 janelas
  normais do teste viola a faixa. Três das cinco falhas são invisíveis para uma
  regra, e o pico é justamente onde o limiar estatístico deve vencer.
- **06/10/2026 — lacuna e sensor travado são parentes por construção.** Depois da
  reamostragem, perda de pacote vira valor repetido, que é o mesmo efeito de um
  sensor congelado; o que as separa é a duração. Lacunas acima de `gap_s`
  encerram o segmento no `data.py` e nunca aparecem dentro de uma janela. Essa
  dependência precisa ser dita na Discussão, não escondida.
- **06/10/2026 — 20 % de janelas corrompidas** no conjunto de avaliação, com as
  cinco classes em rodízio. A fração é baixa de propósito: é o desbalanceamento
  que justifica o AUC-PR, e inflá-lo deixaria o problema artificialmente fácil.

- **06/10/2026 — Modelo A implementado (#31).** `src/models/base.py` fixa a
  interface comum aos três detectores e `src/models/baseline.py` traz o limiar.
  Todo detector produz escore **por amostra**, não só por janela: sem isso a
  latência de detecção da #29 não tem como ser medida.
- **06/10/2026 — agregação única.** O escore de janela é o **máximo** do escore
  por amostra, igual para os três modelos, vindo de `DetectionConfig`. Detecção é
  sobre o pior instante, e isso mantém o escore de janela coerente com a latência.
- **06/10/2026 — a regra nominal de 3σ é inutilizável como enunciada.** Ela
  dispara em ~62 % das janelas **normais**: cada janela traz 60 amostras × 6
  canais = 360 leituras, e `1 − (1 − 0,0027)^360 ≈ 0,62`. É comparação múltipla,
  medida e confirmada em teste. Rende uma frase na Discussão: a regra ingênua não
  é só cega para falha silenciosa, ela é impraticável no ponto de operação que lhe
  dá nome. Isso **não** distorce a comparação, porque o AUC-PR é baseado em
  ordenação e o limiar da avaliação vem da calibração na validação normal (#30).
- **06/10/2026 — sondagem do Modelo A no teste real** (seed 42, execução única,
  **não canônica** — os números do artigo virão da grade da #34). Fração de
  janelas de cada classe que pontuam acima do percentil 95 do normal:

  | Classe | Escore médio | Acima do p95 normal |
  | ------ | -----------: | ------------------: |
  | normal | 2,45 | — |
  | pico | 6,72 | **99,2 %** |
  | ganho | 4,71 | 27,3 % |
  | lacuna | 2,45 | 6,3 % |
  | travado | 2,45 | 3,1 % |
  | deriva | 3,35 | 2,3 % |

  AUC-PR global de 0,5131 sobre taxa-base de 20 %. O limiar acerta o pico quase
  sempre e fica **no nível do acaso** em travado, lacuna e deriva — três das cinco
  classes. É contra esse piso que o autoencoder precisa mostrar valor.

- **07/10/2026 — Metodologia escrita (#37).** Ocupa **0,95 página** medida no PDF
  compilado, dentro do orçamento de 1,0. O `\nocite{*}` temporário do esqueleto foi
  removido, já que existem citações reais; o conjunto de dados entrou no `refs.bib`
  como `weber2023automotive`, com DOI.
- **07/10/2026 — arquitetura dos modelos fechada (#32, #33):** PCA e autoencoder
  usam dimensão latente 16. O autoencoder tem codificador LSTM de uma camada com
  32 unidades, projeção linear para o gargalo, repetição do latente por 60 passos,
  decodificador LSTM de uma camada com 32 unidades e projeção linear para os seis
  canais, totalizando 12.246 parâmetros. O ajuste usa MSE, AdamW com taxa de
  aprendizado `1e-3`, decaimento de pesos `1e-4` e lotes de 64 janelas, por até
  30 épocas. A parada ocorre após cinco épocas sem redução mínima de `1e-4` na
  perda da validação normal, restaurando-se os pesos da melhor época.
- **07/10/2026 — orçamento por palavras não serve.** A estimativa de ~1000 palavras
  por página estava errada: o IEEE em duas colunas comporta cerca de 1240. Medir o
  espaço no PDF compilado, não contar palavras.

- **08/10/2026 — bug corrigido na injeção de falhas.** A asserção de
  `metrics.py`, escrita na #29, expôs um defeito do `faults.py` (#28): **12 de
  638 janelas de teste (1,9 %)** eram rotuladas como anômalas sem terem nenhuma
  amostra alterada. Congelar um canal que já estava constante, ou dobrar um valor
  que vale zero, devolve a janela intacta. Por classe: lacuna 6,3 %, travado
  2,3 %, ganho 0,8 %. Um positivo byte a byte idêntico a um negativo é impossível
  de acertar por qualquer detector e seria indefensável na Metodologia.
  `build_evaluation_set` passa a descartar a injeção sem efeito e sortear outra
  janela, mantendo exata a contagem de anomalias e o rodízio entre as classes.
- **08/10/2026 — sondagem do limiar após a correção** (seed 42, não canônica):
  AUC-PR global de 0,5085; por classe, pico 0,9119, ganho 0,2800, deriva 0,1132,
  travado 0,0504 e lacuna 0,0460. Confirma o piso: o limiar resolve o pico e é
  inútil em travado, lacuna e deriva.

- **08/10/2026 — grade canônica executada (#34), nas DUAS agregações.** 3 modelos
  × 3 seeds × 2 agregações = 18 JSONs. O `run.py` ganhou `--aggregation` e passou a
  gravar `window_aggregation` em cada JSON; sem isso as duas grades ficariam
  indistinguíveis no disco. Local, em CPU; o autoencoder custa ~200 s por execução.

  | Modelo | AUC-PR (máximo) | AUC-PR (média) |
  | ------ | --------------: | -------------: |
  | limiar | 0,5009 ± 0,0066 | 0,3814 ± 0,0102 |
  | PCA | 0,4833 ± 0,0053 | 0,5067 ± 0,0094 |
  | autoencoder | **0,5109 ± 0,0067** | **0,5634 ± 0,0172** |

- **08/10/2026 — a agregação inverte o veredito.** Sob o máximo os três praticamente
  empatam e a tabela é dominada pelo pico. Sob a média o autoencoder abre 48 % sobre
  a linha de base. O máximo favorece falha pontual por construção; a média favorece
  falha sustentada. Nenhuma das duas é "certa" — elas medem coisas diferentes, e
  isso precisa estar na Discussão, não escondido atrás de uma escolha.
- **08/10/2026 — onde o autoencoder realmente ganha:** deriva sob média, 0,3511
  contra 0,0670 do PCA e 0,1268 do limiar. É a não linearidade e a memória temporal
  aparecendo exatamente onde o plano previa.
- **08/10/2026 — travado e lacuna ficam no nível do acaso nos três modelos e nas
  duas agregações.** Provável causa: foram injetadas em `velocidade` e `fluxo_ar`,
  canais que passam longos trechos constantes em operação normal. Um valor
  congelado num canal que já costuma ficar congelado é indistinguível do normal.
  Limitação a declarar.
- **08/10/2026 — o autoencoder está subtreinado.** A melhor época foi 29, 28 e 30 de
  um teto de 30; a parada antecipada nunca disparou e a perda de validação ainda
  caía. Limitação a declarar; elevar o teto seria ajuste após ver o resultado.

- **08/10/2026 — relatório implementado (#35).** `report.py` lê as duas grades,
  grava `resumo.csv`, emite `paper/results-generated.tex` com todas as macros e
  gera a figura de AUC-PR por classe com um painel por agregação. O `main.tex`
  passou a importar as macros e a inserir a tabela por `\ResultTable`; nenhum
  número está digitado no texto. Artigo em **3 páginas**, zero citação indefinida,
  zero estouro de margem.
- **08/10/2026 — coluna de custo usa `parameters.total`, não `trainable`.** O
  limiar e o PCA são ajustados em forma fechada e têm zero parâmetros treinados por
  gradiente, mas carregam 12 e 6.120 valores aprendidos. `trainable` os faria
  aparecer como gratuitos na tabela, o que é falso.
- **08/10/2026 — pendência de procedência:** os 18 JSONs foram gravados com
  `git_dirty: true`, porque o `run.py` ainda tinha alterações não commitadas quando
  a grade rodou. O commit registrado (`8a54b39`) não descreve exatamente o código
  executado. Reexecutar com a árvore limpa custa ~25 min e deixa a procedência
  defensável.

- **08/10/2026 — Discussão e Limitações escrita (#39).** Ocupa 0,57 página medida
  no PDF, contra orçamento de 0,5. Quatro blocos: a agregação decide o veredito,
  onde o autoencoder se separa (deriva) e onde perde (pico), o que o experimento não
  resolve (travado e lacuna), e as limitações. Todo número vem de macro.
- **08/10/2026 — pressão de página.** Com Resumo, Introdução, Trabalhos
  Relacionados e Conclusão ainda vazios, o artigo está em 3 páginas. As seções
  faltantes somam cerca de 1,9 página pelo orçamento, o que levaria a ~4,9 — acima
  do limite de 4. Alguma coisa terá de encolher na revisão final (#42).

- **08/10/2026 — grade canônica regerada com procedência limpa.** Os 18 JSONs
  passam a registrar `git_dirty: false` no commit `3b341dd`. A causa das três
  tentativas anteriores falharem não era o que parecia: `src/results/` é versionado
  e a própria execução o reescreve, então o primeiro JSON gravado sujava a árvore e
  todos os seguintes herdavam a marca. A checagem passou a cobrir só `src/`,
  excluindo `results`, `results-mean` e `data`.
- **08/10/2026 — reprodutibilidade confirmada.** Entre a grade anterior e a
  regerada, mudaram apenas `git_commit`, `git_dirty` e `elapsed_seconds`. **Todas as
  métricas saíram bit a bit idênticas**, nos dezoito arquivos. Vale uma frase na
  Metodologia ou na Discussão.

## O que NÃO pode cair, por mais que aperte

1. A linha de base 3σ — sem ela o resultado do autoencoder não significa nada.
2. As três seeds com média e desvio.
3. O resultado **por classe de falha** — é o miolo da Discussão.
4. A seção de limitações escrita a sério.

## Checklist antes de entregar

- [ ] Cabe em 4 páginas com as referências dentro
- [ ] O `\nocite{*}` temporário foi removido do `main.tex`
- [ ] Nenhum texto-guia ou comentário de orientação sobrou visível no PDF
- [ ] Dataset citado formalmente em `refs.bib`, com DOI
- [ ] Todas as citações do texto aparecem nas referências (sem `[?]` no PDF)
- [ ] AUC-PR justificada explicitamente no texto
- [ ] Seção de limitações escrita a sério, não uma frase protocolar
- [ ] Nenhum número do artigo digitado à mão — todos vêm de `src/results/`
- [ ] Nomes, matrículas e e-mails dos dois integrantes corretos no `\author{}`
- [ ] Os **dois** subiram o PDF no Teams, cada um na pasta da própria matrícula
