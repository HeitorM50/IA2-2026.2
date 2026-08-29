# Artigo 1 — decisões e andamento

**Prazo: 03/09/2026** · vale 20% da nota final · entrega pelo Teams.

## Dupla

Divisão em fatia vertical: cada um leva modelos **e** seções, para que ninguém
fique bloqueado esperando o outro e os dois conheçam o artigo inteiro na
apresentação.

| Integrante | GitHub | Matrícula | Responsabilidade |
| ---------- | ------ | --------- | ---------------- |
| Heitor Macedo | `HeitorM50` | — | pipeline de dados, linha de base, transfer learning · Metodologia, Resumo, Conclusão |
| Gustavo | `guxvr` | — | loop de treino e métricas, CNN do zero, figuras · Introdução, Trabalhos Relacionados, Resultados, `refs.bib` |

A quatro mãos: decisão do tema, Discussão e Limitações, revisão final e entrega.

O andamento fica nas issues do GitHub, no milestone `Artigo 1 — 03/09/2026`.

## Decisões do experimento

- [x] **Rota temática** — **I, comparação entre algoritmos** (decidido em 26/08/2026).
      O escopo é fechado (N modelos, um protocolo) e o resultado sempre existe,
      mesmo que seja "a arquitetura simples ganhou".
- [x] **Dataset público** — **BloodMNIST**, subconjunto RGB de 64 × 64 pixels da
      coleção MedMNIST+, com 17.092 imagens microscópicas de oito tipos de células
      sanguíneas, splits oficiais e distribuição pública no Zenodo com DOI.
- [x] **Modelos a comparar** — regressão logística multinomial, CNN compacta
      treinada do zero e ResNet18 pré-treinada no ImageNet-1K com ajuste fino.
- [x] **Métrica principal** — F1 macro; acurácia, acurácia balanceada, tempo de
      treino e número de parâmetros serão métricas secundárias.
- [x] **Onde treinar** — Google Colab com GPU.
- [x] **Pergunta de pesquisa** — Em classificação multiclasse de tipos celulares
      no BloodMNIST, em que medida uma CNN compacta treinada do zero e uma ResNet18
      pré-treinada no ImageNet-1K superam uma regressão logística em F1 macro, sob
      o mesmo protocolo experimental?

## Registro de decisões

Anotar aqui, com data, cada decisão fechada. Isso vira material da seção de
Metodologia depois.

- **26/08/2026** — repositório e esqueleto LaTeX criados.
- **26/08/2026** — rota temática fechada: **Rota I**, comparação entre algoritmos.
- **26/08/2026** — divisão do trabalho em fatia vertical (tabela acima), com as
  issues abertas no GitHub e milestone vencendo 02/09, um dia antes do prazo real.
- **26/08/2026** — repositório publicado em `github.com/HeitorM50/IA2-2026.2`
  (público), com `guxvr` como colaborador.
- **29/08/2026 — conjunto de dados:** escolhido o **BloodMNIST 64 × 64**, da
  coleção MedMNIST+, com 17.092 imagens RGB de células sanguíneas em oito classes
  e splits oficiais de treino, validação e teste (11.959/1.712/3.421). A coleção é
  pública, tem DOI `10.5281/zenodo.10519652` e é pequena o bastante para as nove
  execuções previstas. Os splits serão mantidos fixos entre modelos e seeds; as
  seeds `42`, `1337` e `2026` controlarão inicialização, ordem dos lotes e aumento
  de dados, evitando misturar variação de amostragem com variação de treinamento.
- **29/08/2026 — modelos:** serão comparados (A) **regressão logística
  multinomial**, a linha de base não neural sobre os pixels normalizados; (B) uma
  **CNN compacta treinada do zero**; e (C) uma **ResNet18 pré-treinada no
  ImageNet-1K**, com a camada classificadora substituída e ajuste fino de toda a
  rede. Os três receberão imagens 64 × 64 e serão avaliados pelo mesmo pipeline.
- **29/08/2026 — métrica principal:** adotado o **F1 macro**, pois as frequências
  das oito classes não são perfeitamente uniformes e cada tipo celular deve ter o
  mesmo peso no resultado; a média macro também penaliza modelos que favorecem
  classes frequentes e considera simultaneamente precisão e revocação. Acurácia,
  acurácia balanceada, tempo de treino e número de parâmetros serão secundárias.
- **29/08/2026 — ambiente:** treinamento no **Google Colab com GPU**, usando um
  único notebook/script e o mesmo ambiente para os nove pares modelo × seed.
- **29/08/2026 — pergunta de pesquisa:** **Em classificação multiclasse de tipos
  celulares no BloodMNIST, em que medida uma CNN compacta treinada do zero e uma
  ResNet18 pré-treinada no ImageNet-1K superam uma regressão logística em F1 macro,
  sob o mesmo protocolo experimental?**
- **29/08/2026 — pipeline de dados:** implementado o download automático com
  `medmnist==3.0.2`, preservando os splits oficiais. A normalização RGB é calculada
  exclusivamente sobre os 11.959 exemplos de treino: média
  `(0,796054; 0,659597; 0,696349)` e desvio padrão
  `(0,223283; 0,254630; 0,094736)`. O treino recebe rotações de até 15 graus e
  espelhamentos horizontal e vertical com probabilidade 0,5; validação e teste
  recebem somente conversão para tensor e normalização. Alterações de cor não são
  usadas porque a coloração microscópica pode carregar informação discriminativa.
- **29/08/2026 — distribuição observada:** na ordem basófilo, eosinófilo,
  eritroblasto, granulócito imaturo, linfócito, monócito, neutrófilo e plaqueta, as
  contagens foram treino `(852, 2181, 1085, 2026, 849, 993, 2330, 1643)`, validação
  `(122, 312, 155, 290, 122, 143, 333, 235)` e teste
  `(244, 624, 311, 579, 243, 284, 666, 470)`. As impressões digitais dos splits são
  invariantes entre as seeds; somente embaralhamento e aumento do treino variam.

## Checklist antes de entregar

- [ ] Cabe em 4 páginas com as referências dentro
- [ ] Nenhum texto-guia ou placeholder `[entre colchetes]` sobrou no PDF
- [ ] Dataset citado formalmente em `refs.bib`
- [ ] Todas as citações do texto aparecem nas referências (sem `[?]` no PDF)
- [ ] Métrica justificada explicitamente
- [ ] Seção de limitações escrita a sério, não uma frase protocolar
- [ ] Nomes, matrículas e e-mails dos dois integrantes corretos no `\author{}`
- [ ] Os **dois** subiram o PDF no Teams, cada um na pasta da própria matrícula
