# Artigo 2 — decisões e andamento

**Prazo: 09/10/2026** · vale 25% da nota final · entrega pelo Teams.

> ⚠️ **Este artigo começou em 06/10, com 3 dias de prazo.** O escopo abaixo já é a
> versão enxuta. Qualquer ampliação precisa ser decidida contra o relógio, não
> contra o que seria ideal.

## Dupla

| Integrante | GitHub | Matrícula | Responsabilidade |
| ---------- | ------ | --------- | ---------------- |
| Heitor Macêdo Ricardo | `HeitorM50` | 241039073 | a definir |
| Gustavo Xavier Evangelista | `guxvr` | 241025247 | a definir |

A divisão em fatia vertical funcionou no Artigo 1 (cada um leva modelos **e**
seções) e deve ser repetida. **Fechar a divisão é a primeira tarefa.**

## Decisões do experimento

- [x] **Rota temática** — **II, aplicação de algoritmos a um problema proposto**.
      A contribuição é a solução; o peso cai na Metodologia.
- [x] **Tema distinto do Artigo 1** — o Artigo 1 foi classificação supervisionada
      de imagens (BloodMNIST, Rota I). Este é série temporal multivariada com
      detecção **não supervisionada**. Exigência do item 6 do plano de ensino
      satisfeita com folga.
- [x] **Janela de ementa** — até 09/10 o curso cobriu RNN/LSTM, autoencoders,
      VAE/GAN e Transformers. O autoencoder recorrente cai no centro.
- [ ] **Conjunto de dados** — Automotive OBD-II (KIT), DOI `10.35097/1130`,
      CC BY 4.0, 11,6 MB. **Pendente de verificação prática** (taxa de amostragem,
      continuidade, quanto há de operação normal contínua).
- [ ] **Modelos** — limiar 3σ, PCA e autoencoder LSTM.
- [x] **Métrica principal** — AUC-PR, pelo desbalanceamento extremo entre janelas
      normais e anômalas. Secundárias: F1 no limiar calibrado, precisão, revocação
      e **latência de detecção**.
- [x] **Seeds** — `42`, `1337`, `2026`, as mesmas do Artigo 1.
- [ ] **Onde treinar** — provavelmente local; o modelo é pequeno. Confirmar.

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
