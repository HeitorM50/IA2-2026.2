# Guia dos artigos de IA II

Complemento ao `CLAUDE.md` da raiz. Aqui ficam as ideias concretas, as fontes de
dados e os detalhes que não cabem no contexto curto.

Versão visual deste guia:
<https://claude.ai/code/artifact/b1b99f98-46ea-4111-be94-97f93be00ec9>

---

## Ideias concretas por rota

### Rota I — Comparação entre algoritmos

A contribuição é o veredito comparativo. Todos os modelos precisam do mesmo split,
da mesma seed e do mesmo pré-processamento, senão a comparação não vale.

- CNN treinada do zero vs. ResNet50 com transfer learning vs. MLP, em um dataset
  de imagens de porte médio (CIFAR-10, Fashion-MNIST, ou algo de nicho no Kaggle).
  Reportar acurácia, F1 macro, tempo de treino e número de parâmetros.
- Gradient boosting vs. MLP vs. regressão logística em dados tabulares, discutindo
  quando a rede neural realmente compensa.
- Comparar estratégias de regularização (dropout, weight decay, data augmentation)
  na mesma arquitetura, medindo o gap entre treino e validação.

### Rota II — Aplicação a um problema proposto

A contribuição é a solução. Exige pelo menos uma linha de base ingênua para o
resultado significar alguma coisa.

- LSTM para previsão de série temporal (consumo energético, qualidade do ar,
  demanda), contra a baseline de persistência (prever o valor anterior).
- Fine-tuning de um modelo pequeno da Hugging Face (BERTimbau, DistilBERT) para
  classificação de texto em português, contra TF-IDF + regressão logística.
- Autoencoder para detecção de anomalia, usando o erro de reconstrução como escore.

### Rota III — Explicabilidade (XAI)

Você treina o modelo e depois investiga por que ele decide o que decide. O achado
mais forte é mostrar que o modelo acerta pelo motivo errado.

- Grad-CAM sobre uma CNN de imagens médicas, verificando se a atenção cai na
  região relevante ou em artefatos do equipamento.
- SHAP sobre um modelo tabular, comparando a importância global das features com
  o que o domínio diz que deveria importar.
- LIME vs. SHAP no mesmo modelo: as duas técnicas concordam? Onde divergem?

### Rota IV — Viés e equidade (fairness)

Precisa de um dataset com atributo de grupo disponível.

- Adult/Census Income, COMPAS ou German Credit: medir demographic parity e
  equalized odds, aplicar reweighting ou threshold ajustado por grupo, e medir
  o custo em acurácia global.
- Desempenho desagregado por subgrupo em um modelo de visão, verificando se a
  taxa de erro é uniforme.

---

## Onde procurar o repositório público de dados

| Fonte | Bom para |
| ----- | -------- |
| UCI Machine Learning Repository | tabular clássico, citável, pequeno |
| Kaggle Datasets | variedade e nicho; confira a licença |
| Hugging Face Datasets | texto e PLN, carregamento em uma linha |
| OpenML | tabular com splits e benchmarks já definidos |
| Papers with Code | achar o dataset a partir do artigo |
| Zenodo | dados de pesquisa com DOI (ótimo para citar) |
| PhysioNet | sinais biomédicos, ECG, EEG |
| dados.gov.br | dados públicos brasileiros |
| Google Dataset Search | busca federada quando nada acima serve |
| `torchvision.datasets` | imagens padrão, download automático |

Prefira fontes com DOI ou citação sugerida — facilita a entrada em `refs.bib` e
mostra rigor na correção.

---

## Erros que mais custam nota

1. **Acurácia em base desbalanceada.** Se 95% das amostras são de uma classe, um
   modelo que chuta sempre essa classe tem 95% de acurácia e é inútil. Use F1,
   AUC-ROC ou acurácia balanceada, e diga por quê.
2. **Sem linha de base.** "O modelo atingiu 0,87" não significa nada sozinho.
3. **Vazamento de dados.** Normalizar antes de dividir treino e teste, ou escolher
   hiperparâmetros olhando o teste.
4. **Execução única sem desvio.** Se der tempo, rode três seeds e reporte média
   e desvio. Se não der, declare isso nas limitações.
5. **Discussão que só repete a tabela.** A seção precisa explicar o porquê e
   apontar onde o modelo falha.
6. **Estourar as 4 páginas** e tentar consertar com `\vspace` negativo. Corte texto.
7. **Deixar texto-guia do template** no PDF final.

---

## Como escrever em dupla sem conflito no git

- Uma frase por linha no `.tex`. Sem isso, qualquer edição no mesmo parágrafo vira
  conflito de merge.
- Cada pessoa em uma branch por seção: `feat/metodologia`, `feat/resultados`.
- Faça `git pull --rebase` antes de começar a sessão de escrita.
- Nunca commitar `main.pdf` nem arquivos `.aux` — já estão no `.gitignore`.
- Se der conflito no `.bbl`, apague o arquivo e recompile; ele é gerado.
