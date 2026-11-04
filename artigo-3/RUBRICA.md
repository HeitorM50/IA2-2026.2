# Rubrica da avaliação humana — Artigo 3 (#52)

Cada autor preenche a própria cópia da planilha cega (`data/avaliacao/planilha-cega.csv`)
**sem abrir** `chave.csv`, sem conversar sobre itens e sem olhar a planilha do outro.
A planilha não mostra a configuração de chunking; não tente deduzi-la pelo tamanho dos trechos.

Para cada item, compare a **resposta** com o **contexto** recuperado e com a **resposta de referência**.

## Critérios

| Critério | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `correcao` | contradiz a referência ou responde outra coisa | parcialmente certa, ou certa com erro relevante | equivalente à referência |
| `completude` | cobre menos da metade dos pontos da referência | cobre parte dos pontos | cobre todos os pontos essenciais |
| `fidelidade` | a ideia central não tem apoio no contexto | há apoio para a ideia central, mas não para detalhes | toda afirmação tem apoio no contexto |

`afirmacoes_nao_sustentadas`: número de afirmações factuais **sem apoio no contexto
recuperado** (0, 1, 2, ou 3 para três ou mais).

## Regras operacionais

1. **Paráfrase não é alucinação.** Uma afirmação dita com outras palavras, mas com o
   mesmo conteúdo de um trecho, é sustentada.
2. **Sustentação é contra o contexto, não contra o mundo.** Uma afirmação verdadeira
   que não está nos trechos conta como não sustentada (o sistema foi instruído a usar
   só o contexto). Isso é registrado; a correção factual é avaliada à parte em `correcao`.
3. **Citação errada:** se a afirmação está no contexto, mas cita o trecho errado,
   ela é sustentada; anote em `observacao`.
4. **Abstenção:** se a resposta é o texto de abstenção, preencha `correcao = 0`,
   `completude = 0`, `fidelidade = 2` e `afirmacoes_nao_sustentadas = 0`. A abstenção
   é detectada automaticamente e entra na métrica de abstenção, não na nota.
5. **Pergunta não respondível:** a referência é "não respondível". Qualquer resposta
   substantiva recebe `correcao = 0`.
6. Na dúvida entre duas notas, escolha a menor e explique em `observacao`.

## Consenso

Depois das duas avaliações:

```sh
.venv/bin/python -m src.avaliacao_humana concordancia av-heitor.csv av-gustavo.csv
```

O kappa de Cohen com pesos quadráticos (por critério) vai para o artigo **antes** do
consenso. Os itens divergentes são discutidos e a nota de consenso é a que entra nas
métricas de resposta.
