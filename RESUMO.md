# Avaliação Sistemas Inteligentes Avançados - A1 Prática

Este repositório contém a implementação das três questões da avaliação.

## Estrutura do Projeto

- `questao1/`: Análise de registros clínicos de insuficiência cardíaca.
- `questao2/`: Classificação de qualidade de vinhos (tintos e brancos).
- `questao3/`: Predição de categorias, métodos de pagamento e faixa etária em vendas da Black Friday.

## Como Executar

Certifique-se de ter as bibliotecas `pandas`, `numpy` e `scikit-learn` instaladas.

```bash
# Questão 1
cd questao1
python questao1.py

# Questão 2
cd questao2
python questao2.py

# Questão 3
cd questao3
python questao3.py
```

## Observações Técnicas

### Questão 1
- **Algoritmo:** RandomForestClassifier.
- **Por que escolhi:** É um modelo que aguenta o tranco. Como a base mistura números e categorias binárias, ele resolve bem o problema sem precisar de muito "malabarismo" no código. Além disso, é mais confiável para evitar erros em dados médicos.
- **Pré-processamento:** Rodei o StandardScaler apenas nas colunas que não eram binárias.

### Questão 2
- **Vencedor:** Random Forest.
- **Por que esse:** No teste de "combate" entre três modelos, o Random Forest ganhou no F1-Score. Ele se deu melhor em classificar vinhos de qualidades diferentes, mesmo com a base tendo muito mais vinhos comuns do que raros.
- **Pipeline:** Juntei as duas tabelas (vinhos tintos e brancos) e coloquei um "scaler" pra ninguém ter vantagem.

### Questão 3
- **Targets:** Product Category, Payment Method (Sintético) e Age Group.
- **Métricas:** Além da acurácia global e F1-Score, foram calculadas Sensibilidade e Especificidade médias (macro) para cada modelo.
- **Inferência:** Demonstra a predição conjunta com grau de certeza (probabilidade).
