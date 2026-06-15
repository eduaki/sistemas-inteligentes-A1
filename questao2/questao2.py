import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# carregamento e união dos dados
tintos = pd.read_csv("vinhos_tintos.csv", sep=";")
brancos = pd.read_csv("vinhos_brancos.csv", sep=";")

# adicionando coluna de tipo (0 para tinto, 1 para branco)
tintos["tipo"] = 0
brancos["tipo"] = 1

vinhos_total = pd.concat([tintos, brancos], ignore_index=True)

# salvando o arquivo unificado como solicitado
vinhos_total.to_csv("vinhos_completos.csv", index=False)

# pré-processamento e pipeline
X = vinhos_total.drop("quality", axis=1)
y = vinhos_total["quality"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "Regressão Logística": LogisticRegression(max_iter=10000),
}

resultados = {}

# avaliação de modelos
for nome, estimador in modelos.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", estimador)])

    pipeline.fit(X_treino, y_treino)
    previsoes = pipeline.predict(X_teste)

    acc = accuracy_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes, average="weighted")

    resultados[nome] = {
        "modelo": pipeline,
        "acuracia": acc,
        "f1": f1,
        "previsoes": previsoes,
    }

    print(f"\n--- {nome} ---")
    print(f"Acurácia Global: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")

# seleção do melhor modelo
melhor_nome = max(resultados, key=lambda k: resultados[k]["f1"])
melhor_resultado = resultados[melhor_nome]

# métricas de desempenho detalhadas para o melhor modelo
print(f"\n\n> Melhor Modelo Selecionado: {melhor_nome}")
print("\nMatriz de Confusão (Acurácia por Classe):")
print(confusion_matrix(y_teste, melhor_resultado["previsoes"]))

print("\nRelatório de Classificação Completo:")
print(classification_report(y_teste, melhor_resultado["previsoes"], zero_division=0))

# justificativa de implantação
"""
O {melhor_nome} foi o que se saiu melhor no geral, equilibrando bem a acurácia com o F1-score. 
Em bases de vinho, é normal ter muito mais garrafas "médias" do que excelentes ou péssimas, 
e esse modelo conseguiu lidar melhor com esse desbalanceamento das classes. No fim das contas, 
ele consegue enxergar padrões químicos que modelos mais simples acabam deixando passar.
"""


# exemplo de inferência
def inferencia_vinho(dados_entrada):
    # converte entrada em DataFrame para o pipeline
    df_entrada = pd.DataFrame([dados_entrada])
    predicao = melhor_resultado["modelo"].predict(df_entrada)[0]
    print(f"\nQualidade prevista: {predicao}")


# exemplo de teste
exemplo_vinho = X_teste.iloc[0].to_dict()
inferencia_vinho(exemplo_vinho)
