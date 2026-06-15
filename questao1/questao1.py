import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# carregamento dos dados
dados = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# pré-processamento
X = dados.drop("DEATH_EVENT", axis=1)
y = dados["DEATH_EVENT"]

# identificando colunas numéricas e binárias
colunas_binarias = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
colunas_numericas = [col for col in X.columns if col not in colunas_binarias]

# normalizando apenas as colunas numéricas
normalizador = StandardScaler()
X[colunas_numericas] = normalizador.fit_transform(X[colunas_numericas])

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# justificativa da escolha do algoritmo
"""
Escolhi o Random Forest porque ele é o mais versátil dos modelos. Como a base tem tanto números 
contínuos quanto informações de sim/não (0/1), ele lida com isso muito bem sem precisar 
de tanto ajuste. Além disso, por usar várias árvores de decisão, ele é mais seguro contra 
overfitting e outliers, o que faz mais sentido quando estamos lidando com diagnósticos médicos.
"""

# treinamento do modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# avaliação rápida
previsoes = modelo.predict(X_teste)
print("Acurácia do modelo:", accuracy_score(y_teste, previsoes))
print("\Classificação:\n", classification_report(y_teste, previsoes))


# inferência
def realizar_inferencia(dados_paciente):
    df_paciente = pd.DataFrame([dados_paciente])
    df_paciente[colunas_numericas] = normalizador.transform(
        df_paciente[colunas_numericas]
    )
    predicao = modelo.predict(df_paciente)[0]
    probabilidade = modelo.predict_proba(df_paciente)[0]

    grupo = "Risco de Evento de Morte" if predicao == 1 else "Sem Risco Imediato"
    print(f"\nO paciente pertence ao grupo: {grupo}")
    print(f"Confiança: {probabilidade[predicao]*100:.2f}%")


# exemplo de paciente desconhecido
exemplo_paciente = {
    "age": 75.0,
    "anaemia": 0,
    "creatinine_phosphokinase": 582,
    "diabetes": 0,
    "ejection_fraction": 20,
    "high_blood_pressure": 1,
    "platelets": 265000.0,
    "serum_creatinine": 1.9,
    "serum_sodium": 130,
    "sex": 1,
    "smoking": 0,
    "time": 4,
}

realizar_inferencia(exemplo_paciente)
