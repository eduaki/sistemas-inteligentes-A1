import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# carregamento e preparação dos dados
vendas = pd.read_csv("black_friday.csv")
vendas = vendas.sample(
    50000, random_state=42
)  # usando uma amostra para agilizar o processamento

# geração de dados pra forma de pagamento
formas_pagamento = ["Cartão de Crédito", "Boleto", "Pix", "Cartão de Débito"]
np.random.seed(42)
vendas["payment_method"] = np.random.choice(formas_pagamento, size=len(vendas))


# pre-processamento
def preparar_dados(df):
    le = LabelEncoder()
    df = df.copy()

    # colunas categóricas
    colunas_cat = ["Gender", "City_Category", "Stay_In_Current_City_Years"]
    for col in colunas_cat:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


vendas_processadas = preparar_dados(vendas)

# definindo características e alvos
caracteristicas = [
    "Gender",
    "Occupation",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Marital_Status",
]
alvos = {
    "categoria": "Product_Category_1",
    "pagamento": "payment_method",
    "idade": "Age",
}

X = vendas_processadas[caracteristicas]


# função para calcular métricas específicas (sensibilidade e especificidade)
def calcular_metricas_completas(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)

    # Sensibilidade (Recall) Global (Macro)
    sensibilidade = np.diag(cm) / np.sum(cm, axis=1)
    sensibilidade_media = np.mean(sensibilidade)

    # Especificidade Global (Macro)
    especificidades = []
    for i in range(len(cm)):
        temp_cm = np.delete(cm, i, 0)
        temp_cm = np.delete(temp_cm, i, 1)
        tn = temp_cm.sum()
        fp = cm[:, i].sum() - cm[i, i]
        especificidades.append(tn / (tn + fp))
    especificidade_media = np.mean(especificidades)

    return sensibilidade_media, especificidade_media


# transformer nos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelos_treinados = {}

# treinamento e avaliação para cada alvo
for chave, coluna_alvo in alvos.items():
    print(f"\n{'='*20} Treinando para: {coluna_alvo} {'='*20}")

    y = vendas_processadas[coluna_alvo].astype(str)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    acc = accuracy_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes, average="weighted")
    sens, espec = calcular_metricas_completas(y_teste, previsoes)

    print(f"Acurácia Global: {acc:.4f}")
    print(f"Sensibilidade (Média): {sens:.4f}")
    print(f"Especificidade (Média): {espec:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")

    modelos_treinados[chave] = {"modelo": modelo, "alvo": coluna_alvo}


# demonstração do sistema inteligente (inferência conjunta)
def sistema_inteligente_venda(dados_venda):
    # transformar dados de entrada
    df_entrada = pd.DataFrame([dados_venda])
    # mock de label encoding
    df_entrada["Gender"] = 1 if dados_venda["Gender"] == "M" else 0
    df_entrada["City_Category"] = ord(dados_venda["City_Category"]) - ord("A")
    df_entrada["Stay_In_Current_City_Years"] = int(
        dados_venda["Stay_In_Current_City_Years"].replace("+", "")
    )

    X_ent = df_entrada[caracteristicas]
    X_ent_scaled = scaler.transform(X_ent)

    print("\n> Resultado do Sistema Inteligente para a Venda:")
    for chave, m in modelos_treinados.items():
        pred = m["modelo"].predict(X_ent_scaled)[0]
        prob = np.max(m["modelo"].predict_proba(X_ent_scaled))
        print(f"- {m['alvo']}: {pred} (Certeza: {prob*100:.2f}%)")


# exemplo de circunstância de venda
exemplo_circunstancia = {
    "Gender": "F",
    "Occupation": 10,
    "City_Category": "B",
    "Stay_In_Current_City_Years": "2",
    "Marital_Status": 0,
}

sistema_inteligente_venda(exemplo_circunstancia)
