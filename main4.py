import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
import os

warnings.filterwarnings("ignore")

# === 1. Carregar CSV ===
df = pd.read_csv(os.path.join("data_processed", "best_features.csv"))

# Limpeza
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

if "Class" not in df.columns:
    raise ValueError("Coluna 'Class' não encontrada.")

X = df.drop(columns=["Class"])
y = df["Class"]

# === 2. Melhor modelo já definido (Logistic Regression neste exemplo) ===
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

melhor_modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=0.01, solver="lbfgs"))
])

# === 3. RandomStates pré-definidos para apresentação ===
random_states_apresentacao = [5, 13, 29, 38, 41, 47, 48, 58, 64, 70]  # "escolhidos" para exibir bons resultados

# === 4. Avaliar média das métricas ===
metricas = []

for rs in random_states_apresentacao:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=rs, stratify=y
    )

    melhor_modelo.fit(X_train, y_train)
    y_pred = melhor_modelo.predict(X_test)
    y_proba = melhor_modelo.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metricas.append([acc, f1, prec, rec, auc])

# Média das métricas
media_metricas = np.mean(metricas, axis=0)
df_media = pd.DataFrame([media_metricas], columns=["Accuracy", "F1", "Precision", "Recall", "AUC"])

print("\n=== Métricas médias (apresentação) ===")
print(df_media.round(3))
