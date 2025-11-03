import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# === 1. Carregar CSV ===
df = pd.read_csv(os.path.join("data_processed", "best_features.csv"))

# Limpeza: remove constantes e não numéricas
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

if "Class" not in df.columns:
    raise ValueError("Coluna 'Class' não encontrada.")

X = df.drop(columns=["Class"])
y = df["Class"]

# === 2. Definir modelos ===
modelos = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=43))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(random_state=43))
    ]),
    "SVC_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=43))
    ]),
    "SVC_poly": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=3, probability=True, random_state=43))
    ]),
    "XGBoost": Pipeline([
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=43))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "GradientBoosting": Pipeline([
        ("clf", GradientBoostingClassifier(random_state=43))
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=500, random_state=43))
    ])
}

# === 3. Avaliar todos os modelos para random_state 0 a 100 ===
resultados_finais = []

random_states = range(0, 101)

for nome, modelo in modelos.items():
    resultados = []
    for rs in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=rs, stratify=y
        )

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo.named_steps["clf"], "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        resultados.append([acc, f1, prec, rec, auc, tn, fp, fn, tp])


    # Média das métricas
    medias = np.nanmean(resultados, axis=0)
    resultados_finais.append([nome] + list(medias))

# === 4. DataFrame com médias ===
df_medias = pd.DataFrame(resultados_finais, columns=["Modelo", "Accuracy", "F1", "Precision", "Recall", "AUC", "TN", "FP", "FN", "TP"])

print("\n=== Médias das métricas para random_state 0 a 100 ===")
print(df_medias)
