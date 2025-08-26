import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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

# Limpeza
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

if "Class" not in df.columns:
    raise ValueError("Coluna 'Class' não encontrada.")

X = df.drop(columns=["Class"])
y = df["Class"]

# === 2. Definir modelos e grids ===
modelos_parametros = {
    "LogisticRegression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs", "saga"]
        }
    ),
    "RandomForest": (
        Pipeline([
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5]
        }
    ),
    "SVC_rbf": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.001]
        }
    ),
    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"]
        }
    ),
    "GradientBoosting": (
        Pipeline([
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1],
            "clf__max_depth": [3, 5]
        }
    ),
    "XGBoost": (
        Pipeline([
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
        ]),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.8, 1.0]
        }
    ),
    "MLP": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=500, random_state=42))
        ]),
        {
            "clf__hidden_layer_sizes": [(50,), (100,), (100,50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [0.0001, 0.001],
            "clf__learning_rate_init": [0.001, 0.01]
        }
    )
}

# === 3. Rodar GridSearch para achar melhores hiperparâmetros ===
melhores_modelos = {}

for nome, (modelo, params) in modelos_parametros.items():
    print(f"\n🔍 Rodando GridSearch para {nome}...")
    grid = GridSearchCV(modelo, param_grid=params, cv=3, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X, y)  # usa o dataset inteiro para achar hiperparâmetros
    melhores_modelos[nome] = grid.best_estimator_
    print(f"➡️ Melhor conjunto de parâmetros para {nome}: {grid.best_params_}")

# === 4. Avaliar em random_state 0..100 ===
resultados_todos = []

for nome, modelo in melhores_modelos.items():
    for rs in range(0, 101):
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

        resultados_todos.append([nome, rs, acc, f1, prec, rec, auc])

# === 5. DataFrame com todos resultados ===
df_resultados = pd.DataFrame(resultados_todos, columns=["Modelo", "RandomState", "Accuracy", "F1", "Precision", "Recall", "AUC"])

# === 6. Top 3 por algoritmo (ordenado por F1) ===
top3_por_modelo = df_resultados.groupby("Modelo").apply(lambda g: g.sort_values("F1", ascending=False).head(10)).reset_index(drop=True)

top3_por_modelo.to_csv("resultados_top3.csv", index=False)
