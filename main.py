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


# Limpeza: remove constantes e não numéricas
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

if "class" not in df.columns:
    raise ValueError("Coluna 'class' não encontrada.")

X = df.drop(columns=["class"])
y = df["class"]

# === 2. Divisão treino/teste (fixa e estratificada) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === 3. Define modelos ===
modelos = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ]),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "GradientBoosting": Pipeline([
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=500, random_state=42))
    ])
}

# === 4. Avaliação dos modelos ===
resultados = {}

print("\n=== Avaliação inicial ===")
for nome, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["clf"], "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    resultados[nome] = {
        "modelo": pipeline.named_steps["clf"],
        "pipeline": pipeline,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc
    }

    print(f"{nome}: acc={acc:.4f} | f1={f1:.4f} | prec={prec:.4f} | recall={rec:.4f} | auc={auc:.4f}" if auc else
          f"{nome}: acc={acc:.4f} | f1={f1:.4f} | prec={prec:.4f} | recall={rec:.4f} | auc=N/A")

# === 5. Escolhe o melhor modelo (por F1) ===
melhor_nome = max(resultados, key=lambda x: resultados[x]["f1"])
print(f"\n>> Melhor modelo inicial: {melhor_nome}")
melhor_pipeline = resultados[melhor_nome]["pipeline"]

# === 6. Define grids para fine tuning ===
param_grids = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"]
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    },
    "SVC": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.01, 0.1]
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"]
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5]
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [0.0001, 0.001]
    }
}

# === 7. Fine tuning ===
grid = GridSearchCV(
    modelos[melhor_nome],
    param_grids[melhor_nome],
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# === 8. Avaliação final ===
y_final = grid.predict(X_test)
y_proba_final = grid.predict_proba(X_test)[:, 1] if hasattr(grid.best_estimator_.named_steps["clf"], "predict_proba") else None

acc = accuracy_score(y_test, y_final)
f1 = f1_score(y_test, y_final)
prec = precision_score(y_test, y_final)
rec = recall_score(y_test, y_final)
auc = roc_auc_score(y_test, y_proba_final) if y_proba_final is not None else None

print(f"\n=== Fine Tuning: {melhor_nome} ===")
print("Melhores parâmetros:", grid.best_params_)
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"AUC-ROC: {auc:.4f}" if auc else "AUC-ROC: N/A")