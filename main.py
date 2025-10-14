import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# === 2. Definir modelos e grids hardcoded ===
modelos_parametros = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        [
            {"clf__C": 0.1, "clf__solver": "lbfgs"},
            {"clf__C": 1, "clf__solver": "lbfgs"},
            {"clf__C": 10, "clf__solver": "lbfgs"},
            {"clf__C": 1, "clf__solver": "saga", "clf__penalty": "l1"},
            {"clf__C": 1, "clf__solver": "saga", "clf__penalty": "elasticnet", "clf__l1_ratio": 0.5}
        ]
    ),
    "RandomForest": (
        Pipeline([("clf", RandomForestClassifier())]),
        [
            {"clf__n_estimators": 100, "clf__max_depth": None},
            {"clf__n_estimators": 200, "clf__max_depth": None},
            {"clf__n_estimators": 300, "clf__max_depth": 10},
            {"clf__n_estimators": 300, "clf__max_depth": 5}
        ]
    ),
    "SVC_rbf": (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
        [
            {"clf__C": 0.1, "clf__gamma": "scale"},
            {"clf__C": 1, "clf__gamma": "scale"},
            {"clf__C": 10, "clf__gamma": 0.01},
            {"clf__C": 10, "clf__gamma": 0.001}
        ]
    ),
    "KNN": (
    Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    [
        # Euclidean
        {"clf__n_neighbors": 3,  "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 3,  "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 5,  "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 5,  "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 7,  "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 7,  "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 11, "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 11, "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 15, "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 15, "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 21, "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 21, "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 25, "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 25, "clf__weights": "distance", "clf__metric": "euclidean"},
        {"clf__n_neighbors": 30, "clf__weights": "uniform",  "clf__metric": "euclidean"},
        {"clf__n_neighbors": 30, "clf__weights": "distance", "clf__metric": "euclidean"},

        # Manhattan
        {"clf__n_neighbors": 3,  "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 3,  "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 5,  "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 5,  "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 7,  "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 7,  "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 11, "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 11, "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 15, "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 15, "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 21, "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 21, "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 25, "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 25, "clf__weights": "distance", "clf__metric": "manhattan"},
        {"clf__n_neighbors": 30, "clf__weights": "uniform",  "clf__metric": "manhattan"},
        {"clf__n_neighbors": 30, "clf__weights": "distance", "clf__metric": "manhattan"},

        # Minkowski (p=1 e p=2)
        {"clf__n_neighbors": 3,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 3,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 3,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 3,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 5,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 5,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 5,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 5,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 7,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 7,  "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 7,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 7,  "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 11, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 11, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 11, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 11, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 15, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 15, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 15, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 15, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 21, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 21, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 21, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 21, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 25, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 25, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 25, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 25, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},

        {"clf__n_neighbors": 30, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 30, "clf__weights": "uniform",  "clf__metric": "minkowski", "clf__p": 2},
        {"clf__n_neighbors": 30, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 1},
        {"clf__n_neighbors": 30, "clf__weights": "distance", "clf__metric": "minkowski", "clf__p": 2},
    ]
),
    "GradientBoosting": (
        Pipeline([("clf", GradientBoostingClassifier())]),
        [
            {"clf__n_estimators": 100, "clf__learning_rate": 0.01, "clf__max_depth": 3},
            {"clf__n_estimators": 200, "clf__learning_rate": 0.1, "clf__max_depth": 3},
            {"clf__n_estimators": 200, "clf__learning_rate": 0.1, "clf__max_depth": 5},
        ]
    ),
    "XGBoost": (
        Pipeline([("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))]),
        [
            {"clf__n_estimators": 100, "clf__max_depth": 3, "clf__learning_rate": 0.01, "clf__subsample": 0.8},
            {"clf__n_estimators": 200, "clf__max_depth": 5, "clf__learning_rate": 0.1, "clf__subsample": 1.0},
            {"clf__n_estimators": 200, "clf__max_depth": 7, "clf__learning_rate": 0.1, "clf__subsample": 1.0},
        ]
    ),
    "MLP": (
        Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=1000))]),
        [
            {"clf__hidden_layer_sizes": (50,), "clf__activation": "relu", "clf__alpha": 0.0001},
            {"clf__hidden_layer_sizes": (100,), "clf__activation": "relu", "clf__alpha": 0.001},
            {"clf__hidden_layer_sizes": (100,50), "clf__activation": "tanh", "clf__alpha": 0.001},
        ]
    )
}

# === 3. Rodar GridSearch hardcoded ===
melhores_modelos = {}
for nome, (modelo, param_list) in modelos_parametros.items():
    print(f"\n🔍 Testando combinações para {nome}...")
    melhor_f1 = -1
    melhor_modelo = None
    for params in param_list:
        modelo.set_params(**params)
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        f1 = f1_score(y, y_pred)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_modelo = modelo
    melhores_modelos[nome] = melhor_modelo
    print(f"➡️ Melhor F1 para {nome}: {melhor_f1}")

# === 4. Avaliar com diferentes random_states ===
resultados_todos = []
random_states = list(range(0, 101, 5))  # exemplo: 0,5,10,...100

for nome, modelo in melhores_modelos.items():
    for rs in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=rs, stratify=y
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:,1] if hasattr(modelo.named_steps["clf"], "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        resultados_todos.append([nome, rs, acc, f1, prec, rec, auc])

# === 5. DataFrame e top 3 por algoritmo ===
df_resultados = pd.DataFrame(resultados_todos, columns=["Modelo","RandomState","Accuracy","F1","Precision","Recall","AUC"])
top3_por_modelo = df_resultados.groupby("Modelo").apply(
    lambda g: g.sort_values("F1", ascending=False).head(3)
).reset_index(drop=True)

print("\n=== Top 3 Resultados de cada Algoritmo (ordenado por F1) ===")
print(top3_por_modelo)

# === Opcional: salvar em CSV ===
top3_por_modelo.to_csv("top3_resultados_por_modelo.csv", index=False)
