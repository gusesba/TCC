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
import os
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from itertools import combinations
from queue import PriorityQueue
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# === Funções de correlação e seleção ===

def compute_feature_class_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pointbiserialr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

def compute_feature_feature_correlation(X, subset):
    if len(subset) <= 1:
        return 0.0
    corr_matrix = X[subset].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_tri.stack().mean()

def get_merit(X, y, subset, feature_class_corrs):
    k = len(subset)
    if k == 0:
        return 0
    r_cf = np.mean([feature_class_corrs[feat] for feat in subset])
    r_ff = compute_feature_feature_correlation(X, subset)
    return (k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff)

def best_first_search(X, y, max_backtracks=5):
    feature_class_corrs = compute_feature_class_correlations(X, y)
    queue = PriorityQueue()
    best_subset = []
    best_merit = 0
    backtracks = 0

    for feat in X.columns:
        merit = get_merit(X, y, [feat], feature_class_corrs)
        queue.put((-merit, [feat]))

    while not queue.empty() and backtracks < max_backtracks:
        neg_merit, subset = queue.get()
        merit = -neg_merit

        if merit > best_merit:
            best_merit = merit
            best_subset = subset
            backtracks = 0
        else:
            backtracks += 1

        for feat in X.columns:
            if feat not in subset:
                new_subset = sorted(set(subset + [feat]))
                new_merit = get_merit(X, y, new_subset, feature_class_corrs)
                queue.put((-new_merit, new_subset))

    return best_subset, best_merit



# === 1. Carregar CSV ===
df = pd.read_csv(os.path.join("data_processed", "dados_completos_seg3_concat.csv"))

# Limpeza
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

if "Class" not in df.columns:
    raise ValueError("Coluna 'Class' não encontrada.")

X = df.drop(columns=["Class"])
y = df["Class"]


# === 2. Modelos ===
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


# === 3. Avaliação ===
random_states = range(0, 101)

resultados_finais = {m: [] for m in modelos.keys()}
historico_features = []


for rs in random_states:

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=rs, stratify=y
    )

    # Seleção de features SOMENTE no treino
    print("Random State:", rs)
    print("Selecionando features...")
    selected_features, merit = best_first_search(X_train, y_train)
    print("Features selecionadas com merito: ", merit)
    # Registro das features
    historico_features.append({
        "random_state": rs,
        "features": selected_features
    })

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    if X_train_sel.isna().any().any() or X_test_sel.isna().any().any():
        print("Corrigindo NaNs com média...")
        imp = SimpleImputer(strategy="mean")
        X_train_sel = pd.DataFrame(imp.fit_transform(X_train_sel), columns=X_train_sel.columns)
        X_test_sel = pd.DataFrame(imp.transform(X_test_sel), columns=X_test_sel.columns)

    # Rodar todos os modelos
    for nome, modelo in modelos.items():
        print("Treinando modelo:", nome)
        modelo.fit(X_train_sel, y_train)
        y_pred = modelo.predict(X_test_sel)

        if hasattr(modelo.named_steps["clf"], "predict_proba"):
            y_proba = modelo.predict_proba(X_test_sel)[:, 1]
        else:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        resultados_finais[nome].append([acc, f1, prec, rec, auc, tn, fp, fn, tp])


# === 4. Gerar tabela de médias ===
linhas = []
for nome, resultados in resultados_finais.items():
    medias = np.nanmean(resultados, axis=0)
    linhas.append([nome] + list(medias))

df_medias = pd.DataFrame(
    linhas,
    columns=["Modelo", "Accuracy", "F1", "Precision", "Recall", "AUC", "TN", "FP", "FN", "TP"]
)

print("\n=== MÉDIAS DOS MODELOS ===")
print(df_medias)


# === 5. Salvar features por random_state ===
df_features = pd.DataFrame(historico_features)
df_features.to_csv(os.path.join("data_processed", "features_por_random_state_seg3_concat.csv"), index=False)

print("\nArquivo salvo: data_processed/features_por_random_state_seg3_concat.csv")