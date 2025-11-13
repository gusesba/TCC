import os
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from itertools import combinations
from queue import PriorityQueue

# Carrega o CSV
df = pd.read_csv(os.path.join("data_processed", "dados_completos_divididos_seg3.csv"))

#cols_to_remove = ["W", "H", "age", "gender"]
#df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])


# Remove colunas constantes (sem variabilidade)
df = df.loc[:, df.nunique() > 1]

# Remove colunas não numéricas (ex: string, boolean, category)
df = df.select_dtypes(include=[np.number])

# Separa features e alvo
if "Class" not in df.columns:
    raise ValueError("A coluna 'Class' não foi encontrada no dataset.")

X = df.drop(columns=["Class"])
y = df["Class"]

# Correlação feature-classe
def compute_feature_class_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pointbiserialr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

# Correlação entre features
def compute_feature_feature_correlation(X, subset):
    if len(subset) <= 1:
        return 0.0
    corr_matrix = X[subset].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_tri.stack().mean()

# Mérito do subconjunto
def get_merit(X, y, subset, feature_class_corrs):
    k = len(subset)
    if k == 0:
        return 0
    r_cf = np.mean([feature_class_corrs[feat] for feat in subset])
    r_ff = compute_feature_feature_correlation(X, subset)
    return (k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff)

# Busca best-first
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

# Executa
selected_features, final_merit = best_first_search(X, y)

print("Melhores features:", selected_features)
print("Mérito final:", final_merit)

df = df[selected_features + ["Class"]]
df.to_csv(os.path.join("data_processed", "best_features_divididos.csv"), index=False)