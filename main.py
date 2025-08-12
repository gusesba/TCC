import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
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

# === 2. Divisão treino/teste (fixa e estratificada) === 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2, stratify=y
)

# === 2.1 Normalização manual para RNN/LSTM ===
scaler_rnn = StandardScaler()
X_train_rnn = scaler_rnn.fit_transform(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = scaler_rnn.transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# === 3. Funções para criar modelos RNN/LSTM ===
def criar_rnn(input_dim):
    model = Sequential()
    model.add(SimpleRNN(16, activation="relu", input_shape=(input_dim, 1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def criar_lstm(input_dim):
    model = Sequential()
    model.add(LSTM(16, activation="tanh", input_shape=(input_dim, 1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# === 4. Define modelos ===
modelos = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ]),
    "SVC_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "SVC_poly": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=3, probability=True, random_state=42))
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
    ]),
    "RNN": KerasClassifier(model=criar_rnn, model__input_dim=X_train.shape[1],
                           epochs=20, batch_size=16, verbose=0),
    "LSTM": KerasClassifier(model=criar_lstm, model__input_dim=X_train.shape[1],
                            epochs=20, batch_size=16, verbose=0)
}

# === 5. Avaliação dos modelos ===
resultados = {}

print("\n=== Avaliação inicial ===")
for nome, pipeline in modelos.items():
    if nome in ["RNN", "LSTM"]:
        pipeline.fit(X_train_rnn, y_train)
        y_pred = (pipeline.predict(X_test_rnn) > 0.5).astype(int)
        y_proba = pipeline.predict_proba(X_test_rnn)[:, 1]
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["clf"], "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    resultados[nome] = {
        "modelo": pipeline if nome in ["RNN", "LSTM"] else pipeline.named_steps["clf"],
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "fp": fp,
        "fn": fn
    }

    print(f"{nome}: acc={acc:.4f} | f1={f1:.4f} | prec={prec:.4f} | recall={rec:.4f} "
          f"| FP={fp} | FN={fn} | auc={auc:.4f}" if auc else
          f"{nome}: acc={acc:.4f} | f1={f1:.4f} | prec={prec:.4f} | recall={rec:.4f} "
          f"| FP={fp} | FN={fn} | auc=N/A")
