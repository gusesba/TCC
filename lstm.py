import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# -------------------------
# Configurações
# -------------------------
DATA_CSV = "./data_processed/dados_filtrados.csv"
AUDIO_DIR = "./data_processed/segments/10s"
SAMPLE_RATE = 200
N_MFCC = 40
MAX_LEN = 40  # time_steps fixo para padding

# -------------------------
# Função para extrair MFCC padronizado
# -------------------------
def extract_mfcc(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # shape: (time_steps, n_mfcc)

    # Padding / truncation
    if mfcc.shape[0] > max_len:
        mfcc = mfcc[:max_len, :]
    else:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
    return mfcc

# -------------------------
# Carregar CSV
# -------------------------
df = pd.read_csv(DATA_CSV)

# Normalizar variáveis contínuas
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
df[["W", "H", "age"]] = scaler.fit_transform(df[["W", "H", "age"]])

# Codificar sexo (0 = feminino, 1 = masculino, por ex.)
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])

# Saída (classe 0/1)
y = df["Class"].values

# Features tabulares
X_tab = df[["W", "H", "age", "gender"]].values

# -------------------------
# Carregar os áudios
# -------------------------
X_audio = []

for i in range(1, len(df)+1):  # assumindo 1..78
    file_path = os.path.join(AUDIO_DIR, str(i), "segment_3.wav")
    mfcc = extract_mfcc(file_path)
    X_audio.append(mfcc)

X_audio = np.array(X_audio)

print("X_audio shape:", X_audio.shape)  # (n_samples, MAX_LEN, N_MFCC)
print("X_tab shape:", X_tab.shape)      # (n_samples, 4)
print("y shape:", y.shape)              # (n_samples,)

# -------------------------
# Definir modelo multimodal
# -------------------------
time_steps = MAX_LEN
n_features = N_MFCC

# Branch áudio
audio_input = Input(shape=(time_steps, n_features))
x = layers.LSTM(128, return_sequences=True)(audio_input)
x = layers.LSTM(64)(x)
x = layers.Dense(64, activation="relu")(x)

# Branch tabular
tabular_input = Input(shape=(4,))
y_in = layers.Dense(32, activation="relu")(tabular_input)
y_in = layers.Dense(16, activation="relu")(y_in)

# Combinação
combined = layers.concatenate([x, y_in])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dropout(0.3)(z)
output = layers.Dense(1, activation="sigmoid")(z)

model = models.Model(inputs=[audio_input, tabular_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Treinamento
# -------------------------
history = model.fit(
    [X_audio, X_tab],
    y,
    validation_split=0.2,
    epochs=30,
    batch_size=16
)
