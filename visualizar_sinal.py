import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# === Configurações ===
arquivo = "2530_AV.wav"  # troque pelo nome do arquivo desejado

# === Leitura do áudio ===
fs, sinal = wav.read(arquivo)

# Se for estéreo, pega apenas um canal
if len(sinal.shape) > 1:
    sinal = sinal[:, 0]

# Normaliza para -1 a 1 (se os dados forem inteiros)
if sinal.dtype != np.float32 and sinal.dtype != np.float64:
    sinal = sinal / np.max(np.abs(sinal))

# Cria eixo do tempo
tempo = np.linspace(0, len(sinal) / fs, num=len(sinal))

# === Plot ===
plt.figure(figsize=(15, 3))
plt.plot(tempo, sinal, color="black", linewidth=0.8)
plt.title("Sinal de Áudio no Tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude Normalizada")

# Definir ticks do eixo X (200ms = 0.2s)
plt.xticks(np.arange(0, max(tempo), 1))

# Definir ticks do eixo Y (0.2 em 0.2)
plt.yticks(np.arange(-1, 1.1, 0.2))

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
