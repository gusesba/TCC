import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# caminho do arquivo
file_path = "9979_AV.wav"

# leitura do áudio
rate, data = wav.read(file_path)

# se for estéreo, pega apenas um canal
if len(data.shape) > 1:
    data = data[:, 0]

# normalização
data = data / np.max(np.abs(data))

# eixo do tempo
t = np.linspace(0, len(data) / rate, num=len(data))

# plot
plt.figure(figsize=(12, 5))
plt.plot(t, data, linewidth=0.8)
plt.title("Sinal no tempo - segment_3.wav")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude normalizada")
plt.grid(True)
plt.show()
