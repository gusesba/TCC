import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile, loadmat
from scipy.signal import resample
from runSpringerSegmentationAlgorithm import runSpringerSegmentationAlgorithm

# ========================
# Carregando arquivo de áudio
# ========================
Fs, audio_data = wavfile.read('segment_7.wav')  # altere o nome do arquivo
x = audio_data[:,0] if audio_data.ndim > 1 else audio_data  # pega apenas um canal se for estéreo

# ========================
# Resample se necessário
# ========================
target_Fs = 1000
if Fs != target_Fs:
    x = resample(x, int(len(x) * target_Fs / Fs))
    Fs = target_Fs

# ========================
# Carrega arquivos .mat do Springer
# ========================
B_matrix = loadmat('Springer_B_matrix.mat')['B_matrix']
pi_vector = loadmat('Springer_pi_vector.mat')['pi_vector'].flatten()
total_obs_distribution = loadmat('Springer_total_obs_distribution.mat')['total_obs_distribution']

# ========================
# Segmentação
# ========================
assigned_states = runSpringerSegmentationAlgorithm(x, Fs, B_matrix, pi_vector, total_obs_distribution, True)

# ========================
# Plotando o sinal e os estados
# ========================
t = np.arange(len(x)) / Fs
plt.figure(figsize=(15,4))
plt.plot(t, x, 'k', label='Sinal cardíaco')

# Cores para os estados (1=S1, 2=Systole, 3=S2, 4=Diastole)
colors = [
    (1,0,0),    # vermelho - S1
    (1,0.5,0),  # laranja - Systole
    (0,0,1),    # azul - S2
    (0,0.8,0)   # verde - Diastole
]

yl = plt.ylim()
for i in range(len(assigned_states)-1):
    plt.fill_between([t[i], t[i+1]], yl[0], yl[1],
                     color=colors[assigned_states[i]-1], alpha=0.2)

plt.plot(t, x, 'k')  # replotar o sinal por cima
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Segmentação do som cardíaco - Springer')
plt.legend(['Sinal cardíaco','S1','Systole','S2','Diastole'])
plt.grid(True)
plt.show()
