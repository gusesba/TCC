import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import pywt

# --- Etapa 1: Butterworth BPF ---
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# --- Etapa 2: DWT (db12, nível 5) ---
def wavelet_filter(signal, wavelet='db12', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    filtered_signal = pywt.waverec(coeffs, wavelet)
    return filtered_signal[:len(signal)]

# --- Etapa 3: Normalização por janelas ---
def normalize_windowed(signal, fs, win_size=1.0):
    samples_per_win = int(fs * win_size)
    norm_signal = np.zeros_like(signal)
    for start in range(0, len(signal), samples_per_win):
        end = min(start + samples_per_win, len(signal))
        window = signal[start:end]
        max_val = np.max(np.abs(window))
        if max_val > 0:
            norm_signal[start:end] = window / max_val
    return norm_signal

# --- Etapa 4: Shannon Energy ---
def shannon_energy(signal, order=3):
    x_pow = np.abs(signal ** order)
    eps = 1e-12
    energy = -x_pow * np.log(x_pow + eps)
    return energy

# --- Etapa 5: Envelope (LPF 10Hz + normalização) ---
def envelope_filter(signal, fs, cutoff=10.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    env = filtfilt(b, a, signal)
    # Normalizar (Eq.3 do paper)
    mu = np.mean(env)
    sigma = np.std(env)
    env_norm = (env - mu) / sigma if sigma > 0 else env - mu
    return env_norm

# === Pipeline completo ===
def process_pcg(filename):
    # Carregar arquivo .wav
    fs, signal = wavfile.read(filename)
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / np.iinfo(np.int16).max

    # 1) Butterworth BPF
    filtered_bpf = butter_bandpass_filter(signal, 2.0, 100.0, fs, order=2)

    # 2) DWT db12 L=5
    filtered_dwt = wavelet_filter(filtered_bpf, wavelet='db12', level=5)

    # 3) Normalização por janelas de 1s
    normalized = normalize_windowed(filtered_dwt, fs, win_size=1.0)

    # 4) Shannon Energy
    shannon = shannon_energy(normalized, order=3)

    # 5) Envelope
    envelope = envelope_filter(shannon, fs, cutoff=10.0, order=2)

    # Plot resultados
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(14, 12))

    plt.subplot(5,1,1)
    plt.plot(t, signal)
    plt.title("Sinal Original")

    plt.subplot(5,1,2)
    plt.plot(t, filtered_bpf, color="red")
    plt.title("Após Butterworth BPF (2–100 Hz)")

    plt.subplot(5,1,3)
    plt.plot(t, filtered_dwt, color="green")
    plt.title("Após DWT (db12, nível 5)")

    plt.subplot(5,1,4)
    plt.plot(t, shannon, color="purple")
    plt.title("Shannon Energy (3ª ordem)")

    plt.subplot(5,1,5)
    plt.plot(t, envelope, color="blue")
    plt.title("Envelope Normalizado (LPF 10 Hz)")

    plt.tight_layout()
    plt.show()

    return signal, filtered_bpf, filtered_dwt, normalized, shannon, envelope, fs

signal, bpf, dwt, norm, shannon, envelope, fs = process_pcg("2530_AV.wav")