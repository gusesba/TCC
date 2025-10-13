# pcg_segmentation.py
# Implementação heurística da segmentação "Noise-Robust Shannon Energy"
# Inspirada no artigo de Arjoune et al. (2024)

import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt


# ----------- Funções utilitárias -----------

def bandpass_filter(x, fs, low=2.0, high=100.0, order=2):
    """Filtro passa-faixa Butterworth"""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, x)


def shannon_energy(x, eps=1e-12):
    """Energia de Shannon"""
    x = x / (np.max(np.abs(x)) + eps)
    a2 = x**2 + eps
    return - a2 * np.log(a2)


def moving_average(x, win_samples):
    """Média móvel simples"""
    kernel = np.ones(win_samples) / win_samples
    return np.convolve(x, kernel, mode='same')


def detect_noisy_segments(x, fs, win_sec=1.0, step_sec=0.5, thresh_z=2.5):
    """Detecta janelas com energia anômala (ruído)"""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    energies, positions = [], []
    for start in range(0, len(x)-win+1, step):
        seg = x[start:start+win]
        energies.append(np.sum(seg**2))
        positions.append(start)
    energies = np.array(energies)
    med, std = np.median(energies), np.std(energies)
    noisy_mask = energies > (med + thresh_z * std)

    noisy_ranges = []
    for i, nm in enumerate(noisy_mask):
        if nm:
            s = positions[i]
            noisy_ranges.append((s, min(s+win, len(x))))

    # mesclar ranges próximos
    merged = []
    for s, e in sorted(noisy_ranges):
        if not merged or s > merged[-1][1] + int(0.1*fs):
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(int(a), int(b)) for a, b in merged]


def detect_peaks_from_shannon(x, fs, smooth_ms=50, peak_prominence=0.02):
    """Detecção de picos via energia de Shannon"""
    E = shannon_energy(x)
    smooth = moving_average(E, int(smooth_ms*fs/1000))
    smooth /= np.max(smooth) + 1e-12
    distance = int(0.15 * fs)  # intervalo fisiológico mínimo ~150ms
    peaks, _ = signal.find_peaks(smooth, distance=distance, prominence=peak_prominence)
    return peaks, smooth, E


def label_s1_s2(peaks, fs):
    """Classificação S1 e S2 por heurística baseada em intervalos"""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    times = np.array(peaks) / fs
    d = np.diff(times)
    thr = (np.median(d) + np.min(d)) / 2.0

    labels = np.full(len(peaks), '', dtype=object)
    i = 0
    while i < len(peaks)-1:
        if d[i] < thr:
            labels[i], labels[i+1] = 'S1', 'S2'
            i += 2
        else:
            if labels[i+1] == '':
                labels[i+1] = 'S1'
            i += 1

    if labels[0] == '':
        labels[0] = 'S1'

    s1_peaks = peaks[np.where(labels == 'S1')[0]]
    s2_peaks = peaks[np.where(labels == 'S2')[0]]
    return s1_peaks, s2_peaks


# ----------- Pipeline principal -----------

def segment_pcg(file_path, params=None, plot=True):
    if params is None:
        params = {'low': 2.0, 'high': 100.0, 'smooth_ms': 40,
                  'prominence': 0.01, 'noise_win': 1.0,
                  'noise_step': 0.5, 'noise_z': 2.5}

    # carregar sinal
    x, fs = sf.read(file_path)
    if x.ndim > 1:
        x = x[:, 0]  # usa canal único se for estéreo

    # filtro
    x_filt = bandpass_filter(x, fs, low=params['low'], high=params['high'])

    # detectar ruído
    noisy = detect_noisy_segments(x_filt, fs, params['noise_win'],
                                  params['noise_step'], params['noise_z'])

    # picos globais
    peaks, smooth, E = detect_peaks_from_shannon(x_filt, fs,
                                                 smooth_ms=params['smooth_ms'],
                                                 peak_prominence=params['prominence'])

    # rotular S1 / S2
    s1_peaks, s2_peaks = label_s1_s2(peaks, fs)

    # plot
    if plot:
        t = np.arange(len(x_filt)) / fs
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, x_filt, label="PCG filtrado")
        for (s, e) in noisy:
            plt.axvspan(s/fs, e/fs, color='red', alpha=0.2)
        plt.legend(); plt.title("Sinal PCG (áreas vermelhas = ruído)")

        plt.subplot(2, 1, 2)
        plt.plot(t, smooth, label="Energia de Shannon (suavizada)")
        plt.scatter(s1_peaks/fs, smooth[s1_peaks], c='g', marker='o', label='S1')
        plt.scatter(s2_peaks/fs, smooth[s2_peaks], c='r', marker='s', label='S2')
        plt.legend(); plt.xlabel("Tempo (s)")
        plt.tight_layout()
        plt.show()

    return {"fs": fs, "filtered": x_filt, "peaks": peaks,
            "s1": s1_peaks, "s2": s2_peaks, "noisy": noisy}

result = segment_pcg("2530_AV.wav", plot=True)