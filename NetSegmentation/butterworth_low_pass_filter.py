import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

def butterworth_low_pass_filter(original_signal, order, cutoff, sampling_frequency, figures=False):
    """
    Versão Python do butterworth_low_pass_filter do MATLAB,
    com plot da resposta em frequência equivalente ao fvtool.
    """
    # --- Coeficientes do filtro Butterworth ---
    nyq = 0.5 * sampling_frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # --- Plot da resposta em frequência ---
    if figures:
        w, h = freqz(b, a, worN=8000)
        plt.figure(figsize=(8,4))
        plt.plot(0.5*sampling_frequency*w/np.pi, abs(h), 'b')
        plt.title("Resposta em frequência do filtro low-pass")
        plt.xlabel('Frequência [Hz]')
        plt.ylabel('Ganho')
        plt.grid(True)
        plt.show()

    # --- Filtragem forward-backward (zero phase) ---
    low_pass_filtered_signal = filtfilt(b, a, original_signal)

    # --- Plot do sinal original x filtrado ---
    if figures:
        plt.figure(figsize=(8,4))
        plt.plot(original_signal, label='Original Signal')
        plt.plot(low_pass_filtered_signal, 'r', label='Low-pass filtered signal')
        plt.legend()
        plt.title("Sinal original vs. sinal filtrado (low-pass)")
        plt.show()

    return low_pass_filtered_signal
