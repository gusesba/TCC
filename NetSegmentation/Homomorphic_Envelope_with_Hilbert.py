import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency, lpf_frequency=8, figures=False):
    """
    Calcula o envelope homomórfico com Hilbert e filtro passa-baixa.
    Equivalente ao MATLAB Homomorphic_Envelope_with_Hilbert.
    """
    # --- Filtro Butterworth 1º ordem ---
    nyq = 0.5 * sampling_frequency
    normal_cutoff = lpf_frequency / nyq
    b, a = butter(1, normal_cutoff, btype='low', analog=False)

    # --- Envelope homomórfico ---
    analytic_signal = hilbert(input_signal)
    homomorphic_envelope = np.exp(filtfilt(b, a, np.log(np.abs(analytic_signal) + 1e-10)))

    # --- Remove spikes do primeiro ponto ---
    homomorphic_envelope[0] = homomorphic_envelope[1]

    # --- Plot opcional ---
    if figures:
        plt.figure(figsize=(10,4))
        plt.plot(input_signal, label='Original Signal')
        plt.plot(homomorphic_envelope, 'r', label='Homomorphic Envelope')
        plt.legend()
        plt.title('Homomorphic Envelope')
        plt.show()

    return homomorphic_envelope
