import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def Hilbert_Envelope(input_signal, sampling_frequency, figures=False):
    """
    Calcula o envelope de Hilbert do sinal.
    Equivalente ao MATLAB Hilbert_Envelope.
    """
    hilbert_envelope = np.abs(hilbert(input_signal))

    # --- Plot opcional ---
    if figures:
        plt.figure(figsize=(10,4))
        plt.plot(input_signal, label='Original Signal')
        plt.plot(hilbert_envelope, 'r', label='Hilbert Envelope')
        plt.legend()
        plt.title('Hilbert Envelope')
        plt.show()

    return hilbert_envelope
