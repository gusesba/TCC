import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

from butterworth_low_pass_filter import butterworth_low_pass_filter
from butterworth_high_pass_filter import butterworth_high_pass_filter
from schmidt_spike_removal import schmidt_spike_removal
from Homomorphic_Envelope_with_Hilbert import Homomorphic_Envelope_with_Hilbert

def getHeartRateSchmidt(audio_data, Fs, figures=False):
    """
    Calcula a frequência cardíaca e o intervalo sistólico
    com base no método de Schmidt (Springer Segmentation).

    Parâmetros:
        audio_data: ndarray — sinal de áudio (fonocardiograma)
        Fs: float — taxa de amostragem (Hz)
        figures: bool — se True, exibe figuras para debug

    Retorna:
        heartRate: float — batimentos por minuto
        systolicTimeInterval: float — intervalo sistólico (segundos)
    """

    # 25–400 Hz Butterworth bandpass (2ª ordem cada)
    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, Fs, False)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, Fs)

    # Remoção de spikes (Schmidt)
    audio_data = schmidt_spike_removal(audio_data, Fs)

    # Envoltória homomórfica via Hilbert
    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)

    # Autocorrelação normalizada
    y = homomorphic_envelope - np.mean(homomorphic_envelope)
    c = correlate(y, y, mode='full')
    c = c / np.max(np.abs(c))
    signal_autocorrelation = c[len(homomorphic_envelope):]

    # Índices de busca (500 ms – 2000 ms)
    min_index = int(0.5 * Fs)
    max_index = int(2.0 * Fs)

    index = np.argmax(signal_autocorrelation[min_index:max_index])
    true_index = index + min_index - 1

    # Frequência cardíaca (bpm)
    heartRate = 60.0 / (true_index / Fs)

    # Intervalo sistólico (200 ms – metade do ciclo cardíaco)
    max_sys_duration = int(((60.0 / heartRate) * Fs) / 2)
    min_sys_duration = int(0.2 * Fs)

    pos = np.argmax(signal_autocorrelation[min_sys_duration:max_sys_duration])
    systolicTimeInterval = (min_sys_duration + pos) / Fs

    # Plot opcional
    if figures:
        plt.figure("Heart rate calculation figure")
        plt.plot(signal_autocorrelation, label='Autocorrelation')
        plt.plot(true_index, signal_autocorrelation[true_index], 'ro', label='HR peak')
        plt.plot(min_sys_duration + pos, signal_autocorrelation[min_sys_duration + pos],
                 'mo', label='Systolic interval peak')
        plt.xlabel('Samples')
        plt.legend()
        plt.show()

    return heartRate, systolicTimeInterval
