import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high, figures=False):
    """
    Calcula a feature PSD usada no Springer HMM.
    Equivalente ao MATLAB get_PSD_feature_Springer_HMM.
    """
    nperseg = int(sampling_frequency / 40)
    noverlap = int(sampling_frequency / 80)
    f, t, Sxx = spectrogram(data, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap, nfft=int(sampling_frequency/2))
    
    if figures:
        plt.figure()
        plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='auto')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.colorbar(label='Power (dB)')
        plt.show()

    # Limite de frequências
    low_idx = np.argmin(np.abs(f - frequency_limit_low))
    high_idx = np.argmin(np.abs(f - frequency_limit_high))

    # Média da PSD sobre a faixa de frequência de interesse
    psd = np.mean(Sxx[low_idx:high_idx+1, :], axis=0)

    if figures:
        t_data = np.arange(len(data)) / sampling_frequency
        t_psd = np.arange(len(psd)) / sampling_frequency
        plt.figure(figsize=(10,4))
        plt.plot((data - np.mean(data))/np.std(data), 'c', label='Signal (normalized)')
        plt.plot((psd - np.mean(psd))/np.std(psd), 'k', label='PSD Feature (normalized)')
        plt.xlabel('Sample index')
        plt.legend()
        plt.title('PSD Feature')
        plt.show()

    return psd
