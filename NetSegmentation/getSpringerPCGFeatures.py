import numpy as np
from scipy.signal import resample
from butterworth_low_pass_filter import butterworth_low_pass_filter
from butterworth_high_pass_filter import butterworth_high_pass_filter
from schmidt_spike_removal import schmidt_spike_removal
from Homomorphic_Envelope_with_Hilbert import Homomorphic_Envelope_with_Hilbert
from Hilbert_Envelope import Hilbert_Envelope
from get_PSD_feature_Springer_HMM import get_PSD_feature_Springer_HMM
from getDWT import getDWT
from normalise_signal import normalise_signal
def getSpringerPCGFeatures(audio_data, Fs, figures=False, include_wavelet=True, featuresFs=200):
    # ========================
    # Pipeline de features
    # ========================
    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, Fs, False)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, Fs)
    audio_data = schmidt_spike_removal(audio_data, Fs)

    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)
    downsampled_homomorphic = resample(homomorphic_envelope, int(len(homomorphic_envelope) * featuresFs / Fs))
    downsampled_homomorphic = normalise_signal(downsampled_homomorphic)

    hilbert_envelope = Hilbert_Envelope(audio_data, Fs)
    downsampled_hilbert = resample(hilbert_envelope, int(len(hilbert_envelope) * featuresFs / Fs))
    downsampled_hilbert = normalise_signal(downsampled_hilbert)

    psd = get_PSD_feature_Springer_HMM(audio_data, Fs, 40, 60)
    psd = resample(psd, len(downsampled_homomorphic))
    psd = normalise_signal(psd)

    if include_wavelet:
        cD, cA = getDWT(audio_data, 3, 'rbio3.9')
        wavelet_feature = np.abs(cD[2])  # nível 3
        wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]
        downsampled_wavelet = resample(wavelet_feature, len(downsampled_homomorphic))
        downsampled_wavelet = normalise_signal(downsampled_wavelet)
        PCG_Features = np.column_stack([downsampled_homomorphic, downsampled_hilbert, psd, downsampled_wavelet])
    else:
        PCG_Features = np.column_stack([downsampled_homomorphic, downsampled_hilbert, psd])

    # --- Plot opcional ---
    if figures:
        import matplotlib.pyplot as plt
        t_audio = np.arange(len(audio_data)) / Fs
        t_feat = np.arange(len(PCG_Features)) / featuresFs
        plt.figure(figsize=(12,4))
        plt.plot(t_audio, audio_data, label='Audio')
        for i in range(PCG_Features.shape[1]):
            plt.plot(t_feat, PCG_Features[:,i], label=f'Feature {i+1}')
        plt.legend()
        plt.xlabel('Tempo (s)')
        plt.show()

    return PCG_Features, featuresFs
