import numpy as np
from getSpringerPCGFeatures import getSpringerPCGFeatures 
from getHeartRateSchmidt import getHeartRateSchmidt
from viterbiDecodePCG_Springer import viterbiDecodePCG_Springer
def runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_obs_distribution, figures=False):
    # --- Extrai features ---
    pcg_features, features_fs = getSpringerPCGFeatures(audio_data, Fs)

    # --- Frequência cardíaca ---
    heart_rate, systolic_interval = getHeartRateSchmidt(audio_data, Fs)

    # --- Decodificação Viterbi ---
    qt = viterbiDecodePCG_Springer(pcg_features, pi_vector, B_matrix, total_obs_distribution, heart_rate, systolic_interval, features_fs)

    # # --- Expande para o tamanho do áudio ---
    # assigned_states = expand_qt(qt, features_fs, Fs, len(audio_data))

    # # --- Plot opcional ---
    # if figures:
    #     import matplotlib.pyplot as plt
    #     t1 = np.arange(len(audio_data)) / Fs
    #     plt.figure(figsize=(15,4))
    #     plt.plot(t1, audio_data / np.max(np.abs(audio_data)), 'k', label='Audio data')
    #     plt.plot(t1, assigned_states, 'r--', label='Derived states')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Amplitude / State')
    #     plt.legend()
    #     plt.title('Derived state sequence')
    #     plt.grid(True)
    #     plt.show()

    # return assigned_states
    return True
