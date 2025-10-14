import numpy as np

def schmidt_spike_removal(original_signal, fs):
    """
    Versão Python do schmidt_spike_removal do MATLAB.
    Remove spikes grandes do sinal baseado na média móvel absoluta.
    """
    # --- Tamanho da janela (500 ms) ---
    windowsize = round(fs / 2)

    # --- Ajusta para múltiplos completos de janela ---
    trailingsamples = len(original_signal) % windowsize
    if trailingsamples > 0:
        signal_trimmed = original_signal[:-trailingsamples]
    else:
        signal_trimmed = original_signal

    # --- Reshape em janelas ---
    sampleframes = signal_trimmed.reshape((windowsize, -1))

    # --- Calcula MAAs (máximos absolutos por janela) ---
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # --- Loop para remover spikes maiores que 3*mediana ---
    while np.any(MAAs > 3 * np.median(MAAs)):
        window_num = np.argmax(MAAs)

        # Posição do spike dentro da janela
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        # Zero crossings (mudança de sinal)
        zero_crossings = np.abs(np.diff(np.sign(sampleframes[:, window_num]), prepend=0)) > 1

        # Início do spike: último zero crossing antes do spike
        crossings_before = np.where(zero_crossings[:spike_position+1])[0]
        spike_start = crossings_before[-1] if crossings_before.size > 0 else 0

        # Fim do spike: primeiro zero crossing depois do spike
        zero_crossings[:spike_position+1] = 0
        crossings_after = np.where(zero_crossings)[0]
        spike_end = crossings_after[0] if crossings_after.size > 0 else windowsize - 1

        # Zera o spike (usando 0.0001 como no MATLAB)
        sampleframes[spike_start:spike_end+1, window_num] = 0.0001

        # Recalcula MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

    # --- Reconstrói o sinal ---
    despiked_signal = sampleframes.reshape(-1)

    # --- Adiciona os trailing samples de volta ---
    if trailingsamples > 0:
        despiked_signal = np.concatenate([despiked_signal, original_signal[len(despiked_signal):]])

    return despiked_signal
