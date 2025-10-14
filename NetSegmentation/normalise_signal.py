import numpy as np

def normalise_signal(signal):
    """
    Normaliza o sinal para ter média 0 e desvio padrão 1,
    equivalente à função MATLAB normalise_signal.m
    """
    signal = np.asarray(signal, dtype=float)
    mean_of_signal = np.mean(signal)
    standard_deviation = np.std(signal)
    
    normalised_signal = (signal - mean_of_signal) / standard_deviation
    return normalised_signal
