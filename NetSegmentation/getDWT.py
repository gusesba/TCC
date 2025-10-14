import numpy as np
import pywt

def getDWT(X, N, Name):
    """
    Tradução Python da função MATLAB getDWT.
    Executa decomposição wavelet discreta (DWT) ou contínua (CWT, para 'morl').

    Parâmetros:
    -----------
    X : np.ndarray
        Sinal de entrada (1D)
    N : int
        Níveis de decomposição
    Name : str
        Nome da wavelet (ex: 'db4', 'sym5', 'morl')

    Retorna:
    --------
    cD : np.ndarray
        Coeficientes de detalhe (N x len(X))
    cA : np.ndarray
        Coeficientes de aproximação (N x len(X))
    """

    X = np.asarray(X).flatten()
    length = len(X)

    # Caso especial: 'morl' → Continuous Wavelet Transform
    if Name.lower() == 'morl':
        scales = np.arange(1, N + 1)
        cwtmatr, _ = pywt.cwt(X, scales, 'morl')
        cD = cwtmatr
        cA = cwtmatr
        return cD, cA

    # Caso geral: DWT (ex: db4, sym5, etc.)
    coeffs = pywt.wavedec(X, Name, level=N)

    # coeffs[0] = aproximação final, coeffs[1:] = detalhes
    cA = np.zeros((N, length))
    cD = np.zeros((N, length))

    for k in range(1, N + 1):
        # Coeficientes de detalhe e aproximação no nível k
        d = pywt.upcoef('d', coeffs[k], Name, level=k, take=length)
        a = pywt.upcoef('a', coeffs[0], Name, level=k, take=length)

        # Substitui valores muito pequenos por zero (como no MATLAB)
        d[np.abs(d) < np.sqrt(np.finfo(float).eps)] = 0
        a[np.abs(a) < np.sqrt(np.finfo(float).eps)] = 0

        cD[k - 1, :] = d
        cA[k - 1, :] = a

    return cD, cA
