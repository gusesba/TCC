def default_Springer_HSMM_options():
    """
    Define as opções padrão do algoritmo Springer HSMM (Hidden Semi-Markov Model)
    utilizado para segmentação de sons cardíacos.
    """

    springer_options = {
        # Frequência de amostragem original dos sinais de áudio
        "audio_Fs": 1000,

        # Frequência para extração de features (downsample)
        # Springer usa 50 Hz no paper original
        "audio_segmentation_Fs": 50,

        # Tolerância de erro na localização de S1 e S2 (em segundos)
        "segmentation_tolerance": 0.1,

        # Se deve usar o código MEX otimizado (no Python normalmente False)
        "use_mex": False,

        # Se deve incluir feature de wavelet
        "include_wavelet_feature": True,
    }

    return springer_options
