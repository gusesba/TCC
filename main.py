from scipy.io import wavfile
from src.features.extractor import FeatureExtractor

if __name__ == "__main__":
    extractor = FeatureExtractor()

    # Exemplo: carregar um .wav e extrair as features
    sr, data = wavfile.read("segment_3.wav")

    # Garantir que o dado esteja mono
    if data.ndim > 1:
        data = data[:, 0]

    features = extractor.extract_from_array(data)
    print(features)
