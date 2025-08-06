import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from src.features.extractor import FeatureExtractor

if __name__ == "__main__":
    # Caminho base
    base_dir = os.path.join("data_processed", "segments", "5s")

    # Segmentos a carregar
    segment_folders = [f"segment_{i}" for i in range(3, 10)]

    # Inicializa o extrator (pode trocar por MinimalFCParameters() se quiser menos features)
    extractor = FeatureExtractor()

    # Lista para armazenar os resultados de cada participante
    all_features = []

    for participant_id in tqdm(range(1, 79), desc="Processando participantes"):
        participant_dir = os.path.join(base_dir, str(participant_id))
        signals = []

        try:
            # Carregar os 1.wav dos segmentos 3 a 9
            for segment in segment_folders:
                wav_path = os.path.join(participant_dir, segment, "1.wav")
                signal, _ = librosa.load(wav_path, sr=None)
                signals.append(signal)

            # Carregar 2.wav do segment_9
            last_wav_path = os.path.join(participant_dir, "segment_9", "2.wav")
            signal_2, _ = librosa.load(last_wav_path, sr=None)
            signals.append(signal_2)

            # Concatenar todos os sinais em um só
            full_signal = np.concatenate(signals)

            # Extrair características
            features_df = extractor.extract_from_array(full_signal, sample_id=participant_id)
            features_df["participant_id"] = participant_id  # Adiciona o ID do participante

            all_features.append(features_df)

        except Exception as e:
            print(f"[Erro] Participante {participant_id}: {e}")

    # Concatena todas as features em um único DataFrame
    final_df = pd.concat(all_features, axis=0).reset_index(drop=True)

    # Opcional: reorganiza para que participant_id fique como primeira coluna
    cols = ["participant_id"] + [c for c in final_df.columns if c != "participant_id"]
    final_df = final_df[cols]

    # Salva em CSV
    final_df.to_csv("features_participantes.csv", index=False)

    print("Extração concluída. Arquivo salvo em features_participantes.csv.")