import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from extractor import FeatureExtractor

if __name__ == "__main__":
    # Caminho base onde estão os áudios divididos
    base_dir = os.path.join("data_processed", "SegmentsDivided")

    # Participantes e segmentos
    participants = range(1, 79)          # 1..78
    segment_nums = range(3, 10)          # 3..9
    state_nums = range(1, 5)             # 1..4 (S1, Systole, S2, Diastole)

    # Inicializa o extrator de características
    extractor = FeatureExtractor()

    # Lista para armazenar as features de todos os arquivos
    all_features = []

    for participant_id in tqdm(participants, desc="Processando participantes"):
        for segment in segment_nums:
            for state in state_nums:
                wav_path = os.path.join(base_dir, str(participant_id), f"segment_{segment}_state_{state}.wav")

                # Verifica se o arquivo existe (alguns podem não conter todos os estados)
                if not os.path.isfile(wav_path):
                    continue

                try:
                    # Carrega o áudio (sem alterar a taxa de amostragem)
                    signal, sr = librosa.load(wav_path, sr=None)

                    # Extrai características
                    features_df = extractor.extract_from_array(signal, sample_id=f"{participant_id}_{segment}_{state}")

                    # Adiciona colunas de identificação
                    features_df["participant_id"] = participant_id
                    features_df["segment"] = segment
                    features_df["state"] = state

                    # Armazena
                    all_features.append(features_df)

                except Exception as e:
                    print(f"[Erro] Participante {participant_id}, Segmento {segment}, Estado {state}: {e}")

    # Concatena todas as features em um único DataFrame
    if len(all_features) == 0:
        print("Nenhum arquivo processado.")
        exit()

    final_df = pd.concat(all_features, axis=0).reset_index(drop=True)

    # Reorganiza colunas (opcional)
    cols = ["participant_id", "segment", "state"] + [c for c in final_df.columns if c not in ["participant_id", "segment", "state"]]
    final_df = final_df[cols]

    # Salva o resultado
    output_path = "data_processed/features_participantes_divididos.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"✅ Extração concluída! Arquivo salvo em: {output_path}")
