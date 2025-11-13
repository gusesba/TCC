import pandas as pd
import os

# Caminhos de entrada e saída
input_path = "data_processed/features_participantes_divididos.csv"
output_path = "data_processed/features_participantes_seg3_concat.csv"

# Carrega o CSV original
df = pd.read_csv(input_path)

df = df[df["state"] == 3]

# Cria uma lista para armazenar os participantes combinados
combined_rows = []

# Agrupa por participante
for participant_id, group in df.groupby("participant_id"):
    group_prefixed = []

    for _, row in group.iterrows():
        segment = int(row["segment"])
        state = int(row["state"])

        # Remove colunas de identificação antes de prefixar
        row_data = row.drop(labels=["participant_id", "segment", "state"])

        # Adiciona prefixo "segX_stateY_" nas colunas
        row_data.index = [f"seg{segment}_state{state}_{col}" for col in row_data.index]

        group_prefixed.append(row_data)

    # Concatena horizontalmente as features de todos os segmentos e estados
    participant_df = pd.concat(group_prefixed).to_frame().T
    participant_df["participant_id"] = participant_id
    combined_rows.append(participant_df)

# Junta todos os participantes
final_df = pd.concat(combined_rows, axis=0).reset_index(drop=True)

# Move participant_id para o início
cols = ["participant_id"] + [c for c in final_df.columns if c != "participant_id"]
final_df = final_df[cols]

# Salva o resultado
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"✅ Arquivo gerado: {output_path}")
