import pandas as pd
import os

# Caminho do arquivo original
input_path = "data_processed/features_participantes_divididos.csv"
output_path = "data_processed/features_participantes_segment4_concat.csv"

# Carrega o CSV original
df = pd.read_csv(input_path)

# Mantém apenas o segmento 4
df = df[df["segment"] == 4].copy()

# Remove colunas redundantes se existirem
df = df.drop(columns=["segment"], errors="ignore")

# Cria um dicionário para armazenar as linhas combinadas
combined_rows = []

# Agrupa por participante
for participant_id, group in df.groupby("participant_id"):
    # Prefixa as colunas com o estado correspondente
    group_prefixed = []
    for _, row in group.iterrows():
        state = int(row["state"])
        # Remove colunas de identificação antes de prefixar
        row_data = row.drop(labels=["participant_id", "state"])
        # Adiciona prefixo ao nome das colunas
        row_data.index = [f"state_{state}_{col}" for col in row_data.index]
        group_prefixed.append(row_data)

    # Concatena horizontalmente as features do mesmo participante
    participant_df = pd.concat(group_prefixed).to_frame().T
    participant_df["participant_id"] = participant_id
    combined_rows.append(participant_df)

# Junta todos os participantes em um único DataFrame
final_df = pd.concat(combined_rows, axis=0).reset_index(drop=True)

# Move participant_id para a primeira coluna
cols = ["participant_id"] + [c for c in final_df.columns if c != "participant_id"]
final_df = final_df[cols]

# Salva o resultado
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"✅ Arquivo gerado: {output_path}")
