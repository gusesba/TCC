import os
import pandas as pd

if __name__ == "__main__":
    dados = pd.read_csv(os.path.join("data_processed", "dados_filtrados.csv"))
    features = pd.read_csv(os.path.join("data_processed", "features_participantes_seg3_concat.csv"))

    df = pd.concat([dados, features], axis=1)
    df.to_csv(os.path.join("data_processed", "dados_completos_divididos_seg3.csv"), index=False)