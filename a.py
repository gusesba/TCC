import os
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from itertools import combinations
from queue import PriorityQueue

# Carrega o CSV
df = pd.read_csv(os.path.join("data_processed", "dados_completos.csv"))

# Remove colunas constantes (sem variabilidade)
df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.select_dtypes(include=[np.number])

df.to_csv(os.path.join("data_processed", "todas_features.csv"), index=False)