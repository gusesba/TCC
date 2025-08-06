from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd
import numpy as np
import dask.dataframe as dd


class FeatureExtractor:
    def __init__(self, default_fc_parameters: dict = None):
        if default_fc_parameters is None:
            self.fc_parameters = EfficientFCParameters()
        else:
            self.fc_parameters = default_fc_parameters

    def extract_from_array(
        self,
        signal: np.ndarray,
        sample_id: int = 0,
    ) -> pd.DataFrame:
        """
        Extrai características de um único array 1D com tsfresh.
        """
        df = pd.DataFrame({
            "value": signal,
            "id": [sample_id] * len(signal),  # Repetindo o ID para cada ponto do sinal
            "time": np.arange(len(signal))
        })

        df_formatted = df[["id", "time", "value"]]
        extracted = extract_features(
            df_formatted,
            column_id="id",
            column_sort="time",
            default_fc_parameters=self.fc_parameters,
            disable_progressbar=False
        )
        return extracted
