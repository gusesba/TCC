from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas as pd
import numpy as np


class FeatureExtractor:
    def __init__(self, default_fc_parameters: dict = None):
        if default_fc_parameters is None:
            self.fc_parameters = ComprehensiveFCParameters()
        else:
            self.fc_parameters = default_fc_parameters

    def extract_from_array(
        self,
        signal: np.ndarray,
        sample_id: int = 0,
        kind: str = "signal"
    ) -> pd.DataFrame:
        """
        Extrai características de um único array 1D com tsfresh.
        """
        df = pd.DataFrame({
            "value": signal,
            "id": sample_id,
            "kind": kind,
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
