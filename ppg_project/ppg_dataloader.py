from typing import Tuple


import pandas as pd
from torch.utils.data import Dataset


class PPGDataloader(Dataset):
    "Custom Dataset for PPG data"

    def __init__(self, data: pd.DataFrame):
        """Initialize custom dataset for PPG samples.

        Args:
            config (PipelineConfig): configuration details for pipeline.
            data (pd.DataFrame): dataframe containing sample details.
        """
        self.data = data

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[pd.Series, pd.Series]:
        """Select an PPG sample and its target.

        Args:
            index (int): randomly selected index.

        Returns:
            Tuple[pd.Series, pd.Series]: data and target of the sample.
        """
        sample_data = self.get_sample_data(index)
        sample_label = self._get_sample_target(index)
        return sample_data, sample_label

    def get_sample_data(self, index: int) -> pd.Series:
        """Get an PPG sample transformed with spectral feature from dataset.

        Args:
            index (int): index randomly selected.

        Returns:
            pd.Series: transformed ppg sample selected.
        """
        return self.data["dl_feature"].iloc[index]

    def _get_sample_target(self, index: int) -> int:
        """Get a target from dataset.

        Args:
            index (int): index randomly selected.

        Returns:
            int: target selected.
        """
        return self.data["target"].iloc[index]
