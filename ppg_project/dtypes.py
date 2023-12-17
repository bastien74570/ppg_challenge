"""dtypes for both training pipeline."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseTrainingPipelineConfig:
    """Base configuration for the training pipeline.
    data_path (str): data path for training set. Default to "../data"
    sampling_rate (int): sampling rate for the data. Default to 100 Hz.
    apply_preprocessing (bool): whether or not to apply preprocessing to the data. Default to True.
    lowcut (int): lowcut frequency for the bandpass filter. Default to 0.5 Hz.
    highcut (int): highcut frequency for the bandpass filter. Default to 5 Hz.
    appply_segmentation (bool): whether or not to segment the data. Default to False.
    segment_window_sec (int): window size for segmentation. Default to 1 second.
    """
    data_path: str = "../data"
    sampling_rate: int = 100
    apply_preprocessing: bool = True
    lowcut: int = 0.5
    highcut: int = 5
    appply_segmentation: bool = False
    segment_window_sec : int = 1
    

@dataclass
class XGBoostTrainingPipelineConfig(BaseTrainingPipelineConfig):
    """Configure for XGBoost training pipeline.
    validation_strategy (str): validation strategy to use. Default to "kfold".
    num_folds (int): number of folds for cross validation during fine tuning. Default to 5.
    features_list (Tuple[str]): list of features to use for training. Default to ("features","peak","spectrogram",)
    apply_pca (bool): wether or not to apply pca. Default to False.
    max_depth (int): maximum depth of the tree. Default to 4.
    min_child_weight (int): minimum sum of instance weight needed in a child. Default to 6.
    subsample (float): subsample ratio of the training instance. Default to 1.
    colsample_bytree (float): subsample ratio of columns when constructing each tree. Default to 0.4.
    """
    validation_strategy: str = "kfold"
    num_folds: int = 5
    apply_pca: bool = False
    features_list: Tuple[str] = (
        "features",
        "peak",
        "spectrogram",
    )
    max_depth: int = 4
    min_child_weight: int = 6
    subsample: float = 1
    colsample_bytree: float = 0.4


@dataclass
class DLTrainingPipelineConfig(BaseTrainingPipelineConfig):
    """Configure for deep learning training pipeline.
    batch_size (int): batch size for training. Default to 64.
    learning_rate (float): learning rate for training. Default to 1e-4.
    n_epochs (int): number of epochs for training. Default to 10.
    early_stopping_tolerance (int): number of epochs to wait before early stopping. Default to 3.
    gradient_accumulation_steps (int): number of gradient accumulation steps. Default to 1.
    test_size_val (float): validation set size. Default to 0.2.
    test_size (float): test set size. Default to 0.2.
    """
    batch_size: int = 64
    learning_rate: float = 1e-4
    n_epochs: int = 30
    dl_features: str = "wavelet"
    early_stopping_tolerance: int = 3
    gradient_accumulation_steps: int = 1
    test_size_val: float = 0.2
    test_size: float = 0.2
