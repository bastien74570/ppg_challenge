"""Utility functions for the PPG project."""
from typing import Tuple

import numpy as np
import pandas as pd
import pywt
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(
    data: np.array, lowcut: int, highcut: int, sampling_rate: int, order: int = 3
) -> np.array:
    """Apply butterworth bandpass filter to data.

    Args:
        data (np.array): data to filter.
        lowcut (int): lowcut frequency.
        highcut (int): highcut frequency.
        sampling_rate (int): sampling rate of the data.
        order (int, optional): order of the filter to apply. Defaults to 3.

    Returns:
        np.array: filtered data.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_wavelet_features(
    signal: np.array,
    sampling_rate: int,
    scales: np.array = np.arange(1, 128),
    wavelet: str = "morl",
) -> Tuple[np.array, np.array]:
    """Compute wavelet features from signal.

    Args:
        signal (np.array): signal to compute features from.
        sampling_rate (int): sampling rate of the signal.
        scales (np.array, optional): scales to use for frequencies. Defaults to np.arange(1, 128).
        wavelet (str, optional): wavelet method to use. Defaults to "morl".

    Returns:
        Tuple[np.array, np.array]: frequencies and coefficients.
    """
    coefficients, frequencies = pywt.cwt(
        signal, scales, wavelet, sampling_period=1 / sampling_rate
    )
    coefficients = np.abs(coefficients)
    return frequencies, coefficients


def get_log_spectrogram_features(
    signal: np.array, sampling_rate: int
) -> Tuple[np.array, np.array, np.array]:
    """Compute log spectrogram features from signal.

    Args:
        signal (np.array): signal to compute features from.
        sampling_rate (int): sampling rate of the signal.

    Returns:
        Tuple[np.array, np.array, np.array]: frequencies, times and log spectrogram.
    """
    frequencies, times, Sxx = spectrogram(signal, sampling_rate)
    log_Sxx = np.log(Sxx)
    return frequencies, times, log_Sxx


def segment_data_in_window(
    data: pd.DataFrame,
    target: pd.DataFrame = None,
    nsec: int = 5,
    sampling_rate: int = 100,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Segment data in window of nsec seconds.

    Args:
        data (pd.DataFrame): dataframe containing values to segment.
        target (pd.DataFrame, optional): target associated to dataframe features. Defaults to None.
        nsec (int, optional): number of seconds for segment. Defaults to 5.
        sampling_rate (int, optional): sampling rate of ppg data. Defaults to 100.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]: segmented data and target.
    """
    ppg_list_col = [c for c in data.columns if c.startswith("ppg")]
    crafted_features_col = [c for c in data.columns if c.startswith("features_")]
    ppg_data = data[ppg_list_col].values
    crafted_data = data[crafted_features_col].values
    segment_length = nsec * sampling_rate
    num_segments = int(ppg_data.shape[1] / segment_length)

    list_data = list()
    list_feature = list()
    list_target = list()
    list_sample_index = list()
    list_window_index = list()

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        ppg_data_window = ppg_data[:, start_idx:end_idx]
        list_data.append(ppg_data_window)
        list_feature.append(crafted_data)
        list_target.append(target["target"].to_numpy())
        list_sample_index.append(np.arange(ppg_data.shape[0]))
        list_window_index.append(i * np.ones(ppg_data.shape[0]))
    data_augmented = np.concatenate(list_data, axis=0)
    target_augmented = np.concatenate(list_target)
    feature_augmented = np.concatenate(list_feature)
    window_index_augmented = np.concatenate(list_window_index)
    sample_index_augmented = np.concatenate(list_sample_index)
    data_window = pd.DataFrame(data_augmented, columns=ppg_list_col[:segment_length])
    target_window = pd.DataFrame(target_augmented, columns=["target"])
    feature_window = pd.DataFrame(feature_augmented, columns=crafted_features_col)
    data_window = pd.concat([data_window, feature_window, target_window], axis=1)
    data_window.insert(0, "window_index", window_index_augmented.astype(int))
    data_window.insert(0, "sample_index", sample_index_augmented.astype(int))
    data_window.sort_values(by="sample_index", inplace=True)
    
    data_augmented = data_window.drop(columns="target").reset_index(drop=True)
    target_augmented = data_window["target"].reset_index(drop=True)
    return data_augmented, target_augmented