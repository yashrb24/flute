from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.fft import fft
from string import ascii_uppercase

def parse_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and applies Z-score normalization to the 'x', 'y', and 'z' columns.

    Parameters:
    -----------
    file_path : str
        Path to the input CSV file.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing normalized 'x', 'y', 'z' values along with the 'label' column.
    """
    df = pd.read_csv(file_path, names=["x", "y", "z", "label"], skiprows=1).dropna()

    return df


def compute_slope(y: Union[List[float], np.ndarray]) -> float:
    """
    Computes the slope of a linear regression fit.

    Parameters:
    -----------
    x : np.ndarray
        Array of input data (independent variable).
    y : Union[List[float], np.ndarray]
        Array of output data (dependent variable).

    Returns:
    --------
    float
        Slope of the best fit line.
    """
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


def rolling_windows(
    df: pd.DataFrame,
    window_size: int = 500,
    slide_distance: int = 100,
    threshold: int = 100,
) -> List[pd.DataFrame]:
    """
    Splits the DataFrame into rolling windows and filters based on label criteria.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    window_size : int, optional
        Size of each window (default is 500).
    slide_distance : int, optional
        Distance to slide between windows (default is 50).
    threshold : int, optional
        Number of 'noise' labels allowed before discarding a window (default is 20).

    Returns:
    --------
    List[pd.DataFrame]
        List of DataFrame windows that meet the filtering criteria.
    """

    df["x"] = (df["x"] - df["x"].mean()) / df["x"].std()
    df["y"] = (df["y"] - df["y"].mean()) / df["y"].std()
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()

    windows = []
    counter = 0
    for start in range(0, len(df) - window_size + 1, slide_distance):
        window = df.iloc[start : start + window_size]

        # Discard window if it contains multiple labels and exceeds the 'noise' threshold
        if (
            len(window["label"].unique()) > 1
            and len(window[window["label"] == "noise"]) > threshold
        ):
            counter += 1
            continue

        windows.append(window)

    return windows


def rolling_windows_without_noise(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Splits the DataFrame into rolling windows excluding the 'noise' label.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.

    Returns:
    --------
    List[pd.DataFrame]
        List of DataFrame windows excluding the 'noise' label.
    """

    df["x"] = (df["x"] - df["x"].mean()) / df["x"].std()
    df["y"] = (df["y"] - df["y"].mean()) / df["y"].std()
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()

    start_indices = []
    end_indices = []

    flag = False
    for i in range(len(df)):
        if flag:
            if df["label"][i] == "noise":
                end_indices.append(i)
                flag = False
            else:
                continue
        else:
            if df["label"][i] != "noise":
                start_indices.append(i)
                flag = True
            else:
                continue
    
    windows = []
    for start, end in zip(start_indices, end_indices):
        window = df.iloc[start : end]
        windows.append(window)
        
    return windows


def rolling_windows_without_noise_v2(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Splits the DataFrame into rolling windows excluding the 'noise' label.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.

    Returns:
    --------
    List[pd.DataFrame]
        List of DataFrame windows excluding the 'noise' label.
    """
    
    df["x"] = (df["x"] - df["x"].mean()) / df["x"].std()
    df["y"] = (df["y"] - df["y"].mean()) / df["y"].std()
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()
    
    windows = []
    for c in ascii_uppercase:
        df_c = df[df["label"] == c]
        windows.append(df_c)
        assert len(df_c) > 0, f"Letter {c} -> the data is empty"
    
    return windows

def fft_features(signal: pd.Series) -> Tuple[float, float]:
    """
    Computes FFT-based features: energy and entropy of the signal.

    Parameters:
    -----------
    signal : pd.Series
        Input signal data.

    Returns:
    --------
    Tuple[float, float]
        Energy and entropy of the signal.
    """
    # Convert Pandas Series to NumPy array
    signal_array = signal.to_numpy()

    # Perform FFT
    fft_values = fft(signal_array)

    # Calculate magnitudes of FFT coefficients
    magnitudes = np.abs(fft_values)

    # Calculate energy
    energy = np.sum(magnitudes**2) / (100 * len(magnitudes))

    # Calculate normalized PSD
    psd = magnitudes**2 / energy

    # Calculate entropy (avoiding log(0))
    entropy = -np.sum(psd * np.log2(psd + 1e-6)) / (100 * len(magnitudes))

    return energy, entropy


def merge_windows(windows: List[pd.DataFrame]) -> Tuple[DataFrame, DataFrame]:
    """
    Merges features from multiple windows and calculates statistical features for each window.

    Parameters:
    -----------
    windows : List[pd.DataFrame]
        List of DataFrame windows.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing a DataFrame of merged window features and a DataFrame of corresponding labels.
    """
    merged_windows = []
    merged_labels = []

    columns = list(windows[0].columns)
    columns.remove("label")

    for window in windows:
        merged = {}

        # Compute statistics and FFT features for each window column
        for col in columns:
            merged[f"{col}_mean"] = window[col].mean()
            merged[f"{col}_std"] = window[col].std()
            merged[f"{col}_median"] = window[col].median()
            merged[f"{col}_min"] = window[col].min()
            merged[f"{col}_max"] = window[col].max()
            merged[f"{col}_energy"], merged[f"{col}_entropy"] = fft_features(
                window[col]
            )
            merged[f"{col}_slope"] = compute_slope(window[col])

        # The majority label in the window is the label for the window
        merged_label = window["label"].mode().values[0]
        # Append the merged features and label to lists
        merged_windows.append(merged)
        merged_labels.append(merged_label)

    return pd.DataFrame(merged_windows), pd.DataFrame(merged_labels)


def merge_windows_v2(windows: List[pd.DataFrame]) -> Tuple[DataFrame, DataFrame]:
    """
    Merges features from multiple windows and calculates statistical features for each window.

    Parameters:
    -----------
    windows : List[pd.DataFrame]
        List of DataFrame windows.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing a DataFrame of merged window features and a DataFrame of corresponding labels.
    """
    merged_windows = []
    merged_labels = []

    columns = ["x", "y", "z"]
    for window in windows:
        merged = {}

        # Compute statistics and FFT features for each window column
        for col in columns:
            merged[f"{col}_mean"] = window[col].mean()
            merged[f"{col}_std"] = window[col].std()
            merged[f"{col}_median"] = window[col].median()
            merged[f"{col}_min"] = window[col].min()
            merged[f"{col}_max"] = window[col].max()
            merged[f"{col}_energy"], merged[f"{col}_entropy"] = fft_features(
                window[col]
            )
            merged[f"{col}_slope"] = compute_slope(window[col])

        merged_label = window["label"].loc[window.index[0]]
        merged_windows.append(merged)
        merged_labels.append(merged_label)
    
    assert len(merged_windows) == len(merged_labels)
    assert len(merged_windows) == len(windows) == 26
    
    return pd.DataFrame(merged_windows), pd.DataFrame(merged_labels)


def process_file(df: pd.DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Processes the input DataFrame by generating rolling windows and merging features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing time-series data.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Processed feature DataFrame and corresponding labels.
    """
    windows = rolling_windows_without_noise(df)
    processed_windows, labels = merge_windows(windows)
    print(f"length of processed_windows: {len(processed_windows)}")
    return processed_windows, labels


def process_file_v2(df: pd.DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Processes the input DataFrame by generating rolling windows and merging features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing time-series data.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Processed feature DataFrame and corresponding labels.
    """
    windows = rolling_windows_without_noise_v2(df)
    processed_windows, labels = merge_windows_v2(windows)
    return processed_windows, labels


if __name__ == "__main__":
    # Example file path to the dataset
    file_name = "/home/candy/Projects/flute/user_data_new/yash/yash_2.csv"

    # Parse the input file
    df = parse_file(file_name)

    # Process the data
    X, y = process_file_v2(df)

    print(X.head())
    # print(y.head())
