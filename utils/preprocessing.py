"""
Preprocessing utilities for Parkinson's detector application.

This module provides functions for data preprocessing including normalization,
feature scaling, data validation, and other data preparation tasks.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def normalize_features(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "standard",
    fit_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[StandardScaler, MinMaxScaler]]:
    """
    Normalize/scale features in the dataset.
    
    Args:
        data: Input data to normalize (numpy array or pandas DataFrame)
        method: Scaling method - 'standard' (z-score) or 'minmax'
        fit_scaler: Fitted scaler object. If None, will fit a new scaler
        
    Returns:
        Tuple of (normalized_data, scaler_object)
    """
    if method == "standard":
        scaler = fit_scaler if fit_scaler is not None else StandardScaler()
    elif method == "minmax":
        scaler = fit_scaler if fit_scaler is not None else MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if isinstance(data, pd.DataFrame):
        if fit_scaler is None:
            normalized = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            normalized = pd.DataFrame(
                scaler.transform(data),
                columns=data.columns,
                index=data.index
            )
    else:
        if fit_scaler is None:
            normalized = scaler.fit_transform(data)
        else:
            normalized = scaler.transform(data)
    
    return normalized, scaler


def remove_outliers(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "iqr",
    threshold: float = 1.5
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Remove outliers from the dataset.
    
    Args:
        data: Input data
        method: Outlier detection method - 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Data with outliers removed
    """
    if isinstance(data, pd.DataFrame):
        data_to_process = data.copy()
    else:
        data_to_process = data.copy()
    
    if method == "iqr":
        if isinstance(data_to_process, pd.DataFrame):
            Q1 = data_to_process.quantile(0.25)
            Q3 = data_to_process.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((data_to_process < (Q1 - threshold * IQR)) | 
                    (data_to_process > (Q3 + threshold * IQR))).any(axis=1)
            return data_to_process[mask]
        else:
            Q1 = np.percentile(data_to_process, 25, axis=0)
            Q3 = np.percentile(data_to_process, 75, axis=0)
            IQR = Q3 - Q1
            mask = ~((data_to_process < (Q1 - threshold * IQR)) | 
                    (data_to_process > (Q3 + threshold * IQR))).any(axis=1)
            return data_to_process[mask]
    
    elif method == "zscore":
        if isinstance(data_to_process, pd.DataFrame):
            z_scores = np.abs((data_to_process - data_to_process.mean()) / data_to_process.std())
            mask = (z_scores < threshold).all(axis=1)
            return data_to_process[mask]
        else:
            z_scores = np.abs((data_to_process - np.mean(data_to_process, axis=0)) / 
                            np.std(data_to_process, axis=0))
            mask = (z_scores < threshold).all(axis=1)
            return data_to_process[mask]
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def handle_missing_values(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "mean",
    fill_value: Optional[float] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Handle missing values in the dataset.
    
    Args:
        data: Input data
        method: Strategy for handling missing values - 'mean', 'median', 'forward_fill', 'drop', or 'fill'
        fill_value: Value to fill when method='fill'
        
    Returns:
        Data with missing values handled
    """
    if isinstance(data, pd.DataFrame):
        data_copy = data.copy()
        
        if method == "mean":
            return data_copy.fillna(data_copy.mean())
        elif method == "median":
            return data_copy.fillna(data_copy.median())
        elif method == "forward_fill":
            return data_copy.fillna(method='ffill').fillna(method='bfill')
        elif method == "drop":
            return data_copy.dropna()
        elif method == "fill" and fill_value is not None:
            return data_copy.fillna(fill_value)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # For numpy arrays
        if method == "drop":
            return data[~np.isnan(data).any(axis=1)]
        elif method == "mean":
            col_means = np.nanmean(data, axis=0)
            mask = np.isnan(data)
            data[mask] = np.take(col_means, np.where(mask)[1])
            return data
        elif method == "fill" and fill_value is not None:
            data_copy = data.copy()
            data_copy[np.isnan(data_copy)] = fill_value
            return data_copy
        else:
            raise ValueError(f"Unknown method for numpy arrays: {method}")


def validate_data(
    data: Union[np.ndarray, pd.DataFrame],
    check_nan: bool = True,
    check_inf: bool = True,
    check_empty: bool = True
) -> bool:
    """
    Validate data quality.
    
    Args:
        data: Input data to validate
        check_nan: Check for NaN values
        check_inf: Check for infinite values
        check_empty: Check if data is empty
        
    Returns:
        True if data passes all checks, False otherwise
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    if check_empty and data_array.size == 0:
        logger.warning("Data is empty")
        return False
    
    if check_nan and np.isnan(data_array).any():
        logger.warning("Data contains NaN values")
        return False
    
    if check_inf and np.isinf(data_array).any():
        logger.warning("Data contains infinite values")
        return False
    
    return True


def split_features_target(
    data: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target variable from a DataFrame.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features_DataFrame, target_Series)
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y


def resample_time_series(
    data: pd.DataFrame,
    sampling_rate: str = "1S",
    method: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series data.
    
    Args:
        data: Time-indexed DataFrame
        sampling_rate: New sampling rate (e.g., '1S', '1min')
        method: Resampling method - 'mean', 'median', 'first', 'last'
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Attempting to convert.")
        data.index = pd.to_datetime(data.index)
    
    if method == "mean":
        return data.resample(sampling_rate).mean()
    elif method == "median":
        return data.resample(sampling_rate).median()
    elif method == "first":
        return data.resample(sampling_rate).first()
    elif method == "last":
        return data.resample(sampling_rate).last()
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def create_sliding_window(
    data: Union[np.ndarray, pd.DataFrame],
    window_size: int,
    step_size: int = 1
) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input data
        window_size: Size of each window
        step_size: Step size between windows
        
    Returns:
        Array of sliding windows
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)
