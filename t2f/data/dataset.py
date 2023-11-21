from typing import List, Tuple
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .reader import load_from_tsfile_to_dataframe


def read_ucr_mts(path: str) -> Tuple[List[np.ndarray], list]:
    """ Wrapper for sktime load_from_tsfile_to_dataframe function """
    # Check mts existence
    if not os.path.isfile:
        raise ValueError(f"THe multivariate time-series file doesn't exist: {path}")

    # Read multivariate time series (mts)
    df, y = load_from_tsfile_to_dataframe(path)

    # Extract list of mts array
    df = df.map(lambda val: val.to_list())
    ts_list = df.apply(lambda row: np.array(row.to_list()).T, axis=1).to_list()

    # Check mts consistency
    assert len(y) == len(ts_list), 'X and Y have different size'
    cond = [len(ts.shape) == 2 for ts in ts_list]
    assert np.all(cond), 'Error in time series format, shape must be 2d'
    return ts_list, list(y)


def read_ucr_datasets(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """ Read ucr datasets """
    # Read list of ucr multivariate time-series
    ts_list = []
    y_list = []
    for path in paths:
        ts, y = read_ucr_mts(path)
        ts_list += ts
        y_list += y

    ts_list = np.array(ts_list)
    y = np.array(y_list)

    # Transform y array into numeric form
    # y = pd.get_dummies(y).sort_index(axis=1).apply(np.argmax, axis=1).values.astype('int')

    return ts_list, y


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode a numpy array of categorical labels into a numeric format.

    Args:
        y (np.ndarray): A numpy array of categorical labels.

    Returns:
        np.ndarray: An array of encoded labels, where each unique category in `y`
        is mapped to an integer.

    Examples:
        > encode_labels(np.array(['apple', 'banana', 'apple', 'orange']))
        array([0, 1, 0, 2])
    """
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit label encoder and return encoded labels
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded
