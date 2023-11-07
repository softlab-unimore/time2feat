from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split


def select_labels(
        x: np.ndarray,
        y: np.ndarray,
        method: Literal['random'],
        size: float,
) -> dict:
    """
    Selects a subset of labels from the given labels array based on a specified method.

    This function currently supports random selection of labels. It returns a dictionary
    with the selected indices from 'x' and their corresponding labels from 'y'.

    Args:
        x: np.ndarray
            An array of samples. Only the length of 'x' is used to determine the indices to be selected.
        y: np.ndarray
            An array of labels corresponding to the samples in 'x'.
        method: Literal['random']
            The method used to select the labels. Currently, only 'random' is supported.
        size: float
            The proportion of the dataset to include in the train split. Should be between 0.0 and 1.0.

    Returns:
        dict
            A dictionary where keys are the indices of the selected samples and values are the corresponding labels.

    Raises:
        ValueError: If the method provided is not supported.

    Examples:
        > x = np.array([0, 1, 2, 3, 4])
        > y = np.array(['a', 'b', 'c', 'd', 'e'])
        > select_labels(x, y, method='random', size=0.6)
        {0: 'a', 2: 'c', 3: 'd'}
    """
    if method == 'random':
        # Extract a subset of labelled data with random sampling
        idx_train, _, y_train, _ = train_test_split(np.arange(len(x)), y, train_size=size)
        labels = {i: j for i, j in zip(idx_train, y_train)}
        return labels
    else:
        # ToDo FDB: insert active learning strategy
        raise ValueError(f'{method} is not supported, must be random')
