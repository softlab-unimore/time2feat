import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score

import matplotlib.pyplot as plt

from .clustering import ClusterWrapper, cluster_metrics


def get_sliding_window_matrix(data: np.array, window: int, stride: int):
    """ Apply sliding window strategy to input matrix """
    # Extra array dimensions
    rows, cols = data.shape
    if window == 0:
        # Special case which return a single row array
        window = rows

    # Compute new window matrix rows
    new_rows = 1 + (rows - window) // stride

    matrix = np.zeros((new_rows, window, cols))

    for i in range(new_rows):
        left = i * stride
        right = left + window
        matrix[i, :, :] = data[left:right, :]

    return matrix


def prepare_data(ts_list: list, labels: list, kernel: int, stride: int = 1):
    """ Concatenate all time series with several labels to extract x and y  """
    # Create slide window matrix for each time series
    x_list = [get_sliding_window_matrix(ts.values, kernel, stride) for ts in ts_list]

    # Assign labels for each matrix values
    y = np.hstack([[i] * len(x) for i, x in zip(labels, x_list)])
    x = np.vstack(x_list)

    return x, y


def compute_simple_cluster_ranking(y_true: np.array, y_pred: np.array):
    """ Compute three clustering metrics and return the average """
    metrics = cluster_metrics(y_true, y_pred)
    return np.mean([metrics['ami'], metrics['rand'], metrics['homogeneity']])


def window_selection(ts_list: list, labels: list):
    """ Select the window with the highest information content """
    assert len(ts_list) == len(labels), 'Number of time series and labels are no equal'
    # Extract length for each time series
    sizes = [len(ts) for ts in ts_list]
    mean_size = np.median(sizes)
    min_size = np.min(sizes)

    # Simple strategy to define several window possibilities
    num = 20
    windows = [int(mean_size * (scale / num)) for scale in range(1, num + 1)]
    windows = [window for window in windows if window <= min_size]

    # Extract the number of labels
    num_labels = len(set(labels))

    ranks = []

    time.sleep(.5)
    for window in tqdm(windows):
        # Apply sliding window process and extract x and y arrays
        x, y = prepare_data(ts_list=ts_list, labels=labels, kernel=window)

        # Apply a fast clustering algorithm
        model = ClusterWrapper(n_clusters=num_labels, model_type='HDBSCAN')
        y_pred = model.fit_predict(x)

        # Compute current clustering score
        rank = compute_simple_cluster_ranking(y, y_pred)
        ranks.append(rank)

    plt.plot(windows, ranks, 'ro')
    plt.show()

    max_idx = np.argmax(ranks)
    return windows[max_idx]
