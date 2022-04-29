import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from t2f.dataset import read_ucr_dataset, update_result
from t2f.window import window_selection, prepare_data
from t2f.extractor import feature_extraction
from t2f.auto import simple_grid_search
from t2f.importance import features_scoring_selection, features_simple_selection
from t2f.preprocessing import apply_transformation
from t2f.clustering import ClusterWrapper, cluster_metrics


def time_difference(t: datetime):
    t_new = datetime.now()
    print('Time: {}'.format(t_new - t))
    return t_new


def pipeline(params: dict):
    # Input and output folder
    data_dir = params['data_dir']
    output_dir = params['output_dir']

    # Model params
    transform_type = params['transform_type']
    model_type = params['model_type']

    # Performance params
    train_size = params['train_size']
    batch_size = params.get('batch_size', 500)
    p = params.get('p', 1)

    # Simple consistency check
    if not os.path.isdir(data_dir) or not os.path.isdir(output_dir):
        raise ValueError('Dataset and/or output folder don\'t exist')

    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    # Record pipeline params
    monitor_dict = {
        'data_dir': os.path.basename(data_dir),
        'train_size': train_size,
        'model_type': model_type,
        'transform_type': transform_type
    }

    print('Read dataset: ', data_dir)
    ts_list, y = read_ucr_dataset(path=data_dir)

    labels = {}
    if train_size > 0:
        # Extract a subset of labelled mts to train the semi-supervised model
        idx_train, y_train, _, _ = train_test_split(np.arange(len(ts_list)), y, train_size=train_size)
        labels = {i: j for i, j in zip(idx_train, y_train)}

    df_features = feature_extraction


if __name__ == '__main__':
    my_params = {
        'data_dir': r'C:\Users\delbu\Projects\Dataset\Multivariate_TS\BasicMotions',  # BasicMotions
        'output_dir': './',

        'train_size': 0,
        'batch_size': 100,
        'p': 1,

        'auto': True,
        'top_k': 100,
        'score_mode': 'simple',  # 'simple', 'domain'
        'transform_type': None,  # None, 'std', 'minmax', 'robust'
        'model_type': 'KMeans',  # 'HDBSCAN', 'AgglomerativeClustering', 'KMeans', 'SpectralClustering'

    }
    res = pipeline(my_params)
