import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score

from t2f.dataset import read_ucr_dataset
from t2f.extractor import feature_extraction
from t2f.importance import feature_selection
from t2f.clustering import ClusterWrapper


def pipeline(params: dict):
    # Input and output folder
    data_dir = params['data_dir']

    # Model params
    transform_type = params['transform_type']
    model_type = params['model_type']

    # Performance params
    train_size = params['train_size']
    batch_size = params.get('batch_size', 500)
    p = params.get('p', 1)

    # Simple consistency check
    if not os.path.isdir(data_dir):
        raise ValueError('Dataset folder don\'t exist')

    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    print('Read ucr dataset: ', data_dir)
    ts_list, y_true = read_ucr_dataset(path=data_dir)
    n_clusters = len(set(y_true))  # Get number of clusters to find

    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))

    labels = {}
    if train_size > 0:
        # Extract a subset of labelled mts to train the semi-supervised model
        idx_train, _, y_train, _ = train_test_split(np.arange(len(ts_list)), y_true, train_size=train_size)
        labels = {i: j for i, j in zip(idx_train, y_train)}
        print('Number of Labels: {}'.format(len(labels)))

    print('Feature extraction')
    df_features = feature_extraction(ts_list, batch_size, p)
    print('Number of extracted features: {}'.format(df_features.shape[1]))

    print('Feature selection')
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_features = feature_selection(df_features, labels, context)
    df_features = df_features[top_features]
    print('Number of selected features: {}'.format(df_features.shape[1]))

    print('Clustering')
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_features.values)

    print('AMI: {:0.4f}'.format(adjusted_mutual_info_score(y_true, y_pred)))


if __name__ == '__main__':
    my_params = {
        'data_dir': r'data/Cricket',  # BasicMotions

        'train_size': 0,
        'batch_size': 500,
        'p': 1,

        'transform_type': 'minmax',  # None, 'std', 'minmax', 'robust'
        'model_type': 'Hierarchical',  # 'Hierarchical', 'KMeans', 'Spectral'

    }
    res = pipeline(my_params)
