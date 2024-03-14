from typing import List, Literal, Optional, Union, Tuple
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from t2f.data.dataset import read_ucr_datasets, encode_labels
from t2f.data.train import select_labels
from t2f.extraction.extractor import feature_extraction
from t2f.selection.selection import feature_selection
from t2f.model.clustering import ClusterWrapper, cluster_metrics


def build_feat_path(checkpoint_dir, ts_files):
    # Constructs the feature file path from the checkpoint directory and time series files
    files_name = "_".join([os.path.basename(x) for x in ts_files]) + '.pickle'
    return os.path.join(checkpoint_dir, files_name)


def feature_extraction_with_checkpoint(
        ts_list: Union[list, np.array],
        intra_type: Literal['tsfresh'],
        inter_type: Literal['distance'],
        batch_size: int = -1,
        p: int = 1,
        checkpoint_dir: str = None,
        ts_files: list = None,
) -> pd.DataFrame:
    assert (checkpoint_dir is not None and ts_files is not None) or checkpoint_dir is None

    # Check if the checkpoint directory is provided
    if checkpoint_dir:

        # Crete checkpoint dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the full path for the features file
        feat_path = build_feat_path(checkpoint_dir, ts_files)

        # Check if the features file already exists
        if os.path.exists(feat_path):
            # If the file exists, load the features from it and return
            with open(feat_path, 'rb') as f:
                return pickle.load(f)['df_features']

    # If the checkpoint file does not exist, extract features using the provided parameters
    df_features = feature_extraction(
        ts_list=ts_list,
        intra_type=intra_type,
        inter_type=inter_type,
        batch_size=batch_size,
        p=p
    )

    # If a checkpoint directory is provided, save the extracted features to a file
    if checkpoint_dir:
        feat_path = build_feat_path(checkpoint_dir, ts_files)
        with open(feat_path, 'wb') as f:
            pickle.dump({'df_features': df_features}, f)

    # Return the extracted features
    return df_features


def pipeline(
        files: List[str],
        intra_type: Literal['tsfresh'],
        inter_type: Literal['distance'],
        transform_type: Optional[Literal['standard', 'minmax', 'robust']],
        model_type: Literal['Hierarchical', 'KMeans', 'Spectral'],
        ranking_type: List[str] = None,  # ['anova']
        ranking_pfa: Optional[float] = 0.9,
        ensemble_type: str = None,
        search_type: Optional[Literal['fixed', 'linear']] = 'fixed',
        train_type: Literal['random'] = 'random',
        train_size: float = 0,
        batch_size: int = 500,
        p: int = 1,
        checkpoint_dir: Optional[str] = None,
        random_seed: Optional[int] = None
) -> Tuple[dict, pd.DataFrame]:
    # Simple consistency check
    if [x for x in files if not os.path.isfile(x)]:
        raise ValueError('At least on time-series path does not exist')
    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    print('Read ucr datasets: ', files)
    ts_list, y_true = read_ucr_datasets(paths=files)
    y_true = encode_labels(y_true)
    n_clusters = len(set(y_true))  # Get number of clusters to find
    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))

    labels = {}
    if train_size > 0:
        if random_seed:
            np.random.seed(random_seed)
        labels = select_labels(x=ts_list, y=y_true, method=train_type, size=train_size)
        print('Number of Labels: {}'.format(len(labels)))

    print('Feature extraction')
    # df_features = feature_extraction(ts_list, intra_type, inter_type, batch_size, p)
    df_features = feature_extraction_with_checkpoint(
        checkpoint_dir=checkpoint_dir, ts_files=files,
        ts_list=ts_list, intra_type=intra_type, inter_type=inter_type, batch_size=batch_size, p=p
    )
    print('Number of extracted features: {}'.format(df_features.shape[1]))
    print(f'Feature selection: {ranking_type}')
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_features, transform_type, df_debug = feature_selection(
        df=df_features,
        labels=labels,
        ranking_type=ranking_type,
        ensemble_type=ensemble_type,
        pfa_variance=ranking_pfa,
        search_type=search_type,
        context=context,
        y_true=list(y_true)
    )
    df_features = df_features[top_features]
    print('Number of selected features: {}'.format(df_features.shape[1]))

    print('Clustering')
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_features.values)

    print('AMI: {:0.4f}'.format(adjusted_mutual_info_score(y_true, y_pred)))
    print('NMI: {:0.4f}'.format(normalized_mutual_info_score(y_true, y_pred)))

    return cluster_metrics(y_true, y_pred), df_debug


RANKING = [
    'anova', 'fisher_score'  'laplace_score', 'trace_ratio100', 'trace_ratio',
    'mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr',
    'rfs', 'mcfs', 'udfs', 'ndfs', 'gini', 'cfs'
]

if __name__ == '__main__':
    dataset = 'BasicMotions'
    for i in range(100):
        print(f"\nRun {i}")
        pipeline(
            files=[f'data/{dataset}/{dataset}_TEST.txt', f'data/{dataset}/{dataset}_TRAIN.txt'],
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=["anova"],
            ensemble_type="average",
            train_type='random',
            train_size=0.2,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir='./checkpoint',
            random_seed=42 + i
        )
    print('Hello World!')
