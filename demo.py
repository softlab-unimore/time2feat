from typing import List, Literal, Optional
import os
from sklearn.metrics import adjusted_mutual_info_score

from t2f.data.dataset import read_ucr_datasets
from t2f.data.train import select_labels
from t2f.extraction.extractor import feature_extraction
from t2f.selection.selection import feature_selection
from t2f.model.clustering import ClusterWrapper


def pipeline(
        files: List[str],
        intra_type: Literal['tsfresh'],
        inter_type: Literal['distance'],
        transform_type: Optional[Literal['std', 'minmax', 'robust']],
        model_type: Literal['Hierarchical', 'KMeans', 'Spectral'],
        ranking_type: Optional[List[Literal['anova']]] = None,
        ensemble_type: Optional[Literal['average', 'reciprocal_rank_fusion', 'condorcet_fuse']] = None,
        train_type: Literal['random'] = None,
        train_size: float = 0,
        batch_size: int = 500,
        p: int = 1
) -> None:
    # Simple consistency check
    if [x for x in files if not os.path.isfile(x)]:
        raise ValueError('At least time-series path don\'t exist')
    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    print('Read ucr datasets: ', files)
    ts_list, y_true = read_ucr_datasets(paths=files)
    n_clusters = len(set(y_true))  # Get number of clusters to find

    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))

    labels = {}
    if train_size > 0:
        labels = select_labels(x=ts_list, y=y_true, method=train_type, size=train_size)
        print('Number of Labels: {}'.format(len(labels)))

    print('Feature extraction')
    df_features = feature_extraction(ts_list=ts_list, intra_type=intra_type, inter_type=inter_type, batch_size=batch_size, p=p)
    print('Number of extracted features: {}'.format(df_features.shape[1]))

    print('Feature selection')
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_features = feature_selection(df=df_features, labels=labels, ranking_type=ranking_type, ensemble_type=ensemble_type, context=context)
    df_features = df_features[top_features]
    print('Number of selected features: {}'.format(df_features.shape[1]))

    print('Clustering')
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_features.values)

    print('AMI: {:0.4f}'.format(adjusted_mutual_info_score(y_true, y_pred)))


if __name__ == '__main__':
    pipeline(
        files=['data/BasicMotions/BasicMotions_TRAIN.txt', 'data/BasicMotions/BasicMotions_TEST.txt'],
        intra_type='tsfresh',
        inter_type='distance',
        transform_type='minmax',
        model_type='Hierarchical',
        ranking_type=['anova', 'anova'],
        ensemble_type='condorcet_fuse',
        train_type='random',
        train_size=0.3,
        batch_size=500,
        p=4,
    )
    print('Hello World!')
