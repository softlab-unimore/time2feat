import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score

from .preprocessing import get_transformer


def _define_model(model_type: str, num_cluster: int):
    """ Define the clustering model """
    if model_type == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=num_cluster)
    elif model_type == 'KMeans':
        model = KMeans(n_clusters=num_cluster, n_init=10)
    elif model_type == 'Spectral':
        model = SpectralClustering(n_clusters=num_cluster)
    else:
        raise ValueError('{} is not supported'.format(model_type))
    return model


def cluster_metrics(y_true: np.array, y_pred: np.array):
    """ Compute main clustering metrics """
    return {
        'ami': adjusted_mutual_info_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred),
        'rand': adjusted_rand_score(y_true, y_pred),
        'homogeneity': homogeneity_score(y_true, y_pred),
    }


class ClusterWrapper(object):
    """ Wrapper for several clustering algorithms """

    def __init__(self, n_clusters: int, model_type: str, transform_type: str = None, normalize: bool = False):
        self.num_cluster = n_clusters
        self.model_type = model_type
        self.normalize = normalize

        self.model = _define_model(model_type, n_clusters)
        self.transform_type = transform_type

    def _normalize(self, x: np.array):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        x = (x - x_mean) / x_std
        return x

    def remove_nan(self, x: np.array):
        if len(x.shape) == 2:
            count = np.isnan(x).any(axis=0).sum()
            if count > 0:
                print('Remove {} nan columns for clustering step'.format(count))
                cond = np.logical_not(np.isnan(x).any(axis=0))
                x = x[:, cond]

            cond = ((x == float('inf')) | (x == float('-inf'))).any(axis=0)

        return x

    def fit_predict(self, x: np.array):
        x = self.remove_nan(x)
        if self.normalize:
            x = self._normalize(x)
        elif self.transform_type:
            transformer = get_transformer(self.transform_type)
            x = transformer.fit_transform(x)
        x = x.reshape((len(x), -1))
        return self.model.fit_predict(x)
