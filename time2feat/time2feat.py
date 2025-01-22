from typing import Optional

import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from time2feat.utils.importance_old import feature_selection
from time2feat.extraction.extractor import feature_extraction
from time2feat.model.clustering import ClusterWrapper


class Time2Feat(object):

    def __init__(self, n_clusters: Optional[int] = None, batch_size=100, p=1, model_type='KMeans', transform_type='std',
                 score_mode='simple',
                 strategy='sk_base',
                 k_best=False, pre_transform=False, top_k=None, pfa_value=0.9):
        """
        Initialize time2feat method with specified parameters.

        Parameters:
        - n_clusters (int): Number of clusters to be used. If None, the number of selected features will be used.
        - batch_size (int, optional): Batch size for processing (default is 100).
        - p (int, optional): Some parameter `p` (default is 1).
        - model_type (str, optional): Type of model to use (default is 'kMeans').
        - transform_type (str, optional): Type of data normalization (default is 'std').
        - score_mode (str, optional): Mode for scoring (default is 'simple').
        - strategy (str, optional): Strategy of features selection to be used (default is 'sk_base').
        - k_best (bool, optional): Whether to use k-best selection (default is False).
        - pre_transform (bool, optional): Whether to apply pre-transformation (default is False).
        - top_k (list of int, optional): List of top-k values (default is [1]).
        - pfa_value (float, optional): Some value `pfa_value` (default is 0.9).
        """
        if top_k is None:
            top_k = [1]
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.p = p
        self.model_type = model_type
        self.transform_type = transform_type
        self.score_mode = score_mode
        self.strategy = strategy
        self.k_best = k_best
        self.pre_transform = pre_transform
        self.top_k = top_k
        self.pfa_value = pfa_value
        self.top_feats = list()
        self.top_ext_feats = list()

    def fit(self, X, labels=None, external_feat: pd.DataFrame = None, select_external_feat: bool = False):
        if labels is None:
            labels = dict()
        df_feats = feature_extraction(X, batch_size=self.batch_size, p=self.p)
        context = {'model_type': self.model_type, 'transform_type': self.transform_type, 'score_mode': self.score_mode,
                   'strategy': self.strategy, 'k_best': self.k_best, 'pre_transform': self.pre_transform,
                   'top_k': self.top_k, 'pfa_value': self.pfa_value}
        top_features = feature_selection(
            df_feats,
            labels=labels,
            context=context,
            external_feat=external_feat,
            select_external_feat=select_external_feat
        )

        if external_feat is not None and select_external_feat:
            top_feats_ts, top_feats_ext = top_features
            df_feats = pd.concat([df_feats[top_feats_ts], external_feat[top_feats_ext]], axis=1)

            self.top_feats = top_feats_ts
            self.top_ext_feats = top_feats_ext
        else:
            df_feats = df_feats[top_features]
            self.top_feats = top_features
        """if external_feat is not None:
            df_feats = pd.concat([df_feats, external_feat], axis=1)"""

        # return df_feats

        # scaler = StandardScaler()
        # df_standardized = scaler.fit_transform(df_feats)
        #
        # # PCA on the selected features
        # pca = PCA()
        # pca.fit(df_standardized)
        #
        # # get the variance for the variables
        # explained_variance = pca.explained_variance_ratio_
        # cumulative_explained_variance = explained_variance.cumsum()

        # select significant features
        # n_components = next(i for i, total_var in enumerate(cumulative_explained_variance) if total_var >= 0.95) + 1

        # Get the principal components
        # components = pca.components_[:n_components]

        # Get the most important features for each component
        # most_important_features = []
        # for component in components:
        #     feature_indices = component.argsort()[-n_components:][::-1]
        #     most_important_features.extend(df_feats.columns[feature_indices])

        # Remove duplicates while preserving order
        # most_important_features = list(dict.fromkeys(most_important_features))

        # Select the most important features from the DataFrame
        # df_selected_features = df_feats[most_important_features]

        # print(f"Selected {n_components} features: {most_important_features}")

        # return df_selected_features

        if self.n_clusters is None:
            scaler = StandardScaler()
            df_standardized = scaler.fit_transform(df_feats)

            # PCA on the selected features
            pca = PCA()
            pca.fit(df_standardized)

            # get the variance for the variables
            explained_variance = pca.explained_variance_ratio_

            n_clusters = KneeLocator(np.asarray([(range(len(explained_variance)))]).squeeze(),
                                     np.asarray(explained_variance).squeeze(),
                                     curve="convex", direction="decreasing").knee

        else:
            n_clusters = self.n_clusters

        return df_feats, n_clusters

    def fit_predict(self, X, labels=None, external_feat: pd.DataFrame = None, select_external_feat: bool = False):
        if labels is None:
            labels = {}
        df_feats, n_clusters = self.fit(X, labels, external_feat, select_external_feat=select_external_feat)
        model = ClusterWrapper(n_clusters=n_clusters, model_type=self.model_type,
                               transform_type=self.transform_type)
        y_pred = model.fit_predict(df_feats)
        return y_pred
