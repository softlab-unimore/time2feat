import numpy as np
import pandas as pd

from collections import Counter


def features_selection_old(df_features: pd.DataFrame, ranks: pd.Series, top_k: int, mode: str = 'simple'):
    """ Select the k most important features in the given mode """
    features = ranks.index.values
    if mode == 'simple':
        # Extract the importance for each feature
        scores = ranks.values
        # Order importance in descending order
        order = np.argsort(scores)[::-1]
        # Select top k feature
        selection = features[order[:top_k]]

    elif mode == 'domain':
        # Extract all domains
        domains = dict(Counter([x.split('__')[0] for x in features]))
        # Split top k for each domain
        top_k_partial = top_k // len(domains)
        # ToDo: check possible scenario (top k partial greater than other features
        # top_types = {k: val if val < top_k_partial else top_k_partial for k, val in count_types.items()}
        selection = []
        for fd in domains.keys():
            # Extract features names and scores in the given domain
            fd_features = np.array([x for x in features if x.startswith(fd)])
            scores = ranks[fd_features].values

            # Order and select the top features
            order = np.argsort(scores)[::-1]
            selection.append(fd_features[order[:top_k_partial]])

        selection = np.concatenate(selection)

    else:
        raise ValueError('Mode {} not supported'.format(mode))

    return df_features[selection]
