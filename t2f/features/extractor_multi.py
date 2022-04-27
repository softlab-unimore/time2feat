import numpy as np

"""
iscipy.stats.binned_statistic_2d.html#scipy.stats.binned_statistic_2d
"""

def extract_multivariate_features(ts: np.array):
    assert len(ts.shape) == 2, 'ts is not a multivariate time series'

    # ToDO: add all feature transformation functions on multivariate time series
    features = {
        'max': np.max(ts)
    }

    return features
