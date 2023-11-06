import numpy as np
import pandas as pd
from tsfresh import extract_features


# Adapt Multivariate time series for extracting features.
def adapt_time_series(ts, sensors_name):
    list_multi_ts = {
        k: ts[:, i].tolist()
        for i, k in enumerate(sensors_name)
    }

    list_id = [1 for _ in ts]
    list_time = [i for i in range(len(ts))]

    dict_df = {'id': list_id, 'time': list_time}
    for sensor in sensors_name:
        dict_df[sensor] = list_multi_ts[sensor]

    df_time_series = pd.DataFrame(dict_df)
    return df_time_series


def extract_univariate_features(ts: np.array, sensors_name: list, feats_select: dict = None):
    dict_ts = adapt_time_series(ts, sensors_name)
    features_extracted = extract_features(
        dict_ts,
        column_id='id',
        column_sort='time',
        n_jobs=0,
        kind_to_fc_parameters=feats_select
    )
    features = features_extracted.T.iloc[:, 0].astype(float).to_dict()
    return features
