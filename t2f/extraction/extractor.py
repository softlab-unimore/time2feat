import os
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from .extractor_single import extract_univariate_features
from .extractor_pair import extract_pair_features


def padding_series(ts_list: list):
    def padding(ts: np.array, new_length: int):
        ts_padded = np.empty((new_length, ts.shape[1]))
        ts_padded[:] = None
        ts_padded[:len(ts), :] = ts
        return ts_padded

    max_length = np.max([len(ts) for ts in ts_list])
    ts_list = [padding(ts, max_length) for ts in ts_list]
    return ts_list


def extract_single_series_features(record: np.array, sensors_name: list):
    # Extract intra-signal features
    # Extract univariate features for each signal in the multivariate time series
    features_extracted = extract_univariate_features(record, sensors_name)
    # Rename key by inserting the predefined name
    features_single = {'single__{}'.format(k): val for k, val in features_extracted.items()}

    return features_single


def extract_single_series_features_batch(ts_list: np.array, batch_size: int = -1, pid: int = 1):
    """ Extract features for each signal in each time series in batch mode """
    sensors_name = [str(i) for i in range(ts_list.shape[2])]
    ts_list = [arr for arr in ts_list]

    if batch_size == -1:
        batch_size = len(ts_list)

    num_batch = int(np.ceil(len(ts_list) / batch_size))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.path.join(curr_dir, '../checkpoint/')
    # print('Start univariate feature extraction in batch : pid {}'.format(pid))
    for i in range(num_batch):
        a = i * batch_size
        b = (i + 1) * batch_size
        ts_batch = ts_list[a:b]

        # Concatenate record and rename sensor names
        record = np.hstack(ts_batch)
        tmp_sensors_name = ["{}aaa{}".format(i, name) for i in range(len(ts_batch)) for name in sensors_name]

        # Extract univariate features for each signal
        # print('Start univariate feature extraction: pid {} batch {}'.format(pid, i))
        features_extracted = extract_univariate_features(record, tmp_sensors_name)
        # print('End univariate feature extraction: pid {} batch {}'.format(pid, i))

        # Reshape extracted features in each signals and format them correctly for each multivariate time series
        df_features = [{} for _ in range(len(ts_batch))]
        for k, val in features_extracted.items():
            pos, name = k.split('aaa', 1)
            pos = int(pos)
            name = 'single__{}'.format(name)
            df_features[pos][name] = val

        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        pd.DataFrame(df_features).to_csv(filename, index=False)

    # print('End univariate feature extraction in batch : pid {}'.format(pid))

    # Construct the complete feature extracted dataset
    # print('Start construction univariate dataset : pid {}'.format(pid))

    df_features = []
    for i in range(num_batch):
        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        df = pd.read_csv(filename)
        df_features.append(df)
        os.remove(filename)

    df_features = pd.concat(df_features, axis=0, ignore_index=True)

    # print('End construction univariate dataset : pid {}'.format(pid))

    return df_features


def extract_pair_series_features(mts: np.array):
    """ Extract features for each time series pair """
    features_pair = {}  # Initialize an empty dictionary to save extracted features

    # Extract each possible combination pair
    indexes = np.arange(mts.shape[1])
    combs = list(itertools.combinations(indexes, r=2))
    for i, j in combs:
        # Extract pair features for each pair in the multivariate time series
        feature = extract_pair_features(mts[:, i], mts[:, j])

        # Rename key by inserting the feature name
        feature = {'pair__{}__{}__{}'.format(k, i, j): val for k, val in feature.items()}

        features_pair.update(feature)
    return features_pair


def feature_extraction_simple(ts_list: (list, np.array), batch_size: int = -1, pid: int = 1):
    """ Extract intra and inter-signals features for each multivariate time series """
    ts_features_list = []

    # Extract features based on pair feature functions
    for ts_record in ts_list:
        features_pair = extract_pair_series_features(ts_record)
        ts_features_list.append(features_pair)

    # Create dataframe for pair features
    df_pair_features = pd.DataFrame(ts_features_list)

    # Extract features based on functions for univariate signals in batch mode
    df_single_features = extract_single_series_features_batch(ts_list, batch_size=batch_size, pid=pid)

    # Create time series feature dataFrame and return
    df_features = pd.concat([df_single_features, df_pair_features], axis=1)

    return df_features


def get_balanced_job(number_pool, number_job):
    """ Define number of records to assign to each processor """
    list_num_job = []
    if number_job <= number_pool:
        for i in range(number_job):
            list_num_job.append(1)
    else:
        for i in range(number_pool):
            list_num_job.append(int(number_job / number_pool))
        for i in range(number_job % number_pool):
            list_num_job[i] = list_num_job[i] + 1

    return list_num_job


def feature_extraction(ts_list: (list, np.array), batch_size: int = -1, p: int = 1):
    """ Multiprocessing implementation of the feature extraction step """
    # Define the number of processors to use0
    max_pool = mp.cpu_count() if p == -1 else p
    num_batch = (len(ts_list) // batch_size) + 1
    max_pool = num_batch if num_batch < max_pool else max_pool

    # ToDo FDB: improve balance records between jobs
    balance_job = get_balanced_job(number_pool=max_pool, number_job=len(ts_list))
    # print('Feature extraction with {} processor and {} batch size'.format(max_pool, batch_size))

    index = 0
    list_arguments = []
    for i, size in enumerate(balance_job):
        list_arguments.append((ts_list[index:index + size], batch_size, i))
        index += size

    # Multi processing script execution
    pool = mp.Pool(max_pool)
    extraction_feats = pool.starmap(feature_extraction_simple, list_arguments)
    pool.close()
    pool.join()

    # Concatenate all results
    df_features = pd.concat(extraction_feats, axis=0, ignore_index=True)
    return df_features
