import os
import itertools
import numpy as np

import pandas as pd

import multiprocessing as mp

from .features.extractor_single import extract_univariate_features
from .features.extractor_multi import extract_multivariate_features
from .features.extractor_pair import extract_pair_features


def get_balanced_job(number_pool, number_job):
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


def extract_single_series_features(record: np.array, sensors_name: list):
    # Extract features_single based on a single time series
    # Extract univariate features for each signal in the multivariate time series
    features_extracted = extract_univariate_features(record, sensors_name)
    # Rename key by inserting the predefined name
    features_single = {'single__{}'.format(k): val for k, val in features_extracted.items()}

    return features_single


def extract_single_series_features_batch(ts_list: list, sensors_name: list, feats_select: list = None,
                                         batch_size: int = -1, pid: int = int):
    # Define batch
    ts_list = [arr for arr in ts_list]

    if batch_size == -1:
        batch_size = len(ts_list)

    num_batch = int(np.ceil(len(ts_list) / batch_size))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.path.join(curr_dir, 'data/')
    print('Start univariate feature extraction in batch : pid {}'.format(pid))
    for i in range(num_batch):
        a = i * batch_size
        b = (i + 1) * batch_size
        ts_batch = ts_list[a:b]

        # Concatenate record and rename sensor names
        record = np.hstack(ts_batch)
        tmp_sensors_name = ["{}aaa{}".format(i, name) for i in range(len(ts_batch)) for name in sensors_name]

        dictFeatsChoose = None
        if feats_select is not None:
            dictFeatsChoose = {}
            for i in range(len(ts_list)):
                for key in feats_select['single'].keys():
                    dictFeatsChoose["{}aaa{}".format(i, key)] = feats_select['single'][key]

        # Extract univariate features for each signal
        print('Start univariate feature extraction: pid {} batch {}'.format(pid, i))
        features_extracted = extract_univariate_features(record, tmp_sensors_name, dictFeatsChoose)
        print('End univariate feature extraction: pid {} batch {}'.format(pid, i))

        # Reshape extracted features in each signals and format them correctly for each multivariate time series
        df_features = [{} for _ in range(len(ts_batch))]
        for k, val in features_extracted.items():
            pos, name = k.split('aaa', 1)
            pos = int(pos)
            name = 'single__{}'.format(name)
            df_features[pos][name] = val

        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        pd.DataFrame(df_features).to_csv(filename, index=False)

    print('End univariate feature extraction in batch : pid {}'.format(pid))

    # Construct the complete feature extracted dataset
    print('Start construction univariate dataset : pid {}'.format(pid))

    df_features = []
    for i in range(num_batch):
        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        df = pd.read_csv(filename)
        df_features.append(df)
        os.remove(filename)

    df_features = pd.concat(df_features, axis=0, ignore_index=True)

    print('End construction univariate dataset : pid {}'.format(pid))

    return df_features


def padding_series(ts_list: list):
    def padding(ts: np.array, new_length: int):
        ts_padded = np.empty((new_length, ts.shape[1]))
        ts_padded[:] = None
        ts_padded[:len(ts), :] = ts
        return ts_padded

    max_length = np.max([len(ts) for ts in ts_list])
    ts_list = [padding(ts, max_length) for ts in ts_list]
    return ts_list


def extract_single_series_features_all(ts_list: list, sensors_name: list, feats_select: list = None):
    # Define batch
    ts_list = [arr for arr in ts_list]
    if len(set([len(ts) for ts in ts_list])) > 1:
        ts_list = padding_series(ts_list)

    # Concatenate record and rename sensor names
    record = np.hstack(ts_list)
    tmp_sensors_name = ["{}aaa{}".format(i, name) for i in range(len(ts_list)) for name in sensors_name]

    dictFeatsChoose = None
    if feats_select is not None:
        dictFeatsChoose = {}
        for i in range(len(ts_list)):
            for key in feats_select['single'].keys():
                dictFeatsChoose["{}aaa{}".format(i, key)] = feats_select['single'][key]

    # Extract univariate features for each signal
    features_extracted = extract_univariate_features(record, tmp_sensors_name, dictFeatsChoose)

    # Reshape extracted features in each signals and format them correctly for each multivariate time series
    df_features = [{} for _ in range(len(ts_list))]
    for k, val in features_extracted.items():
        pos, name = k.split('aaa', 1)
        pos = int(pos)
        name = 'single__{}'.format(name)
        df_features[pos][name] = val

    df_features = pd.DataFrame(df_features)

    return df_features


def extract_multi_series_features(record: np.array):
    # Extract multivariate features from the multivariate time series
    feature = extract_multivariate_features(record)

    # Rename key by inserting the predefined name
    feature = {'multi__{}'.format(k): val for k, val in feature.items()}

    return feature


def extract_pair_series_features(record: np.array, sensors_name: list, feats_choose: list = None):
    # Extract feature based on each time series pair
    features_pair = {}
    if feats_choose is None:
        indexes = np.arange(len(sensors_name))
        # Extract each possible pair combination
        combs = list(itertools.combinations(indexes, r=2))

        for i, j in combs:
            fi = sensors_name[i]
            fj = sensors_name[j]

            # Extract pair features for each pair in the multivariate time series
            feature = extract_pair_features(record[:, i], record[:, j])

            # Rename key by inserting the feature name
            feature = {'pair__{}__{}__{}'.format(k, fi, fj): val for k, val in feature.items()}

            features_pair.update(feature)

    else:
        for info_feats in feats_choose:
            name_feats = info_feats.split('__')[1]
            sensors_one = info_feats.split('__')[2]
            sensors_two = info_feats.split('__')[3]
            i = sensors_name.index(sensors_one)
            j = sensors_name.index(sensors_two)
            feature = extract_pair_features(record[:, i], record[:, j], name_feats)
            # Rename key by inserting the feature name
            feature = {'pair__{}__{}__{}'.format(k, sensors_one, sensors_two): val for k, val in feature.items()}

            features_pair.update(feature)

    return features_pair


def feature_extractor(ts_list: list, sensors_name: list, feats_select: dict = None, batch_size: int = -1, pid: int = 1):
    """ Extract all features for each multivariate time series """
    ts_features_list = []
    # Extract only the given features, otherwise extract all of them
    feats_pair = feats_select['pair'] if feats_select is not None and 'pair' in feats_select else None
    feats_single = feats_select['single'] if feats_select is not None and 'single' in feats_select else None

    # Extract features based on pair feature function
    for ts_record in ts_list:
        features_pair = extract_pair_series_features(ts_record, sensors_name, feats_choose=feats_pair)
        ts_features_list.append(features_pair)

    # Create dataframe for pair features
    df_pair_features = pd.DataFrame(ts_features_list)

    # Extract features based on functions for univariate signals in batch mode
    df_single_features = extract_single_series_features_batch(ts_list, sensors_name, feats_select,
                                                              batch_size=batch_size, pid=pid)

    # Create time series feature dataFrame and return
    df_features = pd.concat([df_single_features, df_pair_features], axis=1)

    return df_features


def feature_extractor_multi(ts_list: list, sensors_name: list, feat_select: dict = None, batch_size: int = -1,
                            p: int = 1):
    """ Multi processing implementation of the feature extractor step """
    max_pool = mp.cpu_count() if p == -1 else p
    num_batch =( len(ts_list) // batch_size) + 1
    max_pool = num_batch if num_batch < max_pool else max_pool

    balance_job = get_balanced_job(number_pool=max_pool, number_job=len(ts_list))

    print('Feature extraction with {} processor and {} batch size'.format(max_pool, batch_size))

    index = 0
    list_arguments = []
    for i, size in enumerate(balance_job):
        list_arguments.append((ts_list[index:index + size], sensors_name, feat_select, batch_size, i))
        index += size

    pool = mp.Pool(max_pool)
    extraction_feats = pool.starmap(feature_extractor, list_arguments)
    pool.close()
    pool.join()

    df_features = pd.concat(extraction_feats, axis=0, ignore_index=True)
    return df_features

# def organize_feats(feats_extrc):
#     dict_model = {}
#     for list_models in feats_extrc:
#         for iteration in list_models:
#             for key in list_models[iteration]:
#                 if not key in dict_model.keys():
#                     dict_model[key] = []
#                 dict_model[key].append(list_models[iteration][key])
#     return dict_model


# def fix_feats_extracted(features_extracted):
#     df_features = {}
#     for jobs in features_extracted:
#         for key in jobs.keys():
#             name = key.split('aaa', 1)[1]
#             name = 'single__{}'.format(name)
#             if not name in df_features.keys():
#                 df_features[name] = []
#             df_features[name].append(jobs[key])
#
#     return pd.DataFrame(df_features)


# def extract_single_series_features_batch_inverse(ts_list: list, sensors_name: list, num_batch: int):
#     # Define batch
#     curr_dir = os.path.dirname(os.path.abspath(__file__))
#     curr_dir = os.path.join(curr_dir, 'data/')
#     ts_list = [arr for arr in ts_list]
#     batch_size = get_balanced_job(num_batch, len(ts_list))
#     curr_dir = os.path.dirname(os.path.abspath(__file__))
#     curr_dir = os.path.join(curr_dir, 'data/')
#     list_arguments = []
#     record = []
#     tmp_sensors_name = []
#     for i in range(num_batch):
#         a = i * batch_size[i]
#         b = (i + 1) * batch_size[i]
#         ts_batch = ts_list[a:b]
#
#         # Concatenate record and rename sensor names
#         record = np.hstack(ts_batch)
#         tmp_sensors_name = ["{}aaa{}".format(i, name) for i in range(len(ts_batch)) for name in sensors_name]
#
#         list_arguments.append((record, tmp_sensors_name))
#         # Extract univariate features for each signal
#
#     pool = mp.Pool(num_batch)
#     features_extracted = pool.starmap(extract_univariate_features, list_arguments)
#     pool.close()
#     pool.join()
#     df_features = fix_feats_extracted(features_extracted)
#     return df_features


# def extract_pair_series_features_old(records: np.array, features_name: list):
#     # Extract feature based on each time series pair
#     features_pair = {}
#     index = 0
#     for record in records:
#         # Extract each possible pair combination
#         indexes = np.arange(len(features_name))
#         combs = list(itertools.combinations(indexes, r=2))
#         for i, j in combs:
#             fi = features_name[i]
#             fj = features_name[j]
#
#             # Extract pair features for each pair in the multivariate time series
#             feature = extract_pair_features(record[:, i], record[:, j])
#
#             # Rename key by inserting the feature name
#             feature = {'pair__{}__{}__{}'.format(k, fi, fj): val for k, val in feature.items()}
#
#             features_pair[index] = feature
#             index += 1
#
#     return features_pair


# def feature_extractor_old(ts_list: list, sensors_name: list, batch_size: int):
#     """ Extract all features for each multivariate time series """
#     ts_features_list = []
#     # Extract features based on pair feature function
#
#     max_pool = mp.cpu_count()
#     balance_job = get_balanced_job(number_pool=max_pool, number_job=len(ts_list))
#     ts_list_divided = []
#     index = 0
#     for size in balance_job:
#         ts_list_divided.append(ts_list[index:index + size])
#         index += size
#
#     list_arguments = []
#     for job in range(len(ts_list_divided)):
#         list_arguments.append((ts_list_divided[job], sensors_name))
#     pool = mp.Pool(max_pool)
#     extraction_feats = pool.starmap(extract_pair_series_features, list_arguments)
#     pool.close()
#     pool.join()
#     df_pair_features = organize_feats(extraction_feats)
#     # Create dataframe for pair features
#     df_pair_features = pd.DataFrame(df_pair_features)
#
#     # Extract features based on functions for univariate signals in batch mode
#     df_single_features = extract_single_series_features_batch_inverse(ts_list, sensors_name, max_pool)
#
#     # Create time series feature dataFrame and return
#     df_features = pd.concat([df_single_features, df_pair_features], axis=1)
#
#     return df_features
