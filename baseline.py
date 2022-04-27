import os
import numpy as np
import pandas as pd
from datetime import datetime

from t2f.dataset import read_train_test, update_result
from t2f.window import window_selection, prepare_data
from t2f.preprocessing import apply_transformation
from t2f.clustering import ClusterWrapper, cluster_metrics


def time_difference(t: datetime):
    t_new = datetime.now()
    print('Time: {}'.format(t_new - t))
    return t_new


def baseline(params: dict):
    # Define pipeline params
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    is_ucr = params['is_ucr']

    window_mode = params['window_mode']
    with_window_selection = params['with_window_selection']
    window = params['window']

    model_type = params['model_type']

    monitor_dict = {
        'data_dir': os.path.basename(data_dir),
        'model': model_type,
    }
    # Start pipeline
    time_start = datetime.now()
    t0 = time_start

    if not os.path.isdir(data_dir) or not os.path.isdir(output_dir):
        print('Dataset or output folder do not exist')

    print('Read dataset: ', data_dir)
    ts_train_list, y_train, ts_test_list, y_test = read_train_test(path=data_dir, is_ucr=is_ucr)
    time_start = time_difference(time_start)

    if window_mode:
        # Sliding window step
        if with_window_selection:
            # Window selection step
            print('Window selection')
            window = window_selection(ts_train_list, y_train)
            print('Found window: ', window)
            time_start = time_difference(time_start)

        # Prepare list of time series window
        print('Sliding window')
        x_train, y_train = prepare_data(ts_list=ts_train_list, labels=y_train, kernel=window)
        x_test, y_test = prepare_data(ts_list=ts_test_list, labels=y_test, kernel=window)
        time_start = time_difference(time_start)

    else:
        # Each series is considered entirely
        x_train = [df.values for df in ts_train_list]
        x_test = [df.values for df in ts_test_list]

    print('{} raw train records with shape {}'.format(len(x_train), x_train[0].shape))
    print('{} raw test records with shape {}'.format(len(x_test), x_test[0].shape))

    # Extract train and test
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_test = np.vstack([x_train, x_test])
    y_test = y_train + y_test

    print('{} train'.format(x_train.shape))
    print('{} test'.format(x_test.shape))

    # Clustering step
    print('Clustering step')
    num_labels = len(set(y_train))
    model = ClusterWrapper(model_type=model_type, num_cluster=num_labels)
    y_pred = model.fit_predict(x_test)
    _ = time_difference(time_start)

    # Compute results
    res_metrics = cluster_metrics(y_test, y_pred)
    print('Result')
    print(res_metrics)

    # Save results
    res_metrics.update(params)
    df_res = pd.DataFrame([res_metrics])
    filename = os.path.join(output_dir, 'baseline_{}.csv'.format(os.path.basename(data_dir)))
    update_result(df_res, filename, overwrite=False)

    t_end = time_difference(t0)
    monitor_dict['Baseline'] = t_end - t0
    filename = os.path.join(output_dir, 'time_baseline.csv')
    update_result(pd.DataFrame([monitor_dict]), filename, overwrite=False)

    return res_metrics


if __name__ == '__main__':
    my_params = {
        'data_dir': 'data/Cricket',
        'output_dir': 'output/',
        'is_ucr': True,

        'window_mode': False,
        'with_window_selection': False,
        'window': 300,

        'batch_size': 2000,

        'top_k': 200,
        'score_mode': 'domain',  # 'simple', 'domain'

        'transform_type': None,  # None, 'std', 'minmax', 'robust'
        'model_type': 'KMeans',  # 'HDBSCAN', 'AgglomerativeClustering', 'KMeans', 'SpectralClustering'

    }
    exceptions_length = ['CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits', 'InsectWingbeat']
    datasets = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket',
                'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'HandMovementDirection', 'Handwriting',
                'Libras', 'LSST', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
                'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary']

    # base_dir = '/export/static/pub/softlab/ucr1/Multivariate_TS/'
    base_dir = r'C:\Users\delbu\Projects\Dataset\Multivariate_TS'
    # datasets = ['ArticularyWordRecognition']

    for dataset in datasets:
        data_dir = os.path.join(base_dir, dataset)
        my_params['data_dir'] = data_dir

        for model in ['KMeans', 'AgglomerativeClustering']:
            print(model)
            my_params['model_type'] = model
            res = baseline(my_params)
