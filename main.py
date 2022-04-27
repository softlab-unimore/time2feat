import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from t2f.dataset import read_train_test, update_result
from t2f.window import window_selection, prepare_data
from t2f.extractor import feature_extractor_multi
from t2f.auto import simple_grid_search
from t2f.importance import features_scoring_selection, features_simple_selection
from t2f.preprocessing import apply_transformation
from t2f.clustering import ClusterWrapper, cluster_metrics

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE


# mpl.style.use('seaborn')

def time_difference(t: datetime):
    t_new = datetime.now()
    print('Time: {}'.format(t_new - t))
    return t_new


def save_running_example(dataset, ts_list, df_features, top_features, y_pred, y_true, output_dir, num_example=1):
    indexes = np.arange(len(ts_list))
    y_true = np.array(y_true)
    print(df_features.shape)
    print(len(top_features))
    flatui = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c", "#9b59b6", "#34495e"]
    types = [x.split('__', 2)[0] for x in df_features.columns]
    sensors = [x.split('__', 2)[2] if types[i] == 'pair' else x.split('__', 2)[1] for i, x in
               enumerate(df_features.columns)]
    features = [x.split('__', 2)[1] if types[i] == 'pair' else x.split('__', 2)[2] for i, x in
                enumerate(df_features.columns)]

    positions = []
    labels = []

    for label in sorted(np.unique(y_true)):
        cond = y_true == label

        fig, axes = plt.subplots(1, num_example, figsize=(5 * num_example, 5))
        for i in range(num_example):
            pos = indexes[cond][i]
            positions.append(pos)
            labels.append(label)
            df = ts_list[pos].copy()
            if len(df.columns) == 6:
                df.columns = ['AccX', 'AccY', 'AccZ', 'GirX', 'GirY', 'GirZ']

            if len(df.columns) == 3:
                df.columns = ['AccX', 'AccY', 'AccZ']

            if num_example == 1:
                df.plot(ax=axes, alpha=0.8, color=flatui)
                axes.legend(loc='upper right')
                # axes.set_facecolor('white')

            else:
                df.plot(ax=axes[i], alpha=0.8, color=flatui)
                axes[i].legend(loc='upper right')

        filename = os.path.join(output_dir, 'example_{}_plot_{}.pdf'.format(dataset, label + 1))
        plt.savefig(filename, trasparent=True, format='pdf', dpi=600)
        plt.show()
        plt.close()

    df_vis = pd.DataFrame(df_features.iloc[positions].values, columns=[types, sensors, features])

    filename = os.path.join(output_dir, 'example_{}_feat.xlsx'.format(dataset))
    df_vis.to_excel(filename)

    df_vis = df_vis.iloc[:, [x in top_features for x in df_features.columns]]
    filename = os.path.join(output_dir, 'example_{}_emb.xlsx'.format(dataset))
    df_vis.to_excel(filename)

    fig, axes = plt.subplots(1, num_example, figsize=(5, 5))
    # tsne = TSNE(learning_rate='auto', init='pca', n_iter=10000)
    tsne = TSNE()
    arr = tsne.fit_transform(df_features[top_features])
    # colors = [flatui[y_true[i]] for i in range(len(arr))]

    names = ['Badminton', 'Running', 'Standing', 'Walking']
    names = ['Epilepsy', 'Running', 'Sawing', 'Walking']

    # legends = [names[y_true[i]] for i in range(len(arr))]
    for label in sorted(np.unique(y_true)):
        cond = y_true == label
        axes.scatter(arr[cond, 0], arr[cond, 1], c=flatui[label], alpha=0.8, label=names[label])

    axes.legend()

    filename = os.path.join(output_dir, 'example_{}_tsne1.pdf'.format(dataset))
    plt.savefig(filename, trasparent=True, format='pdf', dpi=600)
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, num_example, figsize=(5, 5))
    for label in sorted(np.unique(y_true)):
        cond = y_true == label
        axes.scatter(arr[cond, 0], arr[cond, 1], c=flatui[label], alpha=0.8)

    filename = os.path.join(output_dir, 'example_{}_tsne2.pdf'.format(dataset))
    plt.savefig(filename, trasparent=True, format='pdf', dpi=600)
    plt.show()
    plt.close()

    # fig, axes = plt.subplots(1, num_example, figsize=(5, 5))
    # # tsne = TSNE(learning_rate='auto', init='pca', n_iter=10000)
    # tsne = TSNE()
    # df = df_features.copy()
    # df = df.dropna(axis='columns')
    # cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
    # df = df.drop(df.columns[cond], axis=1)
    #
    # arr = tsne.fit_transform(df)
    # # colors = [flatui[y_true[i]] for i in range(len(arr))]
    # names = ['Badminton', 'Running', 'Standing', 'Walking']
    # # legends = [names[y_true[i]] for i in range(len(arr))]
    # for label in sorted(np.unique(y_true)):
    #     cond = y_true == label
    #     axes.scatter(arr[cond, 0], arr[cond, 1], c=flatui[label], alpha=0.8, label=names[label])
    #
    # axes.legend()
    #
    # filename = os.path.join(output_dir, 'example_{}_tsne_global.png'.format(dataset))
    # plt.savefig(filename, trasparent=True, format='png', dpi=600)
    # plt.show()
    # plt.close()

    print('')


def pipeline(params: dict):
    # Define pipeline params
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    is_ucr = params['is_ucr']

    window_mode = params['window_mode']
    with_window_selection = params['with_window_selection']
    window = params['window']

    batch_size = params['batch_size']

    top_k = params['top_k']
    score_mode = params['score_mode']
    transform_type = params['transform_type']
    model_type = params['model_type']

    auto = params.get('auto', False)
    strategy = params.get('strategy', 'sk_base')
    k_best = params.get('k_best', False)
    pre_transform = params.get('pre_transform', False)
    complete = params.get('complete', False)
    partial = params.get('partial', -1)

    monitor = params.get('monitor', False)

    monitor_dict = {
        'data_dir': os.path.basename(data_dir),
        'strategy': strategy,
        'partial': partial,
    }

    # Start pipeline
    time_start = datetime.now()
    t0 = time_start

    if not os.path.isdir(data_dir) or not os.path.isdir(output_dir):
        print('Dataset or output folder do not exist')

    print('Read dataset: ', data_dir)
    ts_train_list, y_train, ts_test_list, y_test = read_train_test(path=data_dir, is_ucr=is_ucr)

    sensors_name = ts_train_list[0].columns.to_list()
    sensors_name = [str(x).replace(' ', '') for x in sensors_name]

    if window_mode:
        # Sliding window step
        if with_window_selection:
            # Window selection step
            print('Window selection')
            window = window_selection(ts_train_list, y_train)
            print('Found window: ', window)

        # Prepare list of time series window
        print('Sliding window')
        x_train, y_train = prepare_data(ts_list=ts_train_list, labels=y_train, kernel=window)
        x_test, y_test = prepare_data(ts_list=ts_test_list, labels=y_test, kernel=window)

        t1 = time_difference(t0)
        monitor_dict['Read Dataset'] = t1 - t0

    else:
        # Each series is considered entirely
        x_train = [df.values for df in ts_train_list]
        x_test = [df.values for df in ts_test_list]

        t1 = time_difference(t0)
        monitor_dict['Read Dataset'] = t1 - t0

    print('{} raw train records with shape {}'.format(len(x_train), x_train[0].shape))
    print('{} raw test records with shape {}'.format(len(x_test), x_test[0].shape))

    feat_train_name = "./mts/data/feat_{}_train.pkl".format(os.path.basename(data_dir))
    feat_test_name = "./mts/data/feat_{}_test.pkl".format(os.path.basename(data_dir))

    if os.path.isfile(feat_train_name) and os.path.isfile(feat_test_name) and not monitor:
        print('Read train extracted features')
        df_train_features = pickle.load(open(feat_train_name, "rb"))
        df_test_features = pickle.load(open(feat_test_name, "rb"))

        t2 = time_difference(t1)
        monitor_dict['Feature Extraction'] = t2 - t1

    else:
        # Feature extraction step
        print('Feature extraction step')

        # df_train_features = feature_extractor_multi(x_train, sensors_name, batch_size=batch_size, p=4)
        #
        # df_test_features = feature_extractor_multi(x_test, sensors_name, batch_size=batch_size, p=4)

        train_size = len(x_train)
        x_all = x_train + x_test
        df_features = feature_extractor_multi(x_all, sensors_name, batch_size=batch_size, p=params.get('p', 2))
        t2 = time_difference(t1)
        monitor_dict['Feature Extraction'] = t2 - t1

        df_train_features = df_features[:train_size]
        df_test_features = df_features[train_size:]

        with open(feat_train_name, "wb") as f:
            pickle.dump(df_train_features, f)

        with open(feat_test_name, "wb") as f:
            pickle.dump(df_test_features, f)

    if complete:
        # Clustering all available data
        print('Concat training and test')
        df_concat = pd.concat([df_train_features, df_test_features], axis=0, ignore_index=True)
        y_concat = list(y_train) + list(y_test)

        if partial > 0:
            # Check partial size respect to dataset size
            if partial > 1 and partial > len(df_train_features):
                partial = len(df_train_features)

            df_train_features, df_test_features, y_train, y_test = train_test_split(df_concat, y_concat,
                                                                                    train_size=partial,
                                                                                    stratify=y_concat)
            df_train_features.reset_index(inplace=True, drop=True)

            df_concat = pd.concat([df_train_features, df_test_features], axis=0, ignore_index=True)
            y_concat = list(y_train) + list(y_test)

        df_test_features = df_concat
        y_test = y_concat

    if strategy == 'none':
        print('No label strategy')
        df_train_features = df_test_features
        y_train = y_test

    if pre_transform and transform_type:
        print('Data normalization step')
        df_train_features.iloc[:, :], df_test_features.iloc[:, :] = apply_transformation(df_train_features,
                                                                                         df_test_features,
                                                                                         transform_type)

    if auto:
        print('Auto select best top k value')
        new_params = simple_grid_search(df_train_features, y_train, df_test_features, params)
        top_k = new_params['top_k']
        score_mode = new_params['score_mode']
        transform_type = new_params['transform_type']
        print('Found params: ', new_params)

    t3 = time_difference(t2)
    monitor_dict['Params Selection'] = t3 - t2

    # Feature scoring step
    print('Feature scoring step')
    # df_train_features = df_train_features[:700]
    # y_train = y_train[:700]
    if not k_best:
        top_features = features_scoring_selection(df_train_features, y_train, mode=score_mode, top_k=top_k,
                                                  strategy=strategy)
        t4 = time_difference(t3)
        monitor_dict['Feature Selection'] = t4 - t3
        monitor_dict['Num Feature'] = len(top_features)

    else:
        top_features = features_simple_selection(df_train_features, y_train, top_k=top_k)
        t4 = time_difference(t3)
        monitor_dict['Feature Selection'] = t4 - t3

    print('Found {} features'.format(len(top_features)))
    print("Features Selected: ", top_features)

    # print('Partial feature extraction test data')
    # feats_choose = create_fc_parameter(top_features)
    # df_test_features = feature_extractor_multi(x_test, sensors_name, feat_select=feats_choose, batch_size=1000, p=4)

    # Feature selection step
    print('Feature selection step')
    x_train = df_train_features[top_features].copy().values
    x_test = df_test_features[top_features].copy().values

    print('{} transformed train'.format(x_train.shape))
    print('{} transformed test'.format(x_test.shape))

    # Transformation step
    if transform_type and not pre_transform:
        print('Data normalization step')
        x_train, x_test = apply_transformation(x_train, x_test, transform_type)

    t5 = time_difference(t4)
    monitor_dict['Data Normalization'] = t5 - t4

    # Clustering step
    print('Clustering step')
    num_labels = len(set(y_train))
    model = ClusterWrapper(model_type=model_type, num_cluster=num_labels)
    y_pred = model.fit_predict(x_test)

    t6 = time_difference(t5)
    monitor_dict['Clustering'] = t6 - t5

    monitor_dict['MTS'] = t6 - time_start

    # Compute results
    res_metrics = cluster_metrics(y_test, y_pred)
    print('Result')
    print(res_metrics)

    # Save results
    res_metrics.update(params)
    df_res = pd.DataFrame([res_metrics])
    filename = os.path.join(output_dir, 'result_{}.csv'.format(os.path.basename(data_dir)))
    update_result(df_res, filename, overwrite=False)

    monitor_filename = os.path.join(output_dir, 'time_mts.csv')
    update_result(pd.DataFrame([monitor_dict]), monitor_filename, overwrite=False)

    if params.get('example', False):
        dataset = "{}_{}".format(os.path.basename(data_dir), "unsupervised" if strategy == 'none' else 'semi')
        save_running_example(
            dataset=dataset,
            ts_list=ts_train_list + ts_test_list,
            df_features=df_test_features,
            top_features=top_features,
            y_pred=y_pred,
            y_true=y_test,
            output_dir=output_dir
        )

    return res_metrics


if __name__ == '__main__':
    my_params = {
        'monitor': False,
        'example': True,
        'data_dir': r'C:\Users\delbu\Projects\Dataset\Multivariate_TS\Epilepsy',  # BasicMotions
        'output_dir': 'example/',
        'is_ucr': True,

        'window_mode': False,
        'with_window_selection': False,
        'window': 300,

        'batch_size': 100,
        'p': -1,

        'auto': True,
        'top_k': 100,
        'strategy': 'none',  # 'tsfresh, 'multi, 'sk_base', 'sk_pvalue', 'none'
        'score_mode': 'simple',  # 'simple', 'domain'
        'transform_type': None,  # None, 'std', 'minmax', 'robust'
        'model_type': 'KMeans',  # 'HDBSCAN', 'AgglomerativeClustering', 'KMeans', 'SpectralClustering'

        'k_best': False,
        'pre_transform': False,

        'complete': True,
        'partial': -1,

    }
    res = pipeline(my_params)

# datasets = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket',
#             'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'HandMovementDirection', 'Handwriting',
#             'Libras', 'LSST', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
#             'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary'] + exceptions_size
#
# base_dir = '/export/static/pub/softlab/ucr1/Multivariate_TS/'
