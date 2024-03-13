import os
import pandas as pd

from t2f.plot.cdd import draw_cd_diagram

NUM_ABLATION = 59


def check_null_results():
    data_dir = './results'
    for results_dirname in os.listdir(data_dir):
        results_dir = os.path.join(data_dir, results_dirname)

        print(f'\n{results_dirname}')
        for results_name in os.listdir(results_dir):
            df = pd.read_csv(os.path.join(results_dir, results_name), index_col=0)

            if len(df) != NUM_ABLATION:
                print(f'{results_name}:\t {len(df)} rows instead of {NUM_ABLATION}')

            nan_rows = df.isna().any(axis=1)
            if nan_rows.any():
                print(f'{results_name}:\t nan rows found in {df.index[nan_rows].to_list()}')


def check_results_difference():
    results_dir = './results'
    results_dir1 = os.path.join(results_dir, '42')  # s4
    th = 0.1
    col = 'ami'
    rows = ['anova']

    view = {}
    for dir2 in os.listdir(results_dir):
        results_dir2 = os.path.join(results_dir, dir2)
        for results_name in os.listdir(results_dir1):
            if results_name not in os.listdir(results_dir2):
                print('Skipping (no file)', results_name)
                continue

            df1 = pd.read_csv(os.path.join(results_dir1, results_name), index_col=0)
            df2 = pd.read_csv(os.path.join(results_dir2, results_name), index_col=0)

            if len(df1) != NUM_ABLATION or len(df2) != NUM_ABLATION:
                print('Skipping (no complete)', results_name)
                continue

            if not rows:
                rows = df1.index.to_list()

            df_diff = df1.loc[rows] - df2.loc[rows]
            if (df_diff[col] > th).any():
                # print(f'\n{results_dir2}/{results_name}:\n{df_diff[df_diff[col] > th]}')

                df_rows = df_diff[df_diff[col] > th]
                df_rows.index.name = 'Model'
                df_rows = df_rows.reset_index()
                df_rows['Seed'] = results_dir2
                df_rows['Dataset'] = results_name

                if results_name not in view:
                    view[results_name] = df_rows
                else:
                    view[results_name] = pd.concat([view[results_name], df_rows], ignore_index=True, axis=0)

    df_view = pd.concat(view.values(), axis=0, ignore_index=True)
    print(df_view)
    print('Here!')


def read_results(results_dir, train_size, metric):
    all_results = []
    for results_name in os.listdir(results_dir):

        if train_size not in results_name:
            continue

        df = pd.read_csv(os.path.join(results_dir, results_name), index_col=0)
        df.index.name = 'classifier_name'
        df = df.reset_index()
        df['accuracy'] = df[metric]
        df['dataset_name'] = results_name.split('_')[0]

        if len(df) != NUM_ABLATION:
            print('Skipping (no complete)', results_name)

        all_results.append(df[['classifier_name', 'dataset_name', 'accuracy']])

    all_results = pd.concat(all_results, ignore_index=True, axis=0)
    return all_results


def mean_results():
    results_dir = './results'
    output_mean_dir = os.path.join(results_dir, 'mean')
    output_std_dir = os.path.join(results_dir, 'std')

    os.makedirs(output_mean_dir, exist_ok=True)
    os.makedirs(output_std_dir, exist_ok=True)

    all_results = {}
    for my_dir in os.listdir(results_dir):
        my_dir = os.path.join(results_dir, my_dir)
        for train_size in ['s20', 's30', 's50']:
            for results_name in os.listdir(my_dir):
                if train_size not in results_name:
                    continue

                df = pd.read_csv(os.path.join(my_dir, results_name), index_col=0)

                if len(df) != NUM_ABLATION:
                    print('Skipping (no complete)', results_name)

                if results_name not in all_results:
                    all_results[results_name] = []

                all_results[results_name].append(df)

    for results_name, dfs in all_results.items():
        df = pd.concat(dfs, axis=0)
        df = df.groupby(level=0).mean()
        df.to_csv(os.path.join(output_mean_dir, results_name), index=True)

        df = pd.concat(dfs, axis=0)
        df = df.groupby(level=0).std()
        df.to_csv(os.path.join(output_std_dir, results_name), index=True)

    print('Here!')


def plot_critical_difference_diagrams():
    results_dir = './results'
    output_dir = './figures'
    metric = 'nmi'

    for my_dir in ['mean']:  # os.listdir(results_dir)
        my_dir = os.path.join(results_dir, my_dir)
        os.makedirs(os.path.join(output_dir, os.path.basename(my_dir)), exist_ok=True)
        for train_size in ['s20', 's30', 's50']:
            all_results = read_results(my_dir, train_size, metric)

            # Option 1: drop dataset with at least one nan
            datasets_with_nan = all_results.loc[all_results.isnull().any(axis=1)]['dataset_name'].unique()
            all_results = all_results[~all_results['dataset_name'].isin(datasets_with_nan)]

            # Option 2: drop nan rows
            # all_results = all_results.dropna(axis=0, how='any')

            all_results.loc[all_results['accuracy'] <= 0, 'accuracy'] = 0.001

            out_filename = os.path.join(output_dir, os.path.basename(my_dir),
                                        f'cdd_{train_size}_{metric}_{os.path.basename(my_dir)}.png')
            draw_cd_diagram(all_results, out_filename, alpha=0.05, title=metric.upper(), labels=True)
    print('Here!')


def compare_w_t2f():
    size = 20
    methods = [
        "anova", "average", 'reciprocal_rank_fusionTop5', 'reciprocal_rank_fusionSimSK'
    ]
    # "averageALL", "averageSimSK", "averageTop3", "averageTop5",
    #     "cfs", "cfs w/o S&PFA", "cife", "cife w/o S&PFA", "cmim", "cmim w/o S&PFA",
    #     "combmnz", "combmnzALL", "combmnzSimSK", "combmnzTop3", "combmnzTop5",
    #     "combsum", "combsumALL", "combsumSimSK", "combsumTop3", "combsumTop5",
    #     "condorcet_fuse", "condorcet_fuseALL", "condorcet_fuseSimSK", "condorcet_fuseTop3", "condorcet_fuseTop5",
    #     "disr", "fisher_score", "gini", "icap", "icap w/o S&PFA",
    #     "inverse_square_rank", "inverse_square_rankALL", "inverse_square_rankSimSK", "inverse_square_rankTop3",
    #     "inverse_square_rankTop5",
    #     "jmi", "laplace_score", "mcfs", "mifs", "mim",
    #     "mrmr", "mrmr w/o S&PFA", "ndfs",
    #     "rank_biased_centroid", "rank_biased_centroidALL", "rank_biased_centroidSimSK", "rank_biased_centroidTop3",
    #     "rank_biased_centroidTop5",
    #     "reciprocal_rank_fusion", "reciprocal_rank_fusionALL", "reciprocal_rank_fusionSimSK",
    #     "reciprocal_rank_fusionTop3", "reciprocal_rank_fusionTop5",
    #     "rfs", "trace_ratio", "trace_ratio100", "udfs"
    # ]
    if size == 20:
        idx = 0
    elif size == 50:
        idx = 1
    else:
        raise ValueError('Invalid size')

    t2f_mean = pd.read_csv('./results/compact/t2f_mean.csv', index_col=0).iloc[:, [idx]]
    t2f_std = pd.read_csv('./results/compact/t2f_std.csv', index_col=0).iloc[:, [idx]]
    # mead_dir = './results/compact/mean'
    mead_dir = './results/compact/mean'
    std_dir = './results/compact/std'

    baselines_mean = []
    baselines_std = []
    for dataset in t2f_mean.index:
        df_mean = pd.read_csv(f'{mead_dir}/{dataset}_s{size}.csv', index_col=0).loc[methods, 'ami']
        df_std = pd.read_csv(f'{std_dir}/{dataset}_s{size}.csv', index_col=0).loc[methods, 'ami']
        df_mean.name = dataset
        df_std.name = dataset

        baselines_mean.append(df_mean)
        baselines_std.append(df_std)

    baselines_mean = pd.concat(baselines_mean, axis=1).T
    cols = [col for col in baselines_mean if col != 'anova']

    win = pd.Series((baselines_mean[cols].values > baselines_mean[['anova']].values).sum(axis=0), index=cols)
    print('Here')

    baselines_std = pd.concat(baselines_std, axis=1).T

    baselines_mean = pd.concat([t2f_mean, baselines_mean], axis=1).round(3)
    baselines_std = pd.concat([t2f_std, baselines_std], axis=1).round(3)
    print('Here!')


if __name__ == '__main__':
    # check_null_results()
    # check_results_difference()
    # mean_results()
    # plot_critical_difference_diagrams()
    # compare_w_t2f()
    print('Hello World!')
