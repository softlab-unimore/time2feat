import os
import pandas as pd

from t2f.plot.cdd import draw_cd_diagram


def check_null_results():
    data_dir = './results'
    for results_dirname in os.listdir(data_dir):
        results_dir = os.path.join(data_dir, results_dirname)

        print(f'\n{results_dirname}')
        for results_name in os.listdir(results_dir):
            df = pd.read_csv(os.path.join(results_dir, results_name), index_col=0)

            if len(df) != 66:
                print(f'{results_name}:\t {len(df)} rows instead of 66')

            nan_rows = df.isna().any(axis=1)
            if nan_rows.any():
                print(f'{results_name}:\t nan rows found in {df.index[nan_rows].to_list()}')


def check_results_difference():
    results_dir1 = './results/4s'
    th = 0.1
    col = 'ami'
    rows = ['anova']

    view = {}
    for results_dir2 in ['./results/42', './results/111', './results/123', './results/456', './results/789']:
        for results_name in os.listdir(results_dir1):
            if results_name not in os.listdir(results_dir2):
                print('Skipping (no file)', results_name)
                continue

            df1 = pd.read_csv(os.path.join(results_dir1, results_name), index_col=0)
            df2 = pd.read_csv(os.path.join(results_dir2, results_name), index_col=0)

            if len(df1) != 66 or len(df2) != 66:
                print('Skipping (no complete)', results_name)
                continue

            if not rows:
                rows = df1.index.to_list()

            df_diff = df2.loc[rows] - df1.loc[rows]
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

        if len(df) != 66:
            print('Skipping (no complete)', results_name)

        all_results.append(df[['classifier_name', 'dataset_name', 'accuracy']])

    all_results = pd.concat(all_results, ignore_index=True, axis=0)
    return all_results


def plot_critical_difference_diagrams():
    output_dir = './figures'
    metric = 'ami'

    for results_dir in ['./results/4s', './results/42', './results/111', './results/123', './results/456',
                        './results/789']:
        for train_size in ['s20', 's30', 's50']:
            all_results = read_results(results_dir, train_size, metric)

            # Option 1: drop dataset with at least one nan
            datasets_with_nan = all_results.loc[all_results.isnull().any(axis=1)]['dataset_name'].unique()
            all_results = all_results[~all_results['dataset_name'].isin(datasets_with_nan)]

            # Option 2: drop nan rows
            # all_results = all_results.dropna(axis=0, how='any')

            all_results.loc[all_results['accuracy'] <= 0, 'accuracy'] = 0.001

            out_filename = os.path.join(output_dir, f'cdd_{train_size}_{metric}_{os.path.basename(results_dir)}.png')
            draw_cd_diagram(all_results, out_filename, alpha=0.05, title=metric.upper(), labels=True)
    print('Here!')


if __name__ == '__main__':
    # check_null_results()
    # check_results_difference()
    # plot_critical_difference_diagrams()

    print('Hello World!')
