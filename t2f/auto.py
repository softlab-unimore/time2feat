import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from .importance import features_scoring_selection, features_simple_selection
from .preprocessing import apply_transformation
from .clustering import ClusterWrapper, cluster_metrics


def simple_grid_search(df_base: pd.DataFrame, y_base: list, df_complete: pd.DataFrame, params: dict):
    cond = 'model_type' in params and 'transform_type' in params and 'score_mode' in params
    assert cond, 'Please provide all mandatory params (model, transform and score mode)'

    cond = (pd.isnull(df_base)) & (pd.isnull(df_complete.iloc[:len(df_base)]))
    mask = (df_base == df_complete.iloc[:len(df_base)])
    mask = np.where(cond, True, mask)
    assert np.all(mask), 'df_base must be in the first record of df_complete'

    # Optional params
    k_best = params.get('k_best', False)
    strategy = params.get('strategy', 'sk_base')
    pre_transform = params.get('pre_transform', False)

    grid_params = {
        'top_k': [1],
        'score_mode': [params['score_mode']],
        'transform_type': [params['transform_type']],
    }

    if strategy.startswith('sk'):
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2
        # grid_params['transform_type'] = ['minmax', 'std']
        grid_params['score_mode'] = ['simple', 'domain']
    elif strategy.startswith('none'):
        grid_params['transform_type'] = ['minmax', 'std', None]
        grid_params['score_mode'] = ['simple']

    else:
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2

    results = []
    time.sleep(0.1)
    for new_params in tqdm(ParameterGrid(grid_params)):
        if not k_best:
            max_rows = len(df_base)  # 800
            top_features = features_scoring_selection(df_base.iloc[:max_rows, :], y_base[:max_rows],
                                                      mode=new_params['score_mode'], top_k=new_params['top_k'],
                                                      strategy=strategy)
        else:
            top_features = features_simple_selection(df_base, y_base, top_k=new_params['top_k'])

        # Extract train and test
        x_train = df_base[top_features].values
        x_test = df_complete[top_features].values

        # Transformation step
        if new_params['transform_type'] and not pre_transform:
            _, x_test = apply_transformation(x_train, x_test, new_params['transform_type'])

        # Clustering step
        num_labels = len(set(y_base))
        model = ClusterWrapper(model_type=params['model_type'], n_clusters=num_labels)
        y_pred = model.fit_predict(x_test)

        # Compute results
        res_metrics = cluster_metrics(y_base, y_pred[:len(y_base)])
        res_metrics.update(new_params)
        results.append(res_metrics)

    df_res = pd.DataFrame(results)
    df_res = df_res.fillna('None')
    df_res = df_res.groupby(['top_k', 'score_mode', 'transform_type'])['nmi'].mean().reset_index()
    df_res = df_res.replace(['None'], [None])
    df_res.sort_values('nmi', ascending=False, inplace=True)
    print(df_res)
    new_params = {
        'top_k': df_res['top_k'].iloc[0],
        'score_mode': df_res['score_mode'].iloc[0],
        'transform_type': df_res['transform_type'].iloc[0],
    }
    return new_params
