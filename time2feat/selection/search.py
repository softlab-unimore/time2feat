import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from .PFA import pfa_scoring
from ..ranking.wrapper import Ranker
from ..model.clustering import ClusterWrapper, cluster_metrics


def simple_grid_search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
):

    grid_params = {
        'top_k': [10, 25, 50, 100, 200, 300] * 2
    }

    results = []
    time.sleep(0.1)
    for new_params in tqdm(ParameterGrid(grid_params)):
        top_features = ranker.ranking(df=df_train, y=y_train, top_k=new_params['top_k'])

        num_labels = len(set(y_train))
        model = ClusterWrapper(model_type=model_type, transform_type=transform_type, n_clusters=num_labels)
        y_pred = model.fit_predict(df_all[top_features])

        # Compute results
        res_metrics = cluster_metrics(y_train, y_pred[:len(y_train)])
        res_metrics.update(new_params)
        results.append(res_metrics)

    df_res = pd.DataFrame(results)
    df_res = df_res.fillna('None')
    df_res = df_res.groupby(['top_k'])['nmi'].mean().reset_index()
    df_res = df_res.replace(['None'], [None])
    df_res.sort_values('nmi', ascending=False, inplace=True)
    # print(df_res)
    # new_params = {
    #     'top_k': df_res['top_k'].iloc[0],
    #     'score_mode': df_res['score_mode'].iloc[0],
    #     'transform_type': df_res['transform_type'].iloc[0],
    # }
    return df_res['top_k'].iloc[0]
