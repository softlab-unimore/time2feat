import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from t2f.ranking.wrapper import Ranker
from .grid import simple_grid_search
from .utils import generate_sequence


def cv_search_on_train_metrics(
        k_split: int,
        top_k_values: list,
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        df_true: pd.DataFrame = None,
        y_true: list = None,
):
    kf = KFold(n_splits=k_split, shuffle=True)
    indexes = np.arange(len(df_train))

    df_test_real = df_all.iloc[len(df_train):, :].reset_index(drop=True)

    results = []
    i = 0
    for train_indexes, test_indexes in kf.split(indexes):
        print('Fold: ', i)
        y_train_fold = [y_train[i] for i in train_indexes]
        df_train_fold = df_train.iloc[train_indexes, :].reset_index(drop=True)

        df_test_fold = df_train.iloc[test_indexes, :].reset_index(drop=True)
        df_all_fold = pd.concat([df_train_fold, df_test_fold, df_test_real], axis=0, ignore_index=True)

        ranker_fold = Ranker(
            ranking_type=ranker.ranking_type,
            ensemble_type=ranker.ensemble_type,
            pfa_variance=ranker.pfa_variance
        )
        ranker_fold.ranking(df=df_train_fold, y=y_train_fold)

        top_k_fold, with_separate_domains_fold, transform_type_fold, pfa_fold, df_debug_fold = simple_grid_search(
            top_k_values=top_k_values,
            ranker=ranker_fold,
            df_train=df_train_fold,
            y_train=y_train_fold,
            df_all=df_all_fold,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true
        )

        df_debug_fold['fold'] = i
        results.append(df_debug_fold)
        i += 1

    df_debug_all = pd.concat(results, axis=0, ignore_index=True)

    # Configuration params to return
    cols = ['top_k', 'with_separate_domains', 'transform_type', 'pfa']  # values will be returned in this order
    # Metrics to use for comparison
    metrics = ['train_ami', 'train_nmi', 'train_rand']  # the order of the metrics is important

    # Convert results to DataFrame and calculate mean NMI score
    df_res = pd.DataFrame(df_debug_all).fillna('None')  # Because transform_type could be None
    df_res = df_res.groupby(cols)[metrics].mean().reset_index()
    df_res = df_res.replace(['None'], [None])  # Replace 'None' with None
    df_res = df_res.sort_values(metrics, ascending=False)

    # Return the best configuration value with the highest average ami score
    top_k, with_separate_domains, transform_type, pfa = df_res[cols].iloc[0].to_list()
    return top_k, with_separate_domains, transform_type, pfa, df_debug_all


def cv_search_on_test_metrics():
    pass


def loo_search():
    pass
