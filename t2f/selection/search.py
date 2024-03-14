from typing import Optional, Literal, Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from ..ranking.wrapper import Ranker
from ..model.clustering import ClusterWrapper, cluster_metrics


def generate_sequence(N):
    sequence = []

    # From 1 to 10 with step 1
    sequence.extend(range(1, 11))

    # From 10 to 50 with step 2
    sequence.extend(range(12, 51, 2))

    # From 50 to 200 with step 10
    sequence.extend(range(60, 201, 10))

    # From 200 to N with step of 1% of N
    current = 210 if 200 + 0.01 * N > 200 else 200 + 0.01 * N
    step = max(1, 0.01 * N)  # Ensure at least 1 step if N is less than 100
    while current <= N:
        sequence.append(int(current))
        current += step

    return sequence


def debug_step(params, model, df_all, y_train, df_true, y_test) -> dict:
    step = {**params}
    # Compute the clustering metrics for the train and test set
    y_pred = model.fit_predict(df_all)
    res = cluster_metrics(y_train, y_pred[:len(y_train)])
    res = {f'train_{k}': v for k, v in res.items()}
    step.update(res)
    y_pred = model.fit_predict(df_true)
    res = cluster_metrics(y_test, y_pred)
    res = {f'test_{k}': v for k, v in res.items()}
    step.update(res)
    return step


def simple_grid_search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        df_true: pd.DataFrame = None,
        y_true: list = None,
) -> Tuple[int, bool, str, float, pd.DataFrame]:
    """Performs a simple grid search over a set of parameters to find the optimal number of top features.

    This function takes a ranking object which is used to rank and select features from the training data. It then
    iterates over a predefined grid of parameters, each time fitting a clustering model on all data with the selected
    top features, and computing clustering metrics. The function returns the best parameters based on the highest mean
    NMI score.


    Args:
        ranker: An instance of a Ranker class with ranking and select methods.
        df_train: A DataFrame containing the training data features.
        y_train: A list containing the training data labels.
        df_all: A DataFrame containing all features from both the training and test sets.
        model_type: A string indicating the type of clustering model to use.
        transform_type: An optional string indicating the type of transformation to apply to the data.

    Returns:
        An integer representing the optimal number of top features that resulted in the highest mean NMI score.
        A boolean indicating whether to use separate domains for feature selection.
        A string indicating the optimal transformation type to use.

    """
    # # Rank features using the ranking object
    # ranker.ranking(df=df_train, y=y_train)

    # Define grid parameters
    grid_params = {
        'top_k': generate_sequence(df_train.shape[1]),  # [10, 25, 50, 100, 200, 300] * 4,
        'with_separate_domains': [True, False],
        'transform_type': ['minmax', 'standard', None],
        'pfa': [0.9, None]
    }

    results = []
    df_debug = []
    time.sleep(0.1)  # Small sleep for tqdm robustness
    for new_params in tqdm(ParameterGrid(grid_params)):
        # Select top K features with the current parameters
        ranker.pfa_variance = new_params['pfa']  # Set PFA variance if applicable
        top_features = ranker.select(df=df_train, top_k=new_params['top_k'],
                                     with_separate_domains=new_params['with_separate_domains'])

        # Determine the number of unique labels
        num_labels = len(set(y_train))
        # Initialize the cluster wrapper with the model and transformation types
        model = ClusterWrapper(model_type=model_type, transform_type=new_params['transform_type'],
                               n_clusters=num_labels)
        # Fit and predict the model
        y_pred = model.fit_predict(df_all[top_features])

        # Compute results and metrics
        res_metrics = cluster_metrics(y_train, y_pred[:len(y_train)])
        res_metrics.update(new_params)
        results.append(res_metrics)

        df_debug.append(debug_step(new_params, model, df_train[top_features], y_train, df_true[top_features], y_true))

    # Configuration params to return
    cols = ['top_k', 'with_separate_domains', 'transform_type', 'pfa']  # values will be returned in this order
    # Metrics to use for comparison
    metrics = ['ami', 'nmi', 'rand']  # the order of the metrics is important

    # Convert results to DataFrame and calculate mean NMI score
    df_res = pd.DataFrame(results).fillna('None')  # Because transform_type could be None
    df_res = df_res.groupby(cols)[metrics].mean().reset_index()
    df_res = df_res.replace(['None'], [None])  # Replace 'None' with None
    df_res = df_res.sort_values(metrics, ascending=False)

    # Return the best configuration value with the highest average ami score
    top_k, with_separate_domains, transform_type, pfa = df_res[cols].iloc[0].to_list()
    return top_k, with_separate_domains, transform_type, pfa, pd.DataFrame(df_debug)


def search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        search_type: Optional[Literal['fixed', 'linear']] = None,
        df_true: pd.DataFrame = None,
        y_true: list = None,
) -> Tuple[int, bool, str, float, pd.DataFrame]:
    """
    Performs a search for the optimal number of top features, with optional separate domains and transformation type.
    """
    if search_type == 'fixed':
        return simple_grid_search(
            ranker=ranker,
            df_train=df_train,
            y_train=y_train,
            df_all=df_all,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true
        )
    elif search_type == 'linear':
        assert False, 'Linear search is not supported'
    else:
        # No search is performed so return the default values
        # 0.9 is the default value for pfa_variance
        # False is the default value for with_separate_domains
        return len(df_train.columns), False, transform_type, 0.9, pd.DataFrame()
