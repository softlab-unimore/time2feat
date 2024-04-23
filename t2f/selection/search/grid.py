from typing import Tuple
import time

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from t2f.ranking.wrapper import Ranker
from t2f.model.clustering import ClusterWrapper, cluster_metrics

from .utils import debug_step, debug_step_test


def simple_grid_search(
        top_k_values: list,
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        df_true: pd.DataFrame = None,
        y_true: list = None,
        is_time2feat: bool = False,
        df_test: pd.DataFrame = None,
        y_test: list = None,
) -> Tuple[int, bool, str, float, pd.DataFrame]:
    """Performs a simple grid search over a set of parameters to find the optimal number of top features.

    This function takes a ranking object which is used to rank and select features from the training data. It then
    iterates over a predefined grid of parameters, each time fitting a clustering model on all data with the selected
    top features, and computing clustering metrics. The function returns the best parameters based on the highest mean
    NMI score.


    Args:
        top_k_values: A list of integers representing the number of top features to consider.
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

    if not is_time2feat:
        # Define grid parameters
        grid_params = {
            'top_k': top_k_values,
            'with_separate_domains': [True, False],
            'transform_type': ['minmax', 'standard', None],
            'pfa': [0.9, None]
        }
    else:
        # Define grid parameters
        grid_params = {
            'top_k': top_k_values,
            'with_separate_domains': [True, False],
            'transform_type': ['minmax', 'standard'],
            'pfa': [0.9]
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
        if df_test is None or y_test is None:
            res_metrics = cluster_metrics(y_train, y_pred[:len(y_train)])
            res_metrics.update(new_params)
            results.append(res_metrics)

            df_debug.append(
                debug_step(new_params, model, df_all[top_features], y_train, df_true[top_features], y_true)
            )
        else:
            res_metrics = cluster_metrics(y_test, y_pred[len(y_train):len(y_train) + len(y_test)])
            res_metrics.update(new_params)
            results.append(res_metrics)

            df_debug.append(
                debug_step_test(new_params, model, df_all[top_features], y_train, df_true[top_features], y_true, y_test)
            )

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
