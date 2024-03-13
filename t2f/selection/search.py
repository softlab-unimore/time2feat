from typing import Optional, Literal, Tuple
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from ..ranking.wrapper import Ranker
from ..model.clustering import ClusterWrapper, cluster_metrics


def simple_grid_search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
) -> Tuple[int, bool, str]:
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
        'top_k': [10, 25, 50, 100, 200, 300] * 2,
        'with_separate_domains': [True, False],
        'transform_type': ['minmax', 'standard', None],
    }

    results = []
    time.sleep(0.1)  # Small sleep for tqdm robustness
    for new_params in tqdm(ParameterGrid(grid_params)):
        # Select top K features
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

    # Convert results to DataFrame and calculate mean NMI score
    df_res = pd.DataFrame(results).fillna('None')  # Because transform_type could be None
    df_res = df_res.groupby(['top_k', 'with_separate_domains', 'transform_type'])['nmi'].mean()
    df_res = df_res.replace(['None'], [None])  # Replace 'None' with None
    df_res.sort_values('nmi', ascending=False, inplace=True)

    # Return the top_k value with the highest mean NMI score
    return df_res['top_k'].iloc[0], df_res['with_separate_domains'].iloc[0], df_res['transform_type'].iloc[0]


def search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        search_type: Optional[Literal['fixed', 'linear']] = None,
) -> Tuple[int, bool, str]:
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
        )
    elif search_type == 'linear':
        assert False, 'Linear search is not supported'
    else:
        # No search is performed so return the default values
        return len(df_train.columns), False, transform_type
