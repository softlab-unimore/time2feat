from typing import Optional, Tuple

import pandas as pd

from ...ranking.wrapper import Ranker

from .utils import generate_sequence
from .cv import cv_search_on_train_metrics
from .grid import simple_grid_search


def search(
        ranker: Ranker,
        df_train: pd.DataFrame,
        y_train: list,
        df_all: pd.DataFrame,
        model_type: str,
        transform_type: str = None,
        search_type: Optional[str] = None,
        df_true: pd.DataFrame = None,
        y_true: list = None,
) -> Tuple[int, bool, str, float, pd.DataFrame]:
    """
    Performs a search for the optimal number of top features, with optional separate domains and transformation type.
    """
    if search_type == 'fixed':
        return simple_grid_search(
            top_k_values=[10, 25, 50, 100, 200, 300] * 4,
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
        return simple_grid_search(
            top_k_values=generate_sequence(df_train.shape[1]),
            ranker=ranker,
            df_train=df_train,
            y_train=y_train,
            df_all=df_all,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true
        )

    elif search_type.startswith('cv'):
        k_split = int(search_type[2:])
        return cv_search_on_train_metrics(
            k_split=k_split,
            top_k_values=generate_sequence(df_train.shape[1]),
            ranker=ranker,
            df_train=df_train,
            y_train=y_train,
            df_all=df_all,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true
        )

    elif search_type.startswith('testcv'):
        k_split = int(search_type[6:])
        return cv_search_on_train_metrics(
            k_split=k_split,
            top_k_values=generate_sequence(df_train.shape[1]),
            ranker=ranker,
            df_train=df_train,
            y_train=y_train,
            df_all=df_all,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true,
            with_test=True,
        )

    elif search_type == 'time2feat':
        return simple_grid_search(
            top_k_values=[10, 25, 50, 100, 200, 300] * 2,
            ranker=ranker,
            df_train=df_train,
            y_train=y_train,
            df_all=df_all,
            model_type=model_type,
            transform_type=transform_type,
            df_true=df_true,
            y_true=y_true,
            is_time2feat=True
        )

    elif search_type is None:
        # No search is performed so return the default values
        # 0.9 is the default value for pfa_variance
        # False is the default value for with_separate_domains
        return len(df_train.columns), False, transform_type, 0.9, pd.DataFrame()
    else:
        raise ValueError(f'Invalid search type: {search_type}')
