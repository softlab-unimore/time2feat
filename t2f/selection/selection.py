from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .PFA import pfa_scoring
from .search import search
from ..ranking.wrapper import Ranker


def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns which contain nan, +inf, and -inf values
    df = df.dropna(axis=1, how='any')
    cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
    df = df.drop(df.columns[cond], axis=1)

    # Apply a simple variance threshold
    selector = VarianceThreshold()
    selector.fit(df)
    # Get only no-constant features
    top_features = selector.get_feature_names_out()
    df = df[top_features]
    return df


def supervised_selection(
        df: pd.DataFrame,
        labels: dict,
        ranking_type: List[str],
        model_type: str,
        transform_type: Optional[str] = None,
        ensemble_type: Optional[str] = None,
        pfa_variance: Optional[float] = 0.9,
        search_type: Optional[Literal['fixed', 'linear']] = None,
        y_true: list = None,
) -> Tuple[List[str], str, pd.DataFrame]:
    # Extract train and test records, with label associated with train data
    train_idx = list(labels.keys())
    test_idx = [i for i in range(len(df)) if i not in train_idx]
    y_train = list(labels.values())

    df_train = df.iloc[train_idx, :].reset_index(drop=True)
    df_test = df.iloc[test_idx, :].reset_index(drop=True)

    # Create the complete dataframe
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    ranker = Ranker(ranking_type=ranking_type, ensemble_type=ensemble_type, pfa_variance=pfa_variance)
    ranker.ranking(df=df_train, y=y_train)

    top_k, with_separate_domains, transform_type, pfa_variance, df_debug = search(
        ranker=ranker,
        df_train=df_train,
        y_train=y_train,
        df_all=df_all,
        model_type=model_type,
        transform_type=transform_type,
        search_type=search_type,
        df_true=df,
        y_true=y_true
    )
    ranker.pfa_variance = pfa_variance  # Set the PFA variance to the best value found
    top_features = ranker.select(df=df_train, top_k=top_k, with_separate_domains=with_separate_domains)

    print(
        f'Top K: {top_k}, Separate Domains: {with_separate_domains}, Transform Type: {transform_type} pfa {pfa_variance}')
    print(f'Top features: {top_features}')

    return top_features, transform_type, df_debug


def unsupervised_selection(df: pd.DataFrame, variance: Optional[float] = 0.9) -> List[str]:
    # Apply PFA to select the feature which by default keep 90% of the information
    if variance:
        top_features, _ = pfa_scoring(df, variance)
    else:
        top_features = df.columns.tolist()
    return top_features


def feature_selection(
        df: pd.DataFrame,
        labels: dict = None,
        ranking_type: List[str] = None,  # ['anova']
        pfa_variance: Optional[float] = 0.9,
        ensemble_type: str = None,
        search_type: Optional[Literal['fixed', 'linear']] = 'fixed',
        context: dict = None,
        y_true: list = None,
) -> Tuple[List[str], str, pd.DataFrame]:
    if 'transform_type' not in context:
        raise ValueError('The context must contain a "transform_type" key.')
    if labels and 'model_type' not in context:
        s = 'When labels are provided, the context must contain both "model_type" keys.'
        raise ValueError(s)
    if labels and not ranking_type:
        raise ValueError('You must select at least one feature ranking method when labels are provided.')
    if labels and ranking_type and len(ranking_type) > 1 and not ensemble_type:
        raise ValueError('When multiple ranking methods are specified, you must also specify an ensemble method.')

    transform_type = context['transform_type']
    df = cleaning(df)

    if labels:
        top_features, transform_type, df_debug = supervised_selection(
            df=df,
            labels=labels,
            ranking_type=ranking_type,
            pfa_variance=pfa_variance,
            model_type=context['model_type'],
            transform_type=transform_type,
            ensemble_type=ensemble_type,
            search_type=search_type,
            y_true=y_true
        )
    else:
        df_debug = pd.DataFrame()
        top_features = unsupervised_selection(df, variance=pfa_variance)
    return top_features, transform_type, df_debug
