from typing import List, Literal, Optional
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .PFA import pfa_scoring
from .search import simple_grid_search
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
        ranking_type: list,
        model_type: str,
        transform_type: str = None,
        ensemble_type: str = None,
) -> List[str]:
    # Extract train and test records, with label associated with train data
    train_idx = list(labels.keys())
    test_idx = [i for i in range(len(df)) if i not in train_idx]
    y_train = list(labels.values())

    df_train = df.iloc[train_idx, :].reset_index(drop=True)
    df_test = df.iloc[test_idx, :].reset_index(drop=True)

    # Create the complete dataframe
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    ranker = Ranker(ranking_type=ranking_type, ensemble_type=ensemble_type)

    top_k = simple_grid_search(
        ranker=ranker,
        df_train=df_train,
        y_train=y_train,
        df_all=df_all,
        model_type=model_type,
        transform_type=transform_type
    )
    top_features = ranker.ranking(df=df_train, y=y_train, top_k=top_k)

    return top_features


def unsupervised_selection(df: pd.DataFrame) -> List[str]:
    # Apply PFA to select the feature which keep 90% of the information
    top_features, _ = pfa_scoring(df, 0.9)
    return top_features


def feature_selection(
        df: pd.DataFrame,
        ranking_type: Optional[List[Literal['anova']]] = None,
        ensemble_type: Optional[Literal['average']] = None,
        labels: dict = None,
        context: dict = None,
) -> List[str]:
    if labels and ('model_type' not in context or 'transform_type' not in context):
        raise ValueError('When labels are provided, the context must contain both "model_type" and "transform_type" keys for supervised selection.')
    if labels and not ranking_type:
        raise ValueError('You must select at least one feature ranking method when labels are provided.')
    if labels and ranking_type and len(ranking_type) > 1 and not ensemble_type:
        raise ValueError('When multiple ranking methods are specified, you must also specify an ensemble method.')

    df = cleaning(df)

    if labels:
        top_features = supervised_selection(
            df=df,
            labels=labels,
            ranking_type=ranking_type,
            model_type=context['model_type'],
            transform_type=context['transform_type'],
            ensemble_type=ensemble_type,
        )
    else:
        top_features = unsupervised_selection(df)
    return top_features
