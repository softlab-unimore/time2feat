from typing import List
import pandas as pd
import numpy as np
import functools


# ToDo GF: insert other approaches

def average(ranks: List[pd.Series]) -> pd.Series:
    """
    Averages the rank dataframes.

    This function takes a list of Pandas Series objects representing ranks,
    concatenates them into a dataframe, replaces NaN values with negative infinity,
    re-applies the ranking, and then computes the mean rank for each feature.

    Args:
        ranks: List of pandas Series, where each series represents feature ranks.

    Returns:
        A pandas Series representing the average rank across the provided rank Series.
    """
    df_ranks = pd.concat(ranks, axis=1)  # Combine all rank Series into a DataFrame.
    df_ranks = df_ranks.fillna(float('-inf'))  # Replace NaNs with negative infinity to exclude them from ranking.
    df_ranks = df_ranks.rank(axis=0)  # Reapply ranks across the columns (rankers).

    return df_ranks.mean(axis=1)  # Return the mean rank for each feature.


def reciprocal_rank_fusion(ranks: List[pd.Series]) -> pd.Series:
    """
    Averages the rank dataframes.

    This function takes a list of Pandas Series objects representing ranks,
    concatenates them into a dataframe, replaces NaN values with negative infinity,
    re-applies the (inverse) ranking, and then computes the reciprocal rank fusion for each feature.

    Based on: Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods. In Proceedings of SIGIR. 758–759. https://doi.org/10.1145/1571941.1572114

    Args:
        ranks: List of pandas Series, where each series represents feature ranks.

    Returns:
        A pandas Series representing the average rank across the provided rank Series.


    """
    df_ranks = pd.concat(ranks, axis=1)  # Combine all rank Series into a DataFrame.
    df_ranks = df_ranks.fillna(float('-inf'))  # Replace NaNs with negative infinity to exclude them from ranking.
    df_ranks = df_ranks.rank(axis=0, ascending=False, method="max")  # Reapply ranks across the columns (rankers).
    df_ranks = 1 / (len(df_ranks.index) + df_ranks)
    return df_ranks.sum(axis=1)  # Return the sum of the reciprocal ranks for each feature.


def condorcet_fuse(ranks: List[pd.Series]) -> pd.Series:
    """
    Averages the rank dataframes.

    This function takes a list of Pandas Series objects representing ranks,
    concatenates them into a dataframe, replaces NaN values with negative infinity,
    and computes the condorcet-fuse ranking for each featuer.

    Based on: Mark Montague and Javed A. Aslam. 2002. Condorcet fusion for improved retrieval.
    In Proceedings of CIKM. 538–548. https://doi.org/10.1145/584792.584881

    Args:
        ranks: List of pandas Series, where each series represents feature ranks.

    Returns:
        A pandas Series representing the condorcet fuse across the provided rank Series.


    """
    df_ranks = pd.concat(ranks, axis=1)  # Combine all rank Series into a DataFrame.
    df_ranks = df_ranks.fillna(float('-inf'))  # Replace NaNs with negative infinity to exclude them from ranking.
    df_ranks = df_ranks.rank(axis=0)  # Reapply ranks across the columns (rankers).

    def _condorcet_comparison(f1: str, f2: str) -> int:
        # check if f1 has been ranked above f2 more than half of the times. In that case, put f1 above f2.
        better_f1 = np.sum(df_ranks.loc[f1] > df_ranks.loc[f2])
        better_f2 = np.sum(df_ranks.loc[f1] < df_ranks.loc[f2])
        return better_f1 - better_f2

    # sort the features, based on the condorcet comparison
    sorted_features = sorted(df_ranks.index.to_list(), key=functools.cmp_to_key(_condorcet_comparison))

    # reconstruct the pandas series based on the sorted features
    df_ranks = pd.Series(np.arange(len(sorted_features)), index=sorted_features)

    return df_ranks
