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


def rank_biased_centroid(ranks: List[pd.Series]) -> pd.Series:
    """
    Averages the rank dataframes.

    This function takes a list of Pandas Series objects representing ranks,
    concatenates them into a dataframe, replaces NaN values with negative infinity,
    and computes the rank biased centroid ranking for each feature.

    Based on: Peter Bailey, Alistair Moffat, Falk Scholer, and Paul Thomas. 2017. Retrieval Consistency in the Presence of Query Variations. In
    Proceedings SIGIR. https://doi.org/10.1145/3077136.3080839

    Args:
        ranks: List of pandas Series, where each series represents feature ranks.

    Returns:
        A pandas Series representing the rank biased centroid across the provided rank Series.


    """
    df_ranks = pd.concat(ranks, axis=1)  # Combine all rank Series into a DataFrame.
    df_ranks = df_ranks.fillna(float('-inf'))  # Replace NaNs with negative infinity to exclude them from ranking.
    df_ranks = df_ranks.rank(axis=0, method="max")  # Reapply ranks across the columns (rankers).

    persistence = 0.98
    np_ranks = np.array(df_ranks)
    invrank = np.max(np_ranks, axis=0)
    invrank = invrank - np_ranks
    decay = (1 - persistence) * persistence ** (invrank)
    np_ranks = np.divide(np_ranks, decay)
    df_ranks = pd.Series(np.mean(np_ranks, axis=1), index=df_ranks.index)

    return df_ranks


def inverse_square_rank(ranks: List[pd.Series]) -> pd.Series:
    """
    Averages the rank dataframes.

    This function takes a list of Pandas Series objects representing ranks,
    concatenates them into a dataframe, replaces NaN values with negative infinity,
    and computes the inverse square rank for each feature.

    Based on: Mourao, A., Martins, F. and Magalhaes, J., 2013. NovaSearch at TREC 2013 Federated Web Search Track: Experiments with rank fusion.
    In TREC. https://trec.nist.gov/pubs/trec22/papers/novasearch-federated.pdf
    Mourão, A., Martins, F. and Magalhaes, J., 2014, June. Inverse square rank fusion for multimodal search. In 2014 12th international workshop on
    content-based multimedia indexing (CBMI) (pp. 1-6). IEEE. 10.1109/CBMI.2014.6849825.

    Args:
        ranks: List of pandas Series, where each series represents feature ranks.

    Returns:
        A pandas Series representing the inverse square rank across the provided rank Series.


    """
    df_ranks = pd.concat(ranks, axis=1)  # Combine all rank Series into a DataFrame.
    df_ranks = df_ranks.fillna(float('-inf'))  # Replace NaNs with negative infinity to exclude them from ranking.
    df_ranks = df_ranks.rank(axis=0, method="max")  # Reapply ranks across the columns (rankers).

    np_ranks = np.array(df_ranks)
    invrank = np.max(np_ranks, axis=0)
    invrank = invrank - np_ranks

    np_ranks = 1/(invrank+1)**2
    df_ranks = pd.Series(np.sum(np_ranks, axis=1), index=df_ranks.index)

    return df_ranks

# other approaches to be considered: references [38, 41, 103] in https://dl.acm.org/doi/pdf/10.1145/3209978.3210186


def combsum(scores: List[pd.Series]) -> pd.Series:

    """
    sums the scores for each feature.

    This function takes a list of Pandas Series objects representing scores,
    concatenates them into a dataframe, replaces NaN values with zeros,
    and computes the sum for each feature

    Based on: Joseph A. Shaw, Edward A. Fox: Combination of Multiple Searches. TREC 1994: 105-108

    Args:
        scores: List of pandas Series, where each series represents feature scores.

    Returns:
        A pandas Series representing the combsum across the provided score Series.

    """
    df_scores = pd.concat(scores, axis=1)  # Combine all rank Series into a DataFrame.
    np_scores = np.array(df_scores)

    df_scores = pd.Series(np.nansum(np_scores, axis=1), index=df_scores.index)


    return df_scores


def combmnz(scores: List[pd.Series]) -> pd.Series:

    """
    sums the scores for each feature and weights the score by the occurrence of each feature.

    This function takes a list of Pandas Series objects representing scores,
    concatenates them into a dataframe, replaces NaN values with zeros,
    and computes the sum for each feature

    Based on: Joseph A. Shaw, Edward A. Fox: Combination of Multiple Searches. TREC 1994: 105-108

    Args:
        scores: List of pandas Series, where each series represents feature scores.

    Returns:
        A pandas Series representing the combsum across the provided score Series.

    """
    df_scores = pd.concat(scores, axis=1)  # Combine all rank Series into a DataFrame.
    np_scores = np.array(df_scores)

    df_scores = pd.Series(np.nansum(np_scores, axis=1), index=df_scores.index)

    df_weights = pd.Series((~np.isnan(np_scores)).sum(axis=1), index=df_scores.index)
    df_scores = pd.concat([df_scores, df_weights], axis=1)
    df_scores[0] = df_scores[0]*df_scores[1]
    df_scores = pd.Series(df_scores[0])

    return df_scores

