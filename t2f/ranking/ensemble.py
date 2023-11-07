from typing import List
import pandas as pd


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
