import warnings

import pandas as pd
from sklearn.feature_selection import SelectKBest


# ToDo FDB: insert other

def anova(df: pd.DataFrame, y: list) -> pd.Series:
    """
    Performs ANOVA feature selection for the provided dataframe.

    This function applies the SelectKBest method from sklearn to the provided dataframe
    to select features based on the ANOVA F-value between label/feature for classification tasks.

    Args:
        df: A pandas DataFrame with the features.
        y: The target variable as a list.

    Returns:
        A pandas Series with features as the index and their corresponding ANOVA F-scores as values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress sklearn warnings
        sk = SelectKBest(k='all')  # Use all features
        sk.fit(df, y)  # Fit the SelectKBest with the dataframe and target

    df_sk = pd.DataFrame({'score': sk.scores_, 'feature': df.columns})  # Create a DataFrame with scores
    df_sk = df_sk.dropna()  # Drop features with NaN scores
    df_sk = df_sk.sort_values('score', ascending=False)  # Sort features by score in descending order
    return pd.Series(df_sk['score'].values, index=df_sk['feature'].values)  # Return Series of scores
