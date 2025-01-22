import warnings
import pandas as pd
from sklearn.feature_selection import SelectKBest

from ..selection.PFA import pfa_scoring


class Ranker(object):
    def __init__(self):
        pass

    def ranking(self, df: pd.DataFrame, y: list, top_k: int) -> list:
        # Identify constant columns
        constant_columns = df.columns[df.nunique() <= 1].tolist()
        # Remove constant columns from the DataFrame
        df = df.drop(constant_columns, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Apply ranking mechanism
            sk = SelectKBest(k='all')
            sk.fit(df, y)


        df_sk = pd.DataFrame({'score': sk.scores_, 'feature': df.columns})
        df_sk = df_sk.dropna()
        df_sk.sort_values('score', ascending=False, inplace=True)

        # Select topk
        ex = pd.Series(df_sk['score'].values[:top_k], index=df_sk['feature'].values[:top_k])

        # Apply PFA to select the feature which keep 90% of the information
        top_features, _ = pfa_scoring(df[ex.index], 0.9)

        return top_features
