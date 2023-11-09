from typing import Literal
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

from .skfeature.function.similarity_based import fisher_score as fs
from .skfeature.function.similarity_based import lap_score as ls
from .skfeature.function.similarity_based import trace_ratio as tr
from .skfeature.function.information_theoretical_based import MIM, MIFS, MRMR, CIFE, JMI, CMIM, ICAP, DISR
from .skfeature.function.sparse_learning_based import RFS, MCFS, UDFS, NDFS
from .skfeature.utility import construct_W
from .skfeature.utility.sparse_learning import construct_label_matrix
from .skfeature.function.statistical_based import gini_index, CFS


def rank(scores: list, features: list) -> pd.Series:
    """Ranks features based on their respective scores in descending order.

    This function takes two lists: 'scores' and 'features', creates a DataFrame from them, 
    drops any rows with NaN values, sorts the features based on their scores in descending order, 
    and then returns a Series with the sorted scores and their corresponding features.

    Args:
        scores (list): A list of scores (numeric).
        features (list): A list of features (strings or numeric) corresponding to the scores.

    Returns:
        pd.Series: A pandas Series where the index consists of sorted features and the values are the 
        corresponding scores, sorted in descending order.

    """
    df_sk = pd.DataFrame({'score': scores, 'feature': features})  # Create a DataFrame with scores
    df_sk = df_sk.dropna()  # Drop features with NaN scores
    df_sk = df_sk.sort_values('score', ascending=False)  # Sort features by score in descending order
    return pd.Series(df_sk['score'].values, index=df_sk['feature'].values)  # Return Series of scores


def anova(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
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

    # Compute and return and ordered Series of scores
    s = rank(scores=sk.scores_, features=df.columns)
    return s


def fisher_score(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """ Compute the Fisher scores. """
    scores = fs.fisher_score(df.values, y)
    s = rank(scores=scores, features=df.columns)
    return s


def laplace_score(df: pd.DataFrame, **kwargs) -> (pd.Series, pd.Series):
    """ Compute the Laplacian scores. """
    kwargs = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    affinity_matrix = construct_W.construct_W(df.values, **kwargs)
    scores = ls.lap_score(df.values, W=affinity_matrix)
    s = rank(scores=scores, features=df.columns)
    return s


def trace_ratio100(df: pd.DataFrame, y: np.ndarray, style: Literal['fisher', 'laplacian'] = 'fisher') -> pd.Series:
    """ Compute trace ratio criterion for the top 100 features. """
    feat_idx, scores, _ = tr.trace_ratio(X=df.values, y=y, n_selected_features=100, style=style)
    s = rank(scores=scores, features=df.columns.values[feat_idx])
    return s


def trace_ratio(df: pd.DataFrame, y: np.ndarray, style: Literal['fisher', 'laplacian'] = 'fisher') -> pd.Series:
    """ Compute trace ratio criterion. """
    feat_idx, scores, _ = tr.trace_ratio(X=df.values, y=y, n_selected_features=len(df.columns), style=style)
    s = rank(scores=scores, features=df.columns.values[feat_idx])
    return s


def mim(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute MIM mutual information based metric. """
    feat_idx, scores, _ = MIM.mim(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def mifs(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute MIFS mutual information based metric. """
    feat_idx, scores, _ = MIFS.mifs(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def mrmr(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute MRMR mutual information based metric. """
    feat_idx, scores, _ = MRMR.mrmr(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def cife(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute CIFE mutual information based metric. """
    feat_idx, scores, _ = CIFE.cife(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def jmi(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute JMI mutual information based metric. """
    feat_idx, scores, _ = JMI.jmi(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def cmim(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute CMIM mutual information based metric. """
    feat_idx, scores, _ = CMIM.cmim(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def icap(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute ICAP mutual information based metric. """
    feat_idx, scores, _ = ICAP.icap(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def disr(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # ToDO FDB: check rank and score order
    """ Compute DISR mutual information based metric. """
    feat_idx, scores, _ = DISR.disr(X=df.values, y=y)
    s = rank(scores=np.arange(1, len(feat_idx) + 1)[::-1], features=df.columns.values[feat_idx])
    return s


def rfs(df: pd.DataFrame, y: np.ndarray, gamma: float = 1) -> pd.Series:
    """ Efficient and Robust Feature Selection (RFS). """
    y_matrix = construct_label_matrix(np.array(y, dtype=int))
    scores = RFS.rfs(X=df.values, Y=y_matrix, gamma=gamma)
    scores = (scores * scores).sum(axis=1)
    s = rank(scores=scores, features=df.columns)
    return s


def mcfs(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """ Multi-Cluster Feature Selection (MCFS). """
    # ToDo FDB: why here is used the max instead of sum?
    scores = MCFS.mcfs(X=df.values, n_selected_features=len(df.columns), n_clusters=len(np.unique(y)))
    scores = scores.max(axis=1)
    s = rank(scores=scores, features=df.columns)
    return s


def udfs(df: pd.DataFrame, y: np.ndarray, gamma: float = 0.1, k: int = 5) -> pd.Series:
    """ L2/1-norm regularized discriminative feature selection for unsupervised learning. """
    scores = UDFS.udfs(X=df.values, n_clusters=len(np.unique(y)), gamma=gamma, k=k)
    scores = (scores * scores).sum(axis=1)
    s = rank(scores=scores, features=df.columns)
    return s


def ndfs(df: pd.DataFrame, y: np.ndarray, alpha: float = 1, beta: float = 1, gamma: float = 10e8) -> pd.Series:
    """ Unsupervised feature selection using non-negative spectral analysis. """
    n_clusters = len(np.unique(y))
    y_matrix = construct_label_matrix(np.array(y, dtype=int))
    t_matrix = np.dot(y_matrix.transpose(), y_matrix)
    f_matrix = np.dot(y_matrix, np.sqrt(np.linalg.inv(t_matrix)))
    f_matrix = f_matrix + 0.02 * np.ones((len(df), n_clusters))
    params = {'X': df.values, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'F0': f_matrix, 'n_clusters': n_clusters}
    scores = NDFS.ndfs(**params)
    scores = (scores * scores).sum(1)
    s = rank(scores=scores, features=df.columns)
    return s


def gini(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """ This function implements the gini index feature selection. """
    scores = gini_index.gini_index(X=df.values, y=y)
    scores = -scores
    s = rank(scores=scores, features=df.columns)
    return s


def cfs(df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """ This function uses a correlation based heuristic to evaluate the worth of features which is called CFS. """
    feat_idx = CFS.cfs(X=df.values, y=y)
    scores = np.ones(len(feat_idx))
    s = rank(scores=list(scores), features=df.columns.values[feat_idx])
    return s
