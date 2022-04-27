import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from tsfresh.feature_selection import relevance

from t2f.utils.PFA import PFA
import warnings


def features_simple_selection(df: pd.DataFrame, labels: list, top_k: int = 20):
    df = df.dropna(axis='columns')
    sk = SelectKBest(k=min(len(df.columns), top_k))
    sk.fit(df, labels)
    return list(sk.get_feature_names_out())


def features_scoring_selection(df: pd.DataFrame, labels: list, mode: str = 'simple', top_k: int = 20,
                               strategy: str = 'multi'):
    """ Scoring the importance for each feature for the given labels """
    df = df.dropna(axis='columns')
    labels = pd.Series((i for i in labels))
    top_k = min(len(df.columns), top_k)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if strategy == 'tsfresh':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=False)
        relevance_features = relevance_features.sort_values(by='p_value')
        relevance_features = relevance_features.dropna(subset=['p_value'])
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'multi':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=True)
        p_columns = [col for col in relevance_features.columns if col.startswith('p_value')]
        relevance_features = relevance_features.dropna(subset=p_columns)
        relevance_features['p_value'] = relevance_features[p_columns].mean(axis=1)
        relevance_features = relevance_features.sort_values(by='p_value')
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'sk_base':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)

        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'score': sk.scores_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        df_sk.sort_values('score', ascending=False, inplace=True)
        ex = pd.Series(df_sk['score'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'sk_pvalue':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)

        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'pvalues': sk.pvalues_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        df_sk.sort_values('pvalues', ascending=True, inplace=True)
        ex = pd.Series(df_sk['pvalues'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'none':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)
        selector = VarianceThreshold()
        selector.fit(df)
        features = selector.get_feature_names_out()
        top_k = len(features)
        ex = pd.Series([1] * top_k, index=features)

    else:
        raise ValueError('Strategy {} is not supported'.format(strategy))

    top_k_feat = {}
    feats_value = {}
    if mode == 'simple':
        for x in ex[:top_k].index:
            top_k_feat[x] = list(df[x])
        featOrd = pd.DataFrame(top_k_feat)
        # print(featOrd)
        feats_choosed, weight_feats = pfa_scoring(featOrd, 0.9)
        for x in feats_choosed:
            feats_value[x] = list(df[x])
    elif mode == 'domain':
        feat_domain = {}
        feats_choosed = {}
        for x in ex.index:
            domain = x.split('__')[0]
            if not domain in feat_domain.keys():
                feat_domain[domain] = {}
            if len(feat_domain[domain]) < top_k:
                feat_domain[domain][x] = list(df[x])
        # print(feat_domain.keys())
        for key in feat_domain.keys():
            featOrd = pd.DataFrame(feat_domain[key])
            feats_choosed[key], weight_feats = pfa_scoring(featOrd, 0.9)
        for type_feat in feats_choosed.keys():
            for key in feats_choosed[type_feat]:
                feats_value[key] = list(df[key])
    # print(pfa_scoring(featOrd, 0.9))
    # return pd.DataFrame(feats_value)
    return list(feats_value.keys())


def pfa_scoring(df: pd.DataFrame, expl_var_selection: float):
    pfa = PFA()
    feat_PFA, expl_variance_ration = pfa.fit(df, expl_var_selection)
    x = pfa.features_
    column_indices = pfa.indices_
    return feat_PFA, expl_variance_ration
