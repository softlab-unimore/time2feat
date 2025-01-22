import time
import numpy as np
import pandas
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from tsfresh.feature_selection import relevance
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from time2feat.selection.PFA import PFA
import warnings

from time2feat.model.preprocessing import apply_transformation
from time2feat.model.clustering import ClusterWrapper, cluster_metrics


def feature_selection(df_feats: pd.DataFrame, labels: dict = None, context: dict = None,
                      external_feat: pd.DataFrame = None,
                      select_external_feat: bool = False):
    df_feats = pd.concat([df_feats, external_feat], axis=1)
    if labels:
        train_idx = list(labels.keys())
        test_idx = [i for i in range(len(df_feats)) if i not in train_idx]
        y_train = list(labels.values())
        df_train_features = df_feats.iloc[train_idx, :].reset_index(drop=True)
        df_test_features = df_feats.iloc[test_idx, :]

        df_test_features = pd.concat([df_train_features, df_test_features], axis=0, ignore_index=True)
        params = context
        new_params = simple_grid_search(df_train_features, y_train, df_test_features, params,
                                        external_feat=external_feat)
        top_k = new_params['top_k']
        score_mode = new_params['score_mode']

        top_features = features_scoring_selection(df_train_features, y_train, mode=score_mode, top_k=top_k,
                                                  strategy=params['strategy'], pfa_values=params['pfa_value'],
                                                  external_feat=external_feat)
    else:
        # if external_feat and select_external_feat, top_features is a tuple with top_feats_ts and top_feats_ext
        top_features = features_scoring_selection(df_feats, [], mode='simple', top_k=1,
                                                  strategy='none', pfa_values=context['pfa_value'],
                                                  external_feat=external_feat,
                                                  select_external_feat=select_external_feat)

    return top_features


def features_simple_selection(df: pd.DataFrame, labels: list, top_k: int = 20, external_feat: pd.DataFrame = None):
    df = df.dropna(axis='columns')
    if external_feat is not None:
        df.drop(columns=external_feat.columns.tolist(), inplace=True)
    sk = SelectKBest(k=min(len(df.columns), top_k))
    sk.fit(df, labels)
    return list(sk.get_feature_names_out())


def features_scoring_selection(df: pd.DataFrame, labels: list, mode: str = 'simple', top_k: int = 20,
                               strategy: str = 'multi', pfa_values: float = 0.9, external_feat: pandas.DataFrame = None,
                               select_external_feat: bool = False):
    """ Scoring the importance for each feature for the given labels """
    df = df.dropna(axis='columns')
    labels = pd.Series((i for i in labels))
    top_k = min(len(df.columns), top_k)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ex_ext = None
    top_k_ext = None

    if strategy == 'tsfresh' and labels.nunique() > 2:
        strategy = 'multi'
    elif strategy == 'multi' and labels.nunique() < 3:
        strategy = 'tsfresh'

    if strategy == 'tsfresh':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=False)
        if external_feat is not None and not select_external_feat:
            relevance_features = relevance_features[~relevance_features['feature'].isin(external_feat.columns.tolist())]
        relevance_features = relevance_features.sort_values(by='p_value')
        relevance_features = relevance_features.dropna(subset=['p_value'])
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'multi':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=True)
        if external_feat is not None and not select_external_feat:
            relevance_features = relevance_features[~relevance_features['feature'].isin(external_feat.columns.tolist())]
        p_columns = [col for col in relevance_features.columns if col.startswith('p_value')]
        relevance_features = relevance_features.dropna(subset=p_columns)
        relevance_features['p_value'] = relevance_features[p_columns].mean(axis=1)
        relevance_features = relevance_features.sort_values(by='p_value')
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'sk_base':
        # ToDo: manage all null case
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)
        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'score': sk.scores_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        if external_feat is not None and not select_external_feat:
            df_sk = df_sk[~df_sk['feature'].isin(external_feat.columns.tolist())]
        df_sk.sort_values('score', ascending=False, inplace=True)
        ex = pd.Series(df_sk['score'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'sk_pvalue':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)
        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'pvalues': sk.pvalues_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        if external_feat is not None and not select_external_feat:
            df_sk = df_sk[~df_sk['feature'].isin(external_feat.columns.tolist())]
        df_sk.sort_values('pvalues', ascending=True, inplace=True)
        ex = pd.Series(df_sk['pvalues'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'none':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)
        selector = VarianceThreshold()
        selector.fit(df)
        features = selector.get_feature_names_out()
        if external_feat is not None:
            features = features[~np.isin(features, external_feat.columns.tolist())]
        top_k = len(features)
        ex = pd.Series([1] * top_k, index=features)

        if external_feat is not None and select_external_feat:
            external_feat = external_feat.drop(df.columns[cond], axis=1)
            selector.fit(external_feat)
            features_ext = selector.get_feature_names_out()

            top_k_ext = len(features_ext)
            ex_ext = pd.Series([1] * top_k_ext, index=features_ext)

    else:
        raise ValueError('Strategy {} is not supported'.format(strategy))

    feats_value = get_pfa_score(df, ex, mode, pfa_values, top_k)
    if strategy == "none" and external_feat is not None and select_external_feat:
        feats_value_ts = feats_value
        feats_value_ext = get_pfa_score(external_feat, ex_ext, mode, pfa_values, top_k_ext)

        return list(feats_value_ts.keys()), list(feats_value_ext.keys())
    else:
        return list(feats_value.keys())


def get_pfa_score(df, ex, mode, pfa_values, top_k):
    top_k_feat = {}
    feats_value = {}
    if mode == 'simple':
        for x in ex[:top_k].index:
            top_k_feat[x] = list(df[x])
        featOrd = pd.DataFrame(top_k_feat)
        # print(featOrd)
        feats_choosed, weight_feats = pfa_scoring(featOrd, pfa_values)
        for x in feats_choosed:
            feats_value[x] = list(df[x])
    elif mode == 'domain':
        feat_domain = {}
        feats_choosed = {}
        for x in ex.index:
            domain = x.split('__')[0]
            if domain not in feat_domain.keys():
                feat_domain[domain] = {}
            if len(feat_domain[domain]) < top_k:
                feat_domain[domain][x] = list(df[x])
        # print(feat_domain.keys())
        for key in feat_domain.keys():
            featOrd = pd.DataFrame(feat_domain[key])
            feats_choosed[key], weight_feats = pfa_scoring(featOrd, pfa_values)
        for type_feat in feats_choosed.keys():
            for key in feats_choosed[type_feat]:
                feats_value[key] = list(df[key])
    return feats_value


def pfa_scoring(df: pd.DataFrame, expl_var_selection: float):
    pfa = PFA()
    feat_PFA, expl_variance_ration = pfa.fit(df, expl_var_selection)
    x = pfa.features_
    column_indices = pfa.indices_
    return feat_PFA, expl_variance_ration


def simple_grid_search(df_base: pd.DataFrame, y_base: list, df_complete: pd.DataFrame, params: dict,
                       external_feat: pd.DataFrame = None):
    cond = 'model_type' in params and 'transform_type' in params and 'score_mode' in params
    assert cond, 'Please provide all mandatory params (model, transform and score mode)'

    cond = (pd.isnull(df_base)) & (pd.isnull(df_complete.iloc[:len(df_base)]))
    mask = (df_base == df_complete.iloc[:len(df_base)])
    mask = np.where(cond, True, mask)
    assert np.all(mask), 'df_base must be in the first record of df_complete'

    # Optional params
    k_best = params.get('k_best', False)
    strategy = params.get('strategy', 'sk_base')
    pre_transform = params.get('pre_transform', False)

    grid_params = {
        'top_k': [params['top_k']],
        'score_mode': [params['score_mode']],
        'transform_type': [params['transform_type']],
    }

    if strategy.startswith('sk'):
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2
        # grid_params['transform_type'] = ['minmax', 'std']
        grid_params['score_mode'] = ['simple', 'domain']
    elif strategy.startswith('none'):
        grid_params['transform_type'] = ['minmax', 'std', None]
        grid_params['score_mode'] = ['simple']

    else:
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2

    results = []
    time.sleep(0.1)
    for new_params in tqdm(ParameterGrid(grid_params)):
        if not k_best:
            max_rows = len(df_base)  # 800
            top_features = features_scoring_selection(df_base.iloc[:max_rows, :], y_base[:max_rows],
                                                      mode=new_params['score_mode'], top_k=new_params['top_k'],
                                                      strategy=strategy, external_feat=external_feat)
        else:
            top_features = features_simple_selection(df_base, y_base, top_k=new_params['top_k'],
                                                     external_feat=external_feat)

        if external_feat is not None:
            top_features.extend(external_feat.columns.tolist())

        # Extract train and test
        x_train = df_base[top_features].values
        x_test = df_complete[top_features].values

        # Transformation step
        if new_params['transform_type'] and not pre_transform:
            _, x_test = apply_transformation(x_train, x_test, new_params['transform_type'])

        # Clustering step
        num_labels = len(set(y_base))
        model = ClusterWrapper(model_type=params['model_type'], n_clusters=num_labels)
        y_pred = model.fit_predict(x_test)

        # Compute results
        res_metrics = cluster_metrics(y_base, y_pred[:len(y_base)])
        res_metrics.update(new_params)
        results.append(res_metrics)

    df_res = pd.DataFrame(results)
    df_res = df_res.fillna('None')
    df_res = df_res.groupby(['top_k', 'score_mode', 'transform_type'])['nmi'].mean().reset_index()
    df_res = df_res.replace(['None'], [None])
    df_res.sort_values('nmi', ascending=False, inplace=True)
    # print(df_res)
    new_params = {
        'top_k': df_res['top_k'].iloc[0],
        'score_mode': df_res['score_mode'].iloc[0],
        'transform_type': df_res['transform_type'].iloc[0],
    }
    return new_params
