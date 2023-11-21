import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, mutual_info_regression, \
    mutual_info_classif, chi2, f_classif

from .skfeature.function.similarity_based import fisher_score, lap_score, SPEC, trace_ratio
from .skfeature.function.information_theoretical_based import MIM, MIFS, MRMR, CIFE, JMI, CMIM, ICAP, DISR
from .skfeature.function.sparse_learning_based import RFS, MCFS, UDFS, NDFS
from .skfeature.utility import construct_W
from .skfeature.utility.sparse_learning import construct_label_matrix
from .skfeature.function.statistical_based import t_score, gini_index, CFS


def select_topk(target: pd.Series, rank: pd.Series, topk: int) -> pd.Series:
    """
    Select the topk elements from the target. The topk elements are selected based on the provided ranking.
    If multiple elements have the same ranking score then the elements are selected in order.

    Args:
        target: collection of elements where to select topk elements
        rank: the rank score for each item in target (from 1 to N, where 1 means the element is the most important)
        topk: the number of topk elements to select

    Returns:
        Selection of the target elements - pd.Series
    """

    sorted_rank = rank.sort_values()
    topk_elements = sorted_rank.head(topk).index

    return target.loc[topk_elements]


class FeatureSelector:
    """
    Class that implements feature selection methods.
    Several methods are supported:
    - FROM SKLEARN ('mi_regr', 'mi_class', 'class_importances', 'regr_importances', 'class_sfs', 'regr_sfs')
    - SIMILARITY BASED ('fisher', 'laplace', 'spec', 'trace_ratio')
    - INFORMATION THEORETICAL BASED ('mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr')
    - SPARSE LEARNING BASED ('rfs', 'mcfs', 'udfs', 'ndfs')
    - STATISTICAL BASED ('chi2', 'low_variance', 't-score', 'f-score', 'gini', 'cfs')

    Most of the methods are taken from https://github.com/jundongl/scikit-feature.

    References:
        Jundong Li et al. 2017. Feature Selection: A Data Perspective. ACM Comput. Surv. https://doi.org/10.1145/3136625

    Examples:
        >>> feat_table = pd.DataFrame(data=np.random.rand(10, 4), columns=range(4))
        >>> feat_table['y'] = range(10)
        >>> feat_dict = {'all': {'all': feat_table}}
        >>> feat_selector = FeatureSelector(feat_dict=feat_dict)
        >>> feat_sel_grid_conf = {
        >>>     # FROM SKLEARN
        >>>     'mi_regr': [{}],
        >>>     'mi_class': [{}],
        >>>     'class_importances': [
        >>>         {'model_name': 'dtc', 'model_params': None},
        >>>         {'model_name': 'dtc', 'model_params': {'max_depth': 5, 'random_state': 42}}
        >>>     ],
        >>>     'regr_importances': [
        >>>         {'model_name': 'lasso', 'model_params': None},
        >>>         {'model_name': 'lasso', 'model_params': {'alpha': 0.8, 'random_state': 42}}
        >>>     ],
        >>>     'class_sfs': [
        >>>         {'model_name': 'dtc', 'num_features': 10, 'direction': 'forward', 'model_params': None},
        >>>         {'model_name': 'dtc', 'num_features': 10, 'direction': 'backward', 'model_params': None},
        >>>         {'model_name': 'dtc', 'num_features': 10, 'direction': 'forward',
        >>>          'model_params': {'max_depth': 5, 'random_state': 42}},
        >>>         {'model_name': 'dtc', 'num_features': 10, 'direction': 'backward',
        >>>          'model_params': {'max_depth': 5, 'random_state': 42}},
        >>>     ],
        >>>     'regr_sfs': [
        >>>         {'model_name': 'lasso', 'num_features': 10, 'direction': 'forward', 'model_params': None},
        >>>         {'model_name': 'lasso', 'num_features': 10, 'direction': 'backward', 'model_params': None},
        >>>         {'model_name': 'lasso', 'num_features': 10, 'direction': 'forward',
        >>>          'model_params': {'alpha': 0.8, 'random_state': 42}},
        >>>         {'model_name': 'lasso', 'num_features': 10, 'direction': 'backward',
        >>>          'model_params': {'alpha': 0.8, 'random_state': 42}},
        >>>     ],
        >>>     # SIMILARITY BASED
        >>>     'fisher': [{}],
        >>>     'laplace': [{}],
        >>>     'spec': [{'style': -1}, {'style': 0}, {'style': 3}],
        >>>     'trace_ratio': [{'num_features': 10, 'style': 'fisher'}, {'num_features': 10, 'style': 'laplacian'}],
        >>>     # INFORMATION THEORETICAL BASED
        >>>     'mim': [{}, {'num_features': 10}],
        >>>     'mifs': [{}, {'num_features': 10}],
        >>>     'mrmr': [{}, {'num_features': 10}],
        >>>     'cife': [{}, {'num_features': 10}],
        >>>     'jmi': [{}, {'num_features': 10}],
        >>>     'cmim': [{}, {'num_features': 10}],
        >>>     'icap': [{}, {'num_features': 10}],
        >>>     'disr': [{}, {'num_features': 10}],
        >>>     # SPARSE LEARNING BASED
        >>>     'rfs': [{'gamma': 1}],
        >>>     'mcfs': [{'num_features': 10}, {'num_features': 10, 'num_clusters': 5}],
        >>>     'udfs': [{'gamma': 0.1, 'k': 5}, {'num_clusters': 5, 'gamma': 0.1, 'k': 5}],
        >>>     'ndfs': [
        >>>         {'alpha': 1, 'beta': 1, 'gamma': 10e8}, {'num_clusters': 5, 'alpha': 1, 'beta': 1,'gamma': 10e8}
        >>>     ],
        >>>     # STATISTICAL BASED
        >>>     'chi2': [{}],
        >>>     'low_variance': [{'thr': 0.8}],
        >>>     't_score': [{}],
        >>>     'f_score': [{}],
        >>>     'gini': [{}],
        >>>     'cfs': [{}]
        >>> }
        >>> feat_selector.get_scores_from_grid(grid_params=feat_sel_grid_conf, output='score')
    """
    # With SVMs and logistic, the parameter C controls the sparsity: the smaller C the fewer features selected.
    # With Lasso, the higher the alpha parameter, the fewer features selected.

    models = {
        ## CLASSIFIERS ##
        # L1
        'svm': {'name': LinearSVC, 'params': {'C': 0.01, 'penalty': "l1", 'dual': False, 'random_state': 42}},
        'logistic': {'name': LogisticRegression,
                     'params': {'C': 0.01, 'penalty': "l1", 'solver': 'liblinear', 'random_state': 42}},
        # No penalty
        'dtc': {'name': DecisionTreeClassifier, 'params': {'max_depth': 3, 'random_state': 42}},
        'rfc': {'name': RandomForestClassifier, 'params': {'max_depth': 3, 'n_estimators': 100, 'random_state': 42}},
        'adaboost_dtc': {'name': AdaBoostClassifier, 'params': {'n_estimators': 100, 'random_state': 42}},
        'gbc': {'name': GradientBoostingClassifier,
                'params': {'max_depth': 3, 'n_estimators': 100, 'random_state': 42}},
        ## REGRESSORS ##
        # L1
        'lasso': {'name': Lasso, 'params': {'alpha': 1.0, 'random_state': 42}},
        # L2
        'ridge': {'name': Ridge, 'params': {'alpha': 1.0, 'random_state': 42}},
    }

    regr_methods = ['mi_regr', 'regr_importances', 'regr_sfs']
    class_methods = ['mi_class', 'class_importances', 'class_sfs', 'fisher', 'trace_ratio', 'mim', 'mifs', 'mrmr',
                     'cife', 'jmi', 'cmim', 'icap', 'disr', 'rfs', 'chi2', 't-score', 'f-score', 'gini', 'cfs']
    unsup_methods = ['laplace', 'spec', 'mcfs', 'udfs', 'ndfs', 'low_variance']
    methods_with_n_feat = ['mcfs', 'mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr', 'trace_ratio',
                           'class_sfs', 'regr_sfs']

    def __init__(self, feat_dict: dict):
        """
        Initialize the DataFrames with the features describing stocks in several timestamps.

        Args:
            feat_dict: dictionary containing for each stock and timestamp a feature table
               Example where all the stocks and timestamps are aggregated:
                    {
                        'all': {
                            'all': feature_tab
                        }
                    }
               Example where the stocks are aggregated, but the timestamps are not (2 times are considered):
                    {
                        'all': {
                            1: feature_tab,
                            2: feature_tab,
                        }
                    }
               Example where neither the stocks nor the timestamps are aggregated:
                    {
                        'stock1': {
                            1: feature_tab,
                            2: feature_tab,
                        },
                        ...
                        'stockN': {
                            1: feature_tab,
                            2: feature_tab,
                        },
                    }
        """

        # The feature tables contain also the label column
        # If the label is a continuous variable then discretize it by computing positive/negative trends
        for stock_feat_dict in feat_dict.values():
            for feat_tab in stock_feat_dict.values():
                y = feat_tab['y']
                if len(y.unique()) > 3:
                    feat_tab['class'] = feat_tab['y'].apply(lambda x: x >= 0)
                else:
                    feat_tab['class'] = feat_tab['y'].copy()

        self.feat_dict = feat_dict
        self.label_col = 'class'

    @staticmethod
    def mutual_info_regr(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        Estimate mutual information for a continuous target variable (i.e., only in a regression scenario).
        Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
        between the variables. It is equal to zero if and only if two random variables are independent, and higher
        values mean higher dependency.
        The higher the score, the more important the feature is.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            The mutual information scores for each feature - pd.Series
            Rank scores - pd.Series
        """

        y = X[label]
        X = X.drop(['y', 'class'], axis=1)

        scores = mutual_info_regression(X=X, y=y, random_state=42)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def mutual_info_class(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        Estimate mutual information for a discrete target variable (i.e., only in a classification scenario).
        Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
        between the variables. It is equal to zero if and only if two random variables are independent, and higher
        values mean higher dependency.
        The higher the score, the more important the feature is.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            The mutual information scores for each feature - pd.Series
            Rank scores - pd.Series
        """

        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        scores = mutual_info_classif(X=X, y=y, random_state=42)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def feat_importances(X: pd.DataFrame, label: str, model_name: str, model_params: dict = None,
                         norm_by_var: bool = False) -> (pd.Series, pd.Series):
        """
        Compute feature scores by training an intrinsically interpretable estimator and by extracting from it some
        feature importance scores (which are stored in a specific attribute, such as coef_ or feature_importances_
        depending on the model).
        The higher the absolute score, the more important the feature is.

        References:
            https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html

        Args:
            X: feature table
            label: the name of the label column in the feature table
            model_name: the name of the model used for extracting feature importances
            model_params: the collection of model params
            norm_by_var: whether to normalize importance scores by feature variance

        Returns:
            Feature importances - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        model_item = FeatureSelector.models[model_name]
        model_obj = model_item['name']
        if model_params is None:
            if 'params' not in model_item:
                raise ValueError("Model configuration not provided!")
            model_params = model_item['params']

        model = model_obj(**model_params)
        model.fit(X, y)

        if hasattr(model, 'coef_'):
            scores = np.reshape(model.coef_, -1)
        elif hasattr(model, 'feature_importances_'):
            scores = model.feature_importances_
        else:
            raise NotImplementedError()

        if norm_by_var is True:
            scores *= X.std(axis=0)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.abs().rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def sfs(X: pd.DataFrame, label: str, model_name: str, num_features: Union[int, float] = None,
            direction: str = 'forward', model_params: dict = None) -> (pd.Series, pd.Series):
        """
        Sequential Feature Selection.
        This Sequential Feature Selector adds (forward selection) or removes (backward selection) features to form a
        feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove
        based on the cross-validation score of an estimator.
        Note that this method selects the features and doesn't score them.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html

        Args:
            X: feature table
            label: the name of the label column in the feature table
            model_name: the name of the model used for selecting the features
            num_features: number of features to select
                          If None, half of the features are selected.
                          If integer, the parameter is the absolute number of features to select.
                          If float between 0 and 1, it is the fraction of features to select.
            direction: whether to perform forward selection or backward selection.
            model_params: the collection of model params

        Returns:
            Feature selected by SFS to the provided model - pd.Series
            Rank scores: 1 is assigned to selected features and NaN to the remaining ones - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        model_item = FeatureSelector.models[model_name]
        model_obj = model_item['name']
        if model_params is None:
            if 'params' not in model_item:
                raise ValueError("Model configuration not provided!")
            model_params = model_item['params']

        model = model_obj(**model_params)
        sfs = SequentialFeatureSelector(
            estimator=model, n_features_to_select=num_features, direction=direction
        )
        sfs.fit(X, y)
        mask = sfs.get_support()
        scores = pd.Series(mask.astype(int), index=X.columns)
        rank = pd.Series(index=X.columns)
        rank.loc[mask] = 1

        return scores, rank

    @staticmethod
    def fisher_score(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        Compute the Fisher scores.
        1. Construct the affinity matrix W in fisher score way
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1
        The larger the fisher score, the more important the feature is.

        References:
            He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
            Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            The Fisher scores - pd.Series
            Ranking scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        scores = fisher_score.fisher_score(X.values, y.values)

        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def laplace_score(X: pd.DataFrame) -> (pd.Series, pd.Series):
        """
        Compute the Laplacian scores.
        1. Construct the affinity matrix W if it is not specified
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)
        The smaller the laplacian score is, the more important the feature is.

        References:
            He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.

        Args:
            X: feature table

        Returns:
            The laplacian scores - pd.Series
            Rank scores - pd.Series
        """
        X = X.drop(['y', 'class'], axis=1)

        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X.values, **kwargs_W)
        scores = lap_score.lap_score(X.values, W=W)

        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(method='min')

        return scores, rank

    @staticmethod
    def spec(X: pd.DataFrame, style: int) -> (pd.Series, pd.Series):
        """
        Compute SPEC scores.
        If style = -1 or 0, the higher the score, the more important the feature is.
        If style != -1 and 0, the lower the score, the more important the feature is.

        References:
            Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.

        Args:
            X: feature table
            style: the type of ranking method to use
                style == -1, the first feature ranking function, use all eigenvalues
                style == 0, the second feature ranking function, use all except the 1st eigenvalue
                style >= 2, the third feature ranking function, use the first k except 1st eigenvalue

        Returns:
            The SPEC scores - pd.Series
            Rank scores - pd.Series
        """
        X = X.drop(['y', 'class'], axis=1)

        # Check for abnormal values
        # Remove the features with an absolute mean greater than 1000
        # If this is not done the SPEC computation will raise an error saying that "eigenvalues did not converge"
        # FIXME: other ideas?
        X_avgs = X.mean(0).abs()
        clean_feat = X.columns[X_avgs < 1000]
        X_clean = X[clean_feat]

        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_clean.values, **kwargs_W)
        scores = SPEC.spec(X=X_clean.values, W=W, style=style)

        out_scores = pd.Series(index=X.columns)
        out_scores.loc[X_clean.columns] = scores

        # If style = -1 or 0, the higher the score, the more important the feature is
        if style == -1 or style == 0:
            rank = out_scores.rank(ascending=False, method='min')
        # If style != -1 and 0, the lower the score, the more important the feature is
        elif style != -1 and style != 0:
            rank = out_scores.rank(method='min')
        else:
            raise ValueError("Wrong style value!")

        return out_scores, rank

    @staticmethod
    def trace_ratio(X: pd.DataFrame, label: str, num_features: int, style: str = 'fisher') -> (pd.Series, pd.Series):
        """
        Implement trace ratio criterion for feature selection.
        The higher the score, the more important the feature is.

        References:
            Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008

        Args:
            X: feature table
            label: the name of the label column in the feature table
            num_features: the number of features to select
            style: the method used for building the affinity matrices
                style == 'fisher', build between-class and within-class affinity matrices in a fisher score way
                style == 'laplacian', build between-class and within-class affinity matrices in a laplacian score way

        Returns:
            The trace-ratio scores for all the features (a NaN value is assigned to non-selected features) - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        feat_idx, scores, _ = trace_ratio.trace_ratio(
            X=X.values, y=y.values, n_selected_features=num_features, style=style
        )

        out_scores = pd.Series(index=X.columns)
        out_scores.iloc[feat_idx] = scores
        rank = out_scores.rank(ascending=False, method='min')

        return out_scores, rank

    @staticmethod
    def mi_family(X: pd.DataFrame, label: str, mi_method: str, num_features: int = None) -> (pd.Series, pd.Series):
        """
        Compute some mutual information based metric.

        !!Attention!!: the scores are NOT in the same order as the ranking!

        Args:
            X: feature table
            label: the name of the label column in the feature table
            mi_method: the name of the feature selection method to use from the category of mutual information metrics
            num_features: optional number of features to select

        Returns:
            Mutual information derived scores - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        if num_features == 'all':
            num_features = X.values.shape[1]

        if mi_method == 'mim':
            if num_features is not None:
                feat_idx, scores, _ = MIM.mim(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = MIM.mim(X=X.values, y=y.values)
        elif mi_method == 'mifs':
            if num_features is not None:
                feat_idx, scores, _ = MIFS.mifs(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = MIFS.mifs(X=X.values, y=y.values)
        elif mi_method == 'mrmr':
            if num_features is not None:
                feat_idx, scores, _ = MRMR.mrmr(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = MRMR.mrmr(X=X.values, y=y.values)
        elif mi_method == 'cife':
            if num_features is not None:
                feat_idx, scores, _ = CIFE.cife(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = CIFE.cife(X=X.values, y=y.values)
        elif mi_method == 'jmi':
            if num_features is not None:
                feat_idx, scores, _ = JMI.jmi(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = JMI.jmi(X=X.values, y=y.values)
        elif mi_method == 'cmim':
            if num_features is not None:
                feat_idx, scores, _ = CMIM.cmim(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = CMIM.cmim(X=X.values, y=y.values)
        elif mi_method == 'icap':
            if num_features is not None:
                feat_idx, scores, _ = ICAP.icap(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = ICAP.icap(X=X.values, y=y.values)
        elif mi_method == 'disr':
            if num_features is not None:
                feat_idx, scores, _ = DISR.disr(X=X.values, y=y.values, n_selected_features=num_features)
            else:
                feat_idx, scores, _ = DISR.disr(X=X.values, y=y.values)
        else:
            raise ValueError("Wrong mutual information metric!")

        out_scores = pd.Series(index=X.columns)
        out_scores.iloc[feat_idx] = scores
        # The feature indices are reported in order by importance: the first index refers to the most important feature
        rank = out_scores.copy()
        rank.iloc[feat_idx] = range(1, len(scores) + 1)

        return out_scores, rank

    @staticmethod
    def rfs(X: pd.DataFrame, label: str, gamma: float = 1) -> (pd.Series, pd.Series):
        """
        Efficient and Robust Feature Selection (RFS) via joint l21-norms minimization
        min_W||X^T W - Y||_2,1 + gamma||W||_2,1.
        The higher the score, the more important the feature is.

        References:
            Nie, Feiping et al. "Efficient and Robust Feature Selection via Joint l2,1-Norms Minimization" NIPS 2010

        Args:
            X: feature table
            label: the name of the label column in the feature table
            gamma: parameter for weighting the contribution of the regularization term

        Returns:
            RFS feature scores - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)
        y = construct_label_matrix(y.values.astype(int))

        scores = RFS.rfs(X=X.values, Y=y, gamma=gamma)
        scores = (scores * scores).sum(1)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def mcfs(X: pd.DataFrame, num_features: int, label: str = None, num_clusters: int = None) -> (pd.Series, pd.Series):
        """
        Multi-Cluster Feature Selection (MCFS).
        The higher the score, the more important the feature is.

        References:
            Cai, Deng et al. "Unsupervised Feature Selection for Multi-Cluster Data." KDD 2010

        Args:
            X: feature table
            num_features: number of features to select
            label: the name of the label column in the feature table. In general this info is not needed because this is
                   an unsupervised method, however if the labels are available they are used to detect automatically the
                   number of clusters, which is usually set as the number of classes in the ground truth
            num_clusters: the number of cluster to create as an intermediate step of MCFS

        Returns:
            MCFS feature scores - pd.Series
            Rank scores - pd.Series
        """
        if label is None and num_clusters is None:
            raise ValueError("Provide one between label and num_clusters!")

        drop_cols = ['y', 'class']
        if num_clusters is None:
            drop_cols = ['y', 'class', label]
            y = X[label].values
            num_clusters = len(np.unique(y))

        X = X.drop(drop_cols, axis=1)

        scores = MCFS.mcfs(X=X.values, n_selected_features=num_features, n_clusters=num_clusters)
        scores = scores.max(1)  # FIXME: why here is used the max instead of sum?
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def udfs(X: pd.DataFrame, label: str = None, num_clusters: int = None, gamma: float = 0.1,
             k: int = 5) -> (pd.Series, pd.Series):
        """
        L2,1-norm regularized discriminative feature selection for unsupervised learning,
        i.e., min_W Tr(W^T M W) + gamma ||W||_{2,1}, s.t. W^T W = I.
        The higher the score, the more important the feature is.

        References:
            Yang,Yi et al. "l2,1-Norm Regularized Discriminative Feature Selection for Unsupervised Learning." AAAI 2012

        Args:
            X: feature table
            label: the name of the label column in the feature table. In general this info is not needed because this is
                   an unsupervised method, however if the labels are available they are used to detect automatically the
                   number of clusters, which is usually set as the number of classes in the ground truth
            num_clusters: the number of cluster to create as an intermediate step of UDFS
            gamma: parameter for weighting the contribution of the regularization term
            k: number of nearest neighbor

        Returns:
            UDFS feature scores - pd.Series
            Rank scores - pd.Series
        """
        if label is None and num_clusters is None:
            raise ValueError("Provide one between label and num_clusters!")

        drop_cols = ['y', 'class']
        if num_clusters is None:
            drop_cols = ['y', 'class', label]
            y = X[label].values
            num_clusters = len(np.unique(y))

        X = X.drop(drop_cols, axis=1)

        scores = UDFS.udfs(X=X.values, n_clusters=num_clusters, gamma=gamma, k=k)
        scores = (scores * scores).sum(1)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def ndfs(X: pd.DataFrame, label: str = None, num_clusters: int = None, alpha: float = 1, beta: float = 1,
             gamma: float = 10e8) -> (pd.Series, pd.Series):
        """
        Unsupervised feature selection using nonnegative spectral analysis, i.e.,
        min_{F,W} Tr(F^T L F) + alpha*(||XW-F||_F^2 + beta*||W||_{2,1}) + gamma/2 * ||F^T F - I||_F^2 s.t. F >= 0.
        The higher the score, the more important the feature is.

        References:
            Li, Zechao, et al. "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI. 2012

        Args:
            X: feature table
            label: the name of the label column in the feature table. In general this info is not needed because this is
                   an unsupervised method, however if the labels are available they are used to detect automatically the
                   number of clusters, which is usually set as the number of classes in the ground truth
            num_clusters: the number of cluster to create as an intermediate step of UDFS
            alpha: parameter alpha in objective function
            beta: parameter beta in objective function
            gamma: a very large number used to force F^T F = I

        Returns:
            NDFS feature scores - pd.Series
            Rank scores - pd.Series
        """
        if label is None and num_clusters is None:
            raise ValueError("Provide one between label and num_clusters!")

        if num_clusters is None:
            y = X[label]
            num_clusters = len(np.unique(y.values))
            X = X.drop(['y', 'class', label], axis=1)
            Y = construct_label_matrix(y.values.astype(int))
            T = np.dot(Y.transpose(), Y)
            F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
            F = F + 0.02 * np.ones((len(X), num_clusters))
            params = {'X': X.values, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'F0': F, 'n_clusters': num_clusters}
        else:
            X = X.drop(['y', 'class'], axis=1)
            params = {'X': X.values, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'n_clusters': num_clusters}

        scores = NDFS.ndfs(**params)
        scores = (scores * scores).sum(1)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def chi_squared(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        Compute chi-squared stats between each non-negative feature (e.g., booleans or frequencies) and class (i.e.,
        only in a classification scenario).
        The null hypothesis for chi2 test is that "two categorical variables are independent".
        So a higher value of chi2 statistic means "two categorical variables are dependent" and more useful for
        classification.
        The higher the score, the more important the feature is.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
            https://ondata.blog/articles/implications-scikit-learn-21455/
            https://stackoverflow.com/questions/49847493/
                using-chi2-test-for-feature-selection-with-continuous-features-scikit-learn

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            Chi2 statistics for each feature - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        # The Chi-square need all the features to be non-negative
        # FIXME: other ideas?
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        scores, pvalues = chi2(X=X_scaled, y=y)
        # FIXME: use also the pvalues

        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def low_variance(X: pd.DataFrame, thr: float) -> (pd.Series, pd.Series):
        """
        Feature selector that removes all low-variance features.
        This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be
        used for unsupervised learning.
        The threshold has to be interpreted as follows: if you want to remove all features that have a value that is
        repeated in more than 80% of the samples, than thr=0.8.
        Note that this method selects the features and doesn't score them.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

        Args:
            X: feature table
            thr: Features with a variance lower than this threshold will be removed.
                 If Variance Threshold = 0 (Remove Constant Features)
                 If Variance Threshold > 0 (Remove Quasi-Constant Features)

        Returns:
            Features with high-variance - pd.Series
            Rank scores: 1 is assigned to selected features and NaN to the remaining ones - pd.Series
        """

        X = X.drop(['y', 'class'], axis=1)

        selector = VarianceThreshold(threshold=thr * (1 - thr))
        selector.fit(X)
        mask = selector.get_support()
        scores = pd.Series(mask, index=X.columns, dtype=int)
        rank = pd.Series(index=X.columns)
        rank.loc[mask] = 1

        return scores, rank

    @staticmethod
    def t_score(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        This function calculates t_score for each feature, where t_score is only used for binary problem
        t_score = |mean1-mean2|/sqrt(((std1^2)/n1)+((std2^2)/n2))).
        The higher the score, the more important the feature is.

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            Tscore for each feature - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)
        if len(np.unique(y.values)) != 2:
            raise ValueError("T-score works only for binary classification problems!")

        scores = t_score.t_score(X=X.values, y=y.values)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def f_score(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        This function implements the anova f_value feature selection (existing method for classification in SKlearn),
        where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2)).
        The higher the score, the more important the feature is.

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            Fscore for each feature - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        scores, pvalues = f_classif(X=X.values, y=y.values)
        # FIXME: use also the pvalues

        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(ascending=False, method='min')

        return scores, rank

    @staticmethod
    def gini(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        This function implements the gini index feature selection.
        The smaller the gini index, the more important the feature is.

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            Gini score for each feature - pd.Series
            Rank scores - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        scores = gini_index.gini_index(X=X.values, y=y.values)
        scores = pd.Series(scores, index=X.columns)
        rank = scores.rank(method='min')

        return scores, rank

    @staticmethod
    def cfs(X: pd.DataFrame, label: str) -> (pd.Series, pd.Series):
        """
        This function uses a correlation based heuristic to evaluate the worth of features which is called CFS.
        Note that this method selects the features and doesn't score them.

        References:
            Zhao, Zheng et al. "Advancing Feature Selection Research - ASU Feature Selection Repository" 2010

        Args:
            X: feature table
            label: the name of the label column in the feature table

        Returns:
            Features selected with CFS - pd.Series
            Rank scores: 1 is assigned to selected features and NaN to the remaining ones - pd.Series
        """
        y = X[label]
        X = X.drop(['y', 'class', label], axis=1)

        feat_idx = CFS.cfs(X=X.values, y=y.values)

        out_scores = pd.Series(data=np.zeros(len(X.columns)), index=X.columns)
        out_scores.iloc[feat_idx] = 1
        rank = pd.Series(index=X.columns)
        rank.iloc[feat_idx] = 1

        return out_scores, rank

    def get_scores(self, method: str, output: str = 'score', topk: int = None, **kwargs) -> dict:
        """
        Compute the feature scores/ranks (depending on output) with the specified feature selection method.
        If specified the scores/ranks of the topk features are returned.

        Args:
            method: the name of the feature selection method
            output: whether to return the scores or the ranks of the features
            topk: whether to select the topk features
            **kwargs: method-specific params

        Returns:
            Feature scores/ranks for each timestamp computed with the specified feature selection method - dict
        """

        # Concatenate the feature tables of all the stocks
        feat_dict = {}
        for stock in self.feat_dict:
            stock_feat_dict = self.feat_dict[stock]
            for ts in stock_feat_dict:
                feat_table = stock_feat_dict[ts]
                if ts not in feat_dict:
                    feat_dict[ts] = feat_table
                else:
                    feat_dict[ts] = pd.concat((feat_dict[ts], feat_table))

        if topk is not None:
            if method in self.methods_with_n_feat:
                kwargs['num_features'] = topk

        feat_ranks = {}
        for ts in feat_dict:

            if method == 'mi_regr':
                scores, rank = self.mutual_info_regr(X=feat_dict[ts], label='y')

            elif method == 'mi_class':
                scores, rank = self.mutual_info_class(X=feat_dict[ts], label='class')

            elif method == 'class_importances':
                scores, rank = self.feat_importances(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'regr_importances':
                scores, rank = self.feat_importances(X=feat_dict[ts], label='y', **kwargs)

            elif method == 'class_sfs':
                scores, rank = self.sfs(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'regr_sfs':
                scores, rank = self.sfs(X=feat_dict[ts], label='y', **kwargs)

            elif method == 'fisher':
                scores, rank = self.fisher_score(X=feat_dict[ts], label='class')

            elif method == 'laplace':
                scores, rank = self.laplace_score(X=feat_dict[ts])

            elif method == 'spec':
                scores, rank = self.spec(X=feat_dict[ts], **kwargs)

            elif method == 'trace_ratio':
                scores, rank = self.trace_ratio(X=feat_dict[ts], label='class', **kwargs)

            elif method in ['mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr']:
                scores, rank = self.mi_family(X=feat_dict[ts], label='class', mi_method=method, **kwargs)

            elif method == 'rfs':
                scores, rank = self.rfs(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'mcfs':
                scores, rank = self.mcfs(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'udfs':
                scores, rank = self.udfs(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'ndfs':
                scores, rank = self.ndfs(X=feat_dict[ts], label='class', **kwargs)

            elif method == 'chi2':
                scores, rank = self.chi_squared(X=feat_dict[ts], label='class')

            elif method == 'low_variance':
                scores, rank = self.low_variance(X=feat_dict[ts], **kwargs)

            elif method == 't-score':
                scores, rank = self.t_score(X=feat_dict[ts], label='class')

            elif method == 'f-score':
                scores, rank = self.f_score(X=feat_dict[ts], label='class')

            elif method == 'gini':
                scores, rank = self.gini(X=feat_dict[ts], label='class')

            elif method == 'cfs':
                scores, rank = self.cfs(X=feat_dict[ts], label='class')

            else:
                raise ValueError("No method found!")

            if output == 'score':
                target = scores
            elif output == 'rank':
                target = rank
            else:
                raise ValueError("Wrong output: only 'score' or 'rank' are available!")

            if topk is not None:
                target = select_topk(target, rank, topk)

            feat_ranks[ts] = target

        return feat_ranks

