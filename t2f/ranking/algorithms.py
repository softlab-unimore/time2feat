# https://jundongl.github.io/scikit-feature/algorithms.html

# ToDo FDB: https://jundongl.github.io/scikit-feature/algorithms.html

# def spec(X: pd.DataFrame, style: int) -> (pd.Series, pd.Series):
#     """
#     Compute SPEC scores.
#     If style = -1 or 0, the higher the score, the more important the feature is.
#     If style != -1 and 0, the lower the score, the more important the feature is.
#
#     References:
#         Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
#
#     Args:
#         X: feature table
#         style: the type of ranking method to use
#             style == -1, the first feature ranking function, use all eigenvalues
#             style == 0, the second feature ranking function, use all except the 1st eigenvalue
#             style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
#
#     Returns:
#         The SPEC scores - pd.Series
#         Rank scores - pd.Series
#     """
#     X = X.drop(['y', 'class'], axis=1)
#
#     # Check for abnormal values
#     # Remove the features with an absolute mean greater than 1000
#     # If this is not done the SPEC computation will raise an error saying that "eigenvalues did not converge"
#     # FIXME: other ideas?
#     X_avgs = X.mean(0).abs()
#     clean_feat = X.columns[X_avgs < 1000]
#     X_clean = X[clean_feat]
#
#     kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
#     W = construct_W.construct_W(X_clean.values, **kwargs_W)
#     scores = SPEC.spec(X=X_clean.values, W=W, style=style)
#
#     out_scores = pd.Series(index=X.columns)
#     out_scores.loc[X_clean.columns] = scores
#
#     # If style = -1 or 0, the higher the score, the more important the feature is
#     if style == -1 or style == 0:
#         rank = out_scores.rank(ascending=False, method='min')
#     # If style != -1 and 0, the lower the score, the more important the feature is
#     elif style != -1 and style != 0:
#         rank = out_scores.rank(method='min')
#     else:
#         raise ValueError("Wrong style value!")
#
#     return out_scores, rank

# ToDo FDB: Sklearn

# def mutual_info_regr(df: pd.DataFrame, y: list) -> pd.Series:
#     """
#     Estimate mutual information for a continuous target variable (i.e., only in a regression scenario).
#     Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
#     between the variables. It is equal to zero if and only if two random variables are independent, and higher
#     values mean higher dependency.
#     The higher the score, the more important the feature is.
#
#     References:
#         https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
#
#     Args:
#         X: feature table
#         label: the name of the label column in the feature table
#
#     Returns:
#         The mutual information scores for each feature - pd.Series
#         Rank scores - pd.Series
#     """
#
#     y = X[label]
#     X = X.drop(['y', 'class'], axis=1)
#
#     scores = mutual_info_regression(X=X, y=y, random_state=42)
#     scores = pd.Series(scores, index=X.columns)
#     rank = scores.rank(ascending=False, method='min')
#
#     return scores, rank


# def mutual_info_class(df: pd.DataFrame, y: list) -> pd.Series:
#     """
#     Estimate mutual information for a discrete target variable (i.e., only in a classification scenario).
#     Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
#     between the variables. It is equal to zero if and only if two random variables are independent, and higher
#     values mean higher dependency.
#     The higher the score, the more important the feature is.
#
#     References:
#         https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
#
#     Args:
#         X: feature table
#         label: the name of the label column in the feature table
#
#     Returns:
#         The mutual information scores for each feature - pd.Series
#         Rank scores - pd.Series
#     """
#
#     y = X[label]
#     X = X.drop(['y', 'class', label], axis=1)
#
#     scores = mutual_info_classif(X=X, y=y, random_state=42)
#     scores = pd.Series(scores, index=X.columns)
#     rank = scores.rank(ascending=False, method='min')
#
#     return scores, rank


# def feat_importances(X: pd.DataFrame, label: str, model_name: str, model_params: dict = None,
#                      norm_by_var: bool = False) -> (pd.Series, pd.Series):
#     """
#     Compute feature scores by training an intrinsically interpretable estimator and by extracting from it some
#     feature importance scores (which are stored in a specific attribute, such as coef_ or feature_importances_
#     depending on the model).
#     The higher the absolute score, the more important the feature is.
#
#     References:
#         https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
#
#     Args:
#         X: feature table
#         label: the name of the label column in the feature table
#         model_name: the name of the model used for extracting feature importances
#         model_params: the collection of model params
#         norm_by_var: whether to normalize importance scores by feature variance
#
#     Returns:
#         Feature importances - pd.Series
#         Rank scores - pd.Series
#     """
#     y = X[label]
#     X = X.drop(['y', 'class', label], axis=1)
#
#     model_item = FeatureSelector.models[model_name]
#     model_obj = model_item['name']
#     if model_params is None:
#         if 'params' not in model_item:
#             raise ValueError("Model configuration not provided!")
#         model_params = model_item['params']
#
#     model = model_obj(**model_params)
#     model.fit(X, y)
#
#     if hasattr(model, 'coef_'):
#         scores = np.reshape(model.coef_, -1)
#     elif hasattr(model, 'feature_importances_'):
#         scores = model.feature_importances_
#     else:
#         raise NotImplementedError()
#
#     if norm_by_var is True:
#         scores *= X.std(axis=0)
#     scores = pd.Series(scores, index=X.columns)
#     rank = scores.abs().rank(ascending=False, method='min')
#
#     return scores, rank


# def sfs(X: pd.DataFrame, label: str, model_name: str, num_features: Union[int, float] = None,
#         direction: str = 'forward', model_params: dict = None) -> (pd.Series, pd.Series):
#     """
#     Sequential Feature Selection.
#     This Sequential Feature Selector adds (forward selection) or removes (backward selection) features to form a
#     feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove
#     based on the cross-validation score of an estimator.
#     Note that this method selects the features and doesn't score them.
#
#     References:
#         https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
#
#     Args:
#         X: feature table
#         label: the name of the label column in the feature table
#         model_name: the name of the model used for selecting the features
#         num_features: number of features to select
#                       If None, half of the features are selected.
#                       If integer, the parameter is the absolute number of features to select.
#                       If float between 0 and 1, it is the fraction of features to select.
#         direction: whether to perform forward selection or backward selection.
#         model_params: the collection of model params
#
#     Returns:
#         Feature selected by SFS to the provided model - pd.Series
#         Rank scores: 1 is assigned to selected features and NaN to the remaining ones - pd.Series
#     """
#     y = X[label]
#     X = X.drop(['y', 'class', label], axis=1)
#
#     model_item = FeatureSelector.models[model_name]
#     model_obj = model_item['name']
#     if model_params is None:
#         if 'params' not in model_item:
#             raise ValueError("Model configuration not provided!")
#         model_params = model_item['params']
#
#     model = model_obj(**model_params)
#     sfs = SequentialFeatureSelector(
#         estimator=model, n_features_to_select=num_features, direction=direction
#     )
#     sfs.fit(X, y)
#     mask = sfs.get_support()
#     scores = pd.Series(mask.astype(int), index=X.columns)
#     rank = pd.Series(index=X.columns)
#     rank.loc[mask] = 1
#
#     return scores, rank

# def chi_squared(df: pd.DataFrame, y: list) -> pd.Series:
#     """ Compute chi-squared stats between each non-negative feature (e.g., booleans or frequencies) and class. """
#     y = X[label]
#     X = X.drop(['y', 'class', label], axis=1)
#
#     # The Chi-square need all the features to be non-negative
#     # FIXME: other ideas?
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     scores, pvalues = chi2(X=X_scaled, y=y)
#     # FIXME: use also the pvalues
#
#     scores = pd.Series(scores, index=X.columns)
#     rank = scores.rank(ascending=False, method='min')
#
#     return scores, rank


# ToDo FDB: unsupervised algorithm

# def ndfs(df: pd.DataFrame, y: list, alpha: float = 1, beta: float = 1, gamma: float = 10e8) -> pd.Series:
#     """ Unsupervised feature selection using nonnegative spectral analysis. """
#     if label is None and num_clusters is None:
#         raise ValueError("Provide one between label and num_clusters!")
#
#     if num_clusters is None:
#         y = X[label]
#         num_clusters = len(np.unique(y.values))
#         X = X.drop(['y', 'class', label], axis=1)
#         Y = construct_label_matrix(y.values.astype(int))
#         T = np.dot(Y.transpose(), Y)
#         F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
#         F = F + 0.02 * np.ones((len(X), num_clusters))
#         params = {'X': X.values, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'F0': F, 'n_clusters': num_clusters}
#     else:
#         X = X.drop(['y', 'class'], axis=1)
#         params = {'X': X.values, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'n_clusters': num_clusters}
#
#     scores = NDFS.ndfs(**params)
#     scores = (scores * scores).sum(1)
#     scores = pd.Series(scores, index=X.columns)
#     rank = scores.rank(ascending=False, method='min')
#
#     return scores, rank
