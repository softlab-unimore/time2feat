from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tsfresh import feature_extraction


def get_transformer(transform_type: str):
    if transform_type == 'std':
        transformer = StandardScaler()
    elif transform_type == 'minmax':
        transformer = MinMaxScaler()
    elif transform_type == 'robust':
        transformer = RobustScaler()
    else:
        raise ValueError('Select the wrong transformer: ', transform_type)

    return transformer


def apply_transformation(x_train, x_test, transform_type: str):
    # np.seterr(divide='ignore', invalid='ignore')
    transformer = get_transformer(transform_type)
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    return x_train, x_test


def create_fc_parameter(features: list):
    single_features = [x.split('single__', 1)[1] for x in features if x.startswith('single__')]
    pair_features = [x for x in features if x.startswith('pair__')]

    single_features = feature_extraction.settings.from_columns(single_features)
    features_dict = {'single': single_features, 'pair': pair_features}
    return features_dict
