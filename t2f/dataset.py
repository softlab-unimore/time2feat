import os
import numpy as np
import pandas as pd

from .reader import load_from_tsfile_to_dataframe


def read_mts(path: str):
    """ Wrapper for sktime load_from_tsfile_to_dataframe function """
    # Read multivariate time series (mts)
    df, y = load_from_tsfile_to_dataframe(path)

    # class_names = pd.get_dummies(y).sort_index(axis=1).columns
    # y = pd.get_dummies(y).sort_index(axis=1).apply(np.argmax, axis=1).to_list()
    # y = [int(x) for x in y]

    # Extract list of mts array
    df = df.applymap(lambda val: val.to_list())
    ts_list = df.apply(lambda row: np.array(row.to_list()).T, axis=1).to_list()

    # Check mts consistency
    assert len(y) == len(ts_list), 'X and Y have different size'
    cond = [len(ts.shape) == 2 for ts in ts_list]
    assert np.all(cond), 'Error in time series format, shape must be 2d'
    return ts_list, list(y)


def read_ucr_train_test(path: str):
    """ Read train and test ucr ts format """
    # Extract directory tag from folder name
    tag = os.path.basename(path)
    if not tag:
        # Possible exception for empty basename
        path, _ = os.path.split(path)
        tag = os.path.basename(path)

    # Define train and test path
    train_file = os.path.join(path, '{}_TRAIN.ts'.format(tag))
    test_file = os.path.join(path, '{}_TEST.ts'.format(tag))

    # Read mts
    ts_train_list, y_train = read_mts(train_file)
    ts_test_list, y_test = read_mts(test_file)

    return ts_train_list, y_train, ts_test_list, y_test


def read_ucr_dataset(path: str):
    """ Read ucr dataset by concatenating train and test files """
    # Read ucr train and test mts
    ts_train_list, y_train, ts_test_list, y_test = read_ucr_train_test(path=path)

    # Concatenate mts and labels
    ts_list = np.array(ts_train_list + ts_test_list)
    y = np.array(y_train + y_test)

    # Transform into numeric form y array
    # y = pd.get_dummies(y).sort_index(axis=1).apply(np.argmax, axis=1).values.astype('int')

    return ts_list, y


def update_result(df: pd.DataFrame, filename: str, overwrite: bool = False):
    if not overwrite and os.path.isfile(filename):
        print('Update current file')
        df_past = pd.read_csv(filename)
        df = pd.concat([df_past, df], ignore_index=True)

    df.to_csv(filename, index=False)
