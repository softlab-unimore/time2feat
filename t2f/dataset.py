import os
import numpy as np
import pandas as pd

from .reader import read_ts


def read_train_test(path: str, is_ucr: bool = False):
    if is_ucr:
        # UCR time series classification format
        # Extract directory tag from folder name
        tag = os.path.basename(path)
        if not tag:
            # Possible exception for empty basename
            path, _ = os.path.split(path)
            tag = os.path.basename(path)

        # Define train and test path
        train_file = os.path.join(path, '{}_TRAIN.ts'.format(tag))
        test_file = os.path.join(path, '{}_TEST.ts'.format(tag))

        ts_train_list, y_train = read_ts(train_file)
        ts_test_list, y_test = read_ts(test_file)
    else:
        # The other format, the industrial use case
        num_train = 1
        ts_train_list, y_train, ts_test_list, y_test = [], [], [], []
        classes = sorted(os.listdir(path))
        for i, class_dir in enumerate(classes):
            class_dir = os.path.join(path, class_dir)
            files = sorted(os.listdir(class_dir))
            # Read time series and expand class label for each series
            ts_list = [pd.read_csv(os.path.join(class_dir, name)) for name in files if name.endswith('.csv')]
            y = [i] * len(ts_list)

            # Save results
            ts_train_list += ts_list[:num_train]
            y_train += y[:num_train]
            ts_test_list += ts_list[num_train:]
            y_test += y[num_train:]

    return ts_train_list, y_train, ts_test_list, y_test


def update_result(df: pd.DataFrame, filename: str, overwrite: bool = False):
    if not overwrite and os.path.isfile(filename):
        print('Update current file')
        df_past = pd.read_csv(filename)
        df = pd.concat([df_past, df], ignore_index=True)

    df.to_csv(filename, index=False)


def read_complete_ucr_dataset(path: str):
    ts_train_list, y_train, ts_test_list, y_test = read_train_test(path=path, is_ucr=True)

    x_train = [df.values for df in ts_train_list]
    x_test = [df.values for df in ts_test_list]
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x = np.vstack([x_train, x_test])
    y = y_train + y_test

    return x, y
