import os

import pandas as pd
from sklearn.model_selection import ParameterGrid

from main import pipeline

all_datasets = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket',
    'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'HandMovementDirection', 'Handwriting',
    'Libras', 'LSST', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary'
]


def tune():
    exceptions_size = [
        'FaceDetection',
        'FingerMovements',
        'Heartbeat',
        'MotorImagery',
        'NATOPS',
        'PEMS-SF',
        'DuckDuckGeese',
    ]
    exceptions_length = ['CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits', 'InsectWingbeat']
    datasets = [
                   'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket',
                   'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'HandMovementDirection',
                   'Handwriting', 'Libras', 'RacketSports', 'SelfRegulationSCP1',
                   'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary'
               ]
    datasets = datasets + ['LSST', 'PenDigits', 'PhonemeSpectra']
    datasets = datasets * 10
    base_dir = '/export/static/pub/softlab/ucr1/Multivariate_TS/'
    # base_dir = r'C:\Users\delbu\Projects\Dataset\Multivariate_TS'
    # datasets = ['LSST']

    errors = []
    for dataset in datasets:
        data_dir = os.path.join(base_dir, dataset)

        base_params = {
            'data_dir': data_dir,
            'monitor': False,
            'example': False,
            'output_dir': 'output/',
            'is_ucr': True,
            'window_mode': False,
            'with_window_selection': False,
            'window': 300,
            'batch_size': 1000,
            'p': 2,
            'complete': True,
            'k_best': False,
            'auto': True,
            'strategy': 'none',  # 'tsfresh, 'multi, 'sk_base', 'sk_pvalue'
        }
        grid_params = {
            'model_type': ['KMeans'],
            'score_mode': ['domain'],
            'top_k': [100],
            'transform_type': ['std'],
            'partial': [0.1, 0.2, 0.3, 0.4, 0.5],
            'pre_transform': [False],
            'strategy': ['none', 'sk_base']
        }
        output_filename = os.path.join(base_params['output_dir'], 'errors.csv')
        for grid in ParameterGrid(grid_params):
            if grid['strategy'] == 'none' and grid['partial'] != 0.5:
                continue

            print('\n Params: ', grid)
            base_params.update(grid)
            # pipeline(base_params)

            try:
                pipeline(base_params)
            except Exception as e:
                base_params['error'] = str(e)
                errors.append(base_params)
                df_err = pd.DataFrame(errors)
                df_err.to_csv(output_filename, index=False)


if __name__ == '__main__':
    tune()

# grid_params = {
#     'score_mode': ['simple', 'domain'],
#     'top_k': [10, 50, 100, 200],
#     'transform_type': ['std', 'minmax'],
#     'model_type': ['AgglomerativeClustering', 'KMeans', 'SpectralClustering'],
#     'partial': [0.1, 0.2, 0.3, 0.4, 0.5]
# }
