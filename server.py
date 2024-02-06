import os
import argparse
from datetime import datetime

import pandas as pd

from demo import pipeline


DATASETS_UCR = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket', 'Epilepsy', 'ERing',
    'EthanolConcentration', 'HandMovementDirection', 'Handwriting', 'Libras', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary', 'LSST', 'PenDigits', 'PhonemeSpectra'
]
# DATASETS_UCR = ['BasicMotions']

RANKING_MAP = {
    # information theoretical based
    'IT': ['mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr'],
    # similarity based
    'Sim': ['fisher_score', 'laplace_score', 'trace_ratio100', 'trace_ratio'],
    # sparse learning based
    'SL': ['rfs', 'mcfs', 'udfs', 'ndfs'],
    # statistical based
    'Stat': ['gini', 'cfs'],
    # Sklearn
    'SK': ['anova']
}

ENSEMBLE = [
    'average',
    'reciprocal_rank_fusion',
    'condorcet_fuse',
    'rank_biased_centroid',
    'inverse_square_rank',
]


def parse_params():
    """ Parse input parameters. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='the path where the UCR datasets are stored.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='the path where the results are stored.')
    parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str,
                        help='the path where the checkpoint are stored.')
    parser.add_argument('-seed', '--seed', default=4, type=int,
                        help='Random seed for train labels selection')

    args = parser.parse_args()
    assert os.path.isdir(args.dataset), 'UCR path does not exist'
    assert os.path.isdir(args.output), 'Output path is not a dir'

    return args.dataset, args.output, args.checkpoint, args.seed


def test_feature_selection_pipeline(
        files: list,
        train_size: float,
        output_dir: str,
        checkpoint_dir: str = './checkpoint',
        seed: int = None,
):
    # Create a results file name based on the base name of the directory of the first file and the train size
    results_name = os.path.basename(os.path.dirname(files[0])) + f'_s{int(train_size) * 100}.csv'
    results_path = os.path.join(output_dir, results_name)
    # Initialize a dictionary to store the results
    results = {}

    # Create a list of all ranking methods from the RANKING_MAP dictionary
    ranking_methods = [val for arr in RANKING_MAP.values() for val in arr]

    # Perform time2feat pipeline for each ranking method individually
    for ranking in ranking_methods:
        print(f'\n{ranking}')
        t1 = datetime.now()
        res = pipeline(
            files=files,
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=[ranking],
            ensemble_type=None,  # 'condorcet_fuse',
            train_type='random',
            train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir=checkpoint_dir,
            random_seed=seed
        )
        t12 = (datetime.now() - t1)
        print(f'{ranking}: {int(t12.total_seconds() / 60)} min\n')

        # Save the current results to a CSV file
        results[ranking] = res
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    # Perform time2feat pipeline for each ranking method individually w/o top-k search and PFA
    for ranking in ['mrmr', 'cife', 'cmim', 'icap', 'cfs']:
        print(f'\n{ranking}')
        t1 = datetime.now()
        res = pipeline(
            files=files,
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=[ranking],
            ranking_pfa=None,
            ensemble_type=None,  # 'condorcet_fuse',
            search_type=None,
            train_type='random',
            train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir=checkpoint_dir,
            random_seed=seed
        )
        t12 = (datetime.now() - t1)
        print(f'{ranking} w/o S&PFA: {int(t12.total_seconds() / 60)} min\n')

        # Save the current results to a CSV file
        results[ranking] = res
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    # Perform time2feat pipeline with all ranking methods and each ensemble method
    for ensemble in ENSEMBLE:
        print(f'\n{ensemble}')
        t1 = datetime.now()
        res = pipeline(
            files=files,
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=ranking_methods,
            ensemble_type=ensemble,  # 'condorcet_fuse',
            train_type='random',
            train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir=checkpoint_dir
        )
        t12 = (datetime.now() - t1)
        print(f'{ensemble}: {int(t12.total_seconds() / 60)} min\n')

        # Save the current results to a CSV file
        results[ensemble] = res
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    # Perform time2feat pipeline based on ranking method groups and all ensemble methods
    for ensemble in ENSEMBLE:
        for k, ranking in RANKING_MAP.items():
            print(f'\n{ensemble} {ranking}')
            t1 = datetime.now()
            res = pipeline(
                files=files,
                intra_type='tsfresh',
                inter_type='distance',
                transform_type='minmax',
                model_type='Hierarchical',
                ranking_type=ranking,
                ensemble_type=ensemble,  # 'condorcet_fuse',
                train_type='random',
                train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
                batch_size=500,
                p=4,
                checkpoint_dir=checkpoint_dir
            )
            t12 = (datetime.now() - t1)
            print(f'{ensemble} {k}: {int(t12.total_seconds() / 60)} min\n')

            # Save the current results to a CSV file
            results[f'{ensemble}{k}'] = res
            pd.DataFrame(results).T.to_csv(results_path, index=True)

    return results


def main():
    data_dir, output_dir, checkpoint_dir, seed = parse_params()

    for dataset in DATASETS_UCR:
        print(f'\n{dataset}')
        dataset_dir = os.path.join(data_dir, dataset)

        if not os.path.isdir(dataset_dir):
            print('No dataset')
            continue

        files = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x.endswith('.ts')]

        if len(files) != 2:
            print('No train and test files in dataset directory')
            continue

        for train_size in [0.2, 0.3, 0.5]:
            _ = test_feature_selection_pipeline(files, train_size, output_dir, checkpoint_dir, seed)


if __name__ == '__main__':
    main()
    print('Hello World!')
