import os
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from datetime import datetime

import pandas as pd

from demo import pipeline

DATASETS_UCR = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket', 'Epilepsy', 'ERing',
    'EthanolConcentration', 'HandMovementDirection', 'Handwriting', 'Libras', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary', 'LSST', 'PenDigits', 'PhonemeSpectra'
]

SELECTED_DATESET_UCR = ['Libras', 'BasicMotions', 'UWaveGestureLibrary', 'Handwriting', 'SelfRegulationsSCP1',
                        'Cricket']
DELETE_DATASET_UCR = ['AtrialFibrillation', 'StandWalkJump', 'HandMovementDirection', 'SelfRegulationSCP2',
                      'PhonemeSpectra', 'LSST']
# DATASETS_UCR = ['BasicMotions']

RANKING_MAP = {
    # sparse learning based
    'SL': ['udfs', 'rfs', 'mcfs', 'ndfs'],
    # information theoretical based
    'IT': ['mim', 'mifs', 'mrmr', 'cife', 'jmi', 'cmim', 'icap', 'disr'],
    # similarity based
    'Sim': ['fisher_score', 'laplace_score', 'trace_ratio100', 'trace_ratio'],
    # statistical based
    'Stat': ['gini', 'cfs'],
    # Sklearn
    'SK': ['anova']
}

ENSEMBLE_RANKING = {
    'ALL': [val for arr in RANKING_MAP.values() for val in arr],
    'SimSK': ['anova', 'fisher_score', 'laplace_score', 'trace_ratio100', 'trace_ratio'],
    'Top3': ['anova', 'fisher_score', 'trace_ratio100'],
    'Top5': ['anova', 'fisher_score', 'trace_ratio100', 'trace_ratio', 'gini'],
}

ENSEMBLE = [
    'average',
    'reciprocal_rank_fusion',
    'condorcet_fuse',
    'rank_biased_centroid',
    'inverse_square_rank',
    'combsum',
    'combmnz'
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
    results_name = os.path.basename(os.path.dirname(files[0])) + f'_s{int(train_size * 100)}.csv'
    results_path = os.path.join(output_dir, results_name)
    # Initialize a dictionary to store the results
    results = {}

    # Create a list of all ranking methods from the RANKING_MAP dictionary
    ranking_methods = [val for arr in RANKING_MAP.values() for val in arr]

    # Perform time2feat pipeline for each ranking method individually
    for ranking in ranking_methods:
        print(f'\n{ranking}')
        t1 = datetime.now()
        try:
            res, _ = pipeline(
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
        except:
            traceback.print_exc()
            res = {}
        t12 = (datetime.now() - t1)
        print(f'{ranking}: {int(t12.total_seconds() / 60)} min\n')

        # Save the current results to a CSV file
        results[ranking] = res
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    # Perform time2feat pipeline for each ranking method individually w/o top-k search and PFA
    for ranking in ['mrmr', 'cife', 'cmim', 'icap', 'cfs']:
        print(f'\n{ranking}')
        t1 = datetime.now()
        try:
            res, _ = pipeline(
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
        except:
            traceback.print_exc()
            res = {}
        t12 = (datetime.now() - t1)
        print(f'{ranking} w/o S&PFA: {int(t12.total_seconds() / 60)} min\n')

        # Save the current results to a CSV file
        results[f'{ranking} w/o S&PFA'] = res
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    # Perform time2feat pipeline based on ranking method groups and all ensemble methods
    for ensemble in ENSEMBLE:
        for k, ranking in ENSEMBLE_RANKING.items():
            if len(ranking) < 2:
                continue

            print(f'\n{ensemble} {ranking}')
            t1 = datetime.now()
            try:
                res, _ = pipeline(
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
            except:
                traceback.print_exc()
                res = {}

            t12 = (datetime.now() - t1)
            print(f'{ensemble} {k}: {int(t12.total_seconds() / 60)} min\n')

            # Save the current results to a CSV file
            results[f'{ensemble}{k}'] = res
            pd.DataFrame(results).T.to_csv(results_path, index=True)

    return results


def debug_ranking_pipeline(
        files: list,
        train_size: float,
        output_dir: str,
        checkpoint_dir: str = './checkpoint',
        seed: int = None,
):
    # Create a results file name based on the base name of the directory of the first file and the train size
    results_name = os.path.basename(os.path.dirname(files[0])) + f'_s{int(train_size * 100)}.csv'
    results = {}

    for ranker in ['anova', 'fisher_score']:
        print(f'\n{ranker}\n')
        res, df_debug = pipeline(
            files=files,
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=[ranker],
            ensemble_type=None,  # 'condorcet_fuse',
            train_type='random',
            train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir=checkpoint_dir,
            random_seed=seed
        )
        results[ranker] = res
        debug_path = os.path.join(output_dir, f"debug_{ranker}_{results_name}")
        df_debug.to_csv(debug_path, index=False)

    results_path = os.path.join(output_dir, f"test_{results_name}")
    pd.DataFrame(results).T.to_csv(results_path, index=True)


def main():
    data_dir, output_dir, checkpoint_dir, seed = parse_params()

    for dataset in SELECTED_DATESET_UCR:
        print(f'\n{dataset}')

        if not os.path.isdir(os.path.join(data_dir, dataset)):
            print(f'{dataset} does not exist')
            continue

        files = [
            os.path.join(data_dir, dataset, f'{dataset}_TEST.ts'),
            os.path.join(data_dir, dataset, f'{dataset}_TRAIN.ts'),
        ]

        for train_size in [0.2, 0.3, 0.5]:
            # _ = test_feature_selection_pipeline(files, train_size, output_dir, checkpoint_dir, seed)
            debug_ranking_pipeline(files, train_size, output_dir, checkpoint_dir, seed)


if __name__ == '__main__':
    main()
    print('Hello World!')
