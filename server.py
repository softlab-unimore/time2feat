import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import pandas as pd

from demo import pipeline

DATASETS_UCR = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket', 'Epilepsy', 'ERing',
    'EthanolConcentration', 'HandMovementDirection', 'Handwriting', 'Libras', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary',  # 'LSST', 'PenDigits', 'PhonemeSpectra'
]

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

# ENSEMBLE_RANKING = {
#     'ALL': [val for arr in RANKING_MAP.values() for val in arr],
#     'SimSK': ['anova', 'fisher_score', 'laplace_score', 'trace_ratio100', 'trace_ratio'],
#     'Top3': ['anova', 'fisher_score', 'trace_ratio100'],
#     'Top5': ['anova', 'fisher_score', 'trace_ratio100', 'trace_ratio', 'gini'],
# }

ENSEMBLE_RANKING = {
    'all': ['anova', 'cfs', 'fisher_score', 'gini', 'laplace_score', 'trace_ratio', 'trace_ratio100'],
    'top5': ['anova', 'fisher_score', 'laplace_score', 'trace_ratio', 'trace_ratio100'],
    'aft': ['anova', 'fisher_score', 'trace_ratio100'],
    'afl': ['anova', 'fisher_score', 'laplace_score'],
    'aft': ['anova', 'fisher_score', 'trace_ratio'],
    'alt': ['anova', 'laplace_score', 'trace_ratio'],
    'al1': ['anova', 'laplace_score', 'trace_ratio100']
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


def debug_ranking_pipeline(
        files: list,
        train_size: float,
        output_dir: str,
        checkpoint_dir: str = './checkpoint',
        seed: int = None,
        train_real: bool = False,
):
    # Create a results file name based on the base name of the directory of the first file and the train size
    results_name = os.path.basename(os.path.dirname(files[0])) + f'_s{int(train_size * 100)}.csv'
    results = {}

    print('time2feat')
    res, df_debug = pipeline(
        files=files,
        intra_type='tsfresh',
        inter_type='distance',
        transform_type='minmax',
        model_type='Hierarchical',
        ranking_type=['anova'],
        ensemble_type=None,  # 'condorcet_fuse',
        search_type='time2feat',
        train_type='random',
        train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
        batch_size=500,
        p=4,
        checkpoint_dir=checkpoint_dir,
        random_seed=seed,
        train_real=train_real
    )
    results['time2feat'] = res
    debug_path = os.path.join(output_dir, f"debug_time2feat_{results_name}")
    df_debug.to_csv(debug_path, index=False)

    print('Single ranker')
    for ranker in ['anova', 'fisher_score', 'laplace_score', 'trace_ratio100', 'trace_ratio', 'gini', 'cfs']:
        print(f'\n{ranker}\n')
        res, df_debug = pipeline(
            files=files,
            intra_type='tsfresh',
            inter_type='distance',
            transform_type='minmax',
            model_type='Hierarchical',
            ranking_type=[ranker],
            ensemble_type=None,  # 'condorcet_fuse',
            search_type='cv5',
            train_type='random',
            train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
            batch_size=500,
            p=4,
            checkpoint_dir=checkpoint_dir,
            random_seed=seed,
            train_real=train_real
        )
        results[ranker] = res
        debug_path = os.path.join(output_dir, f"debug_{ranker}_{results_name}")
        df_debug.to_csv(debug_path, index=False)

    print('Fusion')
    for ensemble in ENSEMBLE:
        for rsetid, rankers_set in ENSEMBLE_RANKING.items():
            res, df_debug = pipeline(
                files=files,
                intra_type='tsfresh',
                inter_type='distance',
                transform_type='minmax',
                model_type='Hierarchical',
                ranking_type=rankers_set,
                ensemble_type=ensemble,  # 'condorcet_fuse',
                search_type='cv5',
                train_type='random',
                train_size=train_size,  # 0.2, 0.3, 0.4, 0.5
                batch_size=500,
                p=4,
                checkpoint_dir=checkpoint_dir,
                random_seed=seed,
                train_real=train_real
            )
            results[f"{ensemble}-{rsetid}"] = res
            debug_path = os.path.join(output_dir, f"debug_{ensemble}-{rsetid}_{results_name}")
            df_debug.to_csv(debug_path, index=False)

    results_path = os.path.join(output_dir, f"test_{results_name}")
    pd.DataFrame(results).T.to_csv(results_path, index=True)


def main():
    data_dir, output_dir, checkpoint_dir, seed = parse_params()

    for dataset in DATASETS_UCR:
        print(f'\n{dataset}')

        if not os.path.isdir(os.path.join(data_dir, dataset)):
            print(f'{dataset} does not exist')
            continue

        files = [
            os.path.join(data_dir, dataset, f'{dataset}_TEST.ts'),
            os.path.join(data_dir, dataset, f'{dataset}_TRAIN.ts'),
        ]

        for train_size in [0.2]:
            debug_ranking_pipeline(files, train_size, output_dir, checkpoint_dir, seed)


if __name__ == '__main__':
    main()
    print('Hello World!')
