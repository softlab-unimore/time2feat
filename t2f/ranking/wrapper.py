from typing import Dict, List, Literal, Optional, Callable

import numpy as np
import pandas as pd

from .baseline import anova, fisher_score, laplace_score, trace_ratio, trace_ratio100, mim, mifs, mrmr, cife, jmi, \
    cmim, icap, disr, rfs, mcfs, udfs, ndfs, gini, cfs
from .ensemble import average, reciprocal_rank_fusion, condorcet_fuse, rank_biased_centroid, inverse_square_rank
from ..selection.PFA import pfa_scoring


class Ranker(object):
    """
    A class to rank features based on different ranking methods and ensemble them.

    Attributes:
        ranking_type: A list of ranking method names to use.
        ensemble_type: The ensemble method to combine rankings. If only one ranking method is provided,
                       no ensemble method is used.
    """

    RANKER_MAPPING: Dict[str, Callable[[pd.DataFrame, np.ndarray], pd.Series]] = {
        'anova': anova,
        'fisher_score': fisher_score,
        'laplace_score': laplace_score,
        'trace_ratio100': trace_ratio100,
        'trace_ratio': trace_ratio,
        'mim': mim,
        'mifs': mifs,
        'mrmr': mrmr,
        'cife': cife,
        'jmi': jmi,
        'cmim': cmim,
        'icap': icap,
        'disr': disr,
        'rfs': rfs,
        'mcfs': mcfs,
        'udfs': udfs,
        'ndfs': ndfs,
        'gini': gini,
        'cfs': cfs,
    }

    # ToDo GF: insert other approaches
    ENSEMBLE_MAPPING: Dict[str, Callable[[List[pd.Series]], pd.Series]] = {
        'average': average,
        'reciprocal_rank_fusion': reciprocal_rank_fusion,
        'condorcet_fuse': condorcet_fuse,
        'rank_biased_centroid': rank_biased_centroid,
        'inverse_square_rank': inverse_square_rank
    }

    def __init__(
            self,
            ranking_type: List[Literal['anova']],
            ensemble_type=Optional[Literal['average']]
    ):
        """
        Initializes the Ranker object with specified ranking and ensemble methods.
        """
        self.ranking_type = ranking_type  # Store the ranking methods
        self.ensemble_type = ensemble_type  # Store the ensemble method

        self.with_ensemble = len(self.ranking_type) > 1  # Determine if ensemble is necessary

        valid_rankers = set(self.RANKER_MAPPING.keys())
        if not all(r in valid_rankers for r in self.ranking_type):
            raise ValueError(f"Invalid ranking type. Valid options are: {valid_rankers}")

        if self.with_ensemble and self.ensemble_type not in self.ENSEMBLE_MAPPING:
            raise ValueError(f"Invalid ensemble type. Valid options are: {list(self.ENSEMBLE_MAPPING.keys())}")

        self.rankers = [self.RANKER_MAPPING[x] for x in self.ranking_type]  # Create list of ranker functions
        self.ensembler = self.ENSEMBLE_MAPPING.get(self.ensemble_type, None)  # Get the ensembler function

    def ranking(self, df: pd.DataFrame, y: list, top_k: int) -> list:
        # Identify constant columns to avoid including them in feature selection
        constant_columns = df.columns[df.nunique() <= 1].tolist()
        df = df.drop(constant_columns, axis=1)  # Remove constant columns from the DataFrame

        ranks = []  # List to store ranks from each ranker
        for i, ranker in enumerate(self.rankers):
            rank = ranker(df, np.array(y))  # Get the rank from each ranker
            rank.name = self.ranking_type[i]  # Name the rank series for identification
            ranks.append(rank)  # Append the rank to the list

        # Combine ranks using the ensemble method if applicable
        if self.with_ensemble:
            rank = self.ensembler(ranks)  # Apply ensemble method
        else:
            rank = ranks[0]  # Use the rank from the first ranker

        top_feats = rank.index.values[:top_k]  # Select the top_k features based on rank
        # Apply PFA to further select features that retain most information
        top_features, _ = pfa_scoring(df[top_feats], 0.9)

        return top_features  # Return the list of top features
