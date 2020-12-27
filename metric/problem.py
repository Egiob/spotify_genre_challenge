import numpy as np
import os
from ot import emd2
from rampwf.score_types import BaseScoreType


class EMDScore(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='emd score', precision=3):
        self.__name__ = name
        self.name = name
        ground_metric_path = \
            os.path.join(os.path.dirname(__file__), "ground_metric.npy")
        self.ground_metric = np.load(ground_metric_path)
        self.ground_metric /= self.ground_metric.max()
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        scores = []

        mask = ~np.any(np.isnan(y_proba), axis=1)
        y_proba = y_proba[mask]
        y_true_proba = y_true_proba[mask]

        for this_y_true, this_y_proba in zip(y_true_proba, y_proba):
            this_y_true_max = this_y_true.max()
            this_y_proba_max = this_y_proba.max()

            # special treatment for the all zero cases
            if (this_y_true_max * this_y_proba_max) == 0:
                if this_y_true_max or this_y_proba_max:
                    scores.append(1.)  # as ground_metric max is 1
                else:
                    scores.append(0.)
                continue

            this_y_true = this_y_true.astype(np.float64) / this_y_true.sum()
            this_y_proba = this_y_proba.astype(np.float64) / this_y_proba.sum()

            score = emd2(this_y_true, this_y_proba, self.ground_metric,
                         numItermax=10e8)
            scores.append(score)

        assert len(scores) == len(y_true_proba)
        assert len(y_proba) == len(y_true_proba)
        return np.mean(scores)
