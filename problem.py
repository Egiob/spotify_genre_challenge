import os

import numpy as np
import pandas as pd
from ot import emd2
import rampwf as rw

from rampwf.score_types import BaseScoreType
from rampwf.prediction_types.base import BasePrediction
from sklearn.model_selection import ShuffleSplit
from rampwf.workflows import SKLearnPipeline
from sklearn.base import is_classifier

import functools
import warnings

class JaccardError(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='jaccard error', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        mask = ~np.any(np.isnan(y_proba), axis=1)

        score = 1 - jaccard_score(y_true_proba[mask],
                                  y_proba[mask],
                                  average='samples')
        return score

class EMDScore(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='emd score', precision=3):
        self.__name__ = name
        self.name = name
        self.ground_metric = np.load('metric/metric_genres_considered.npy')
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

score_types = [
    EMDScore(name='EMD score', precision=5),
    JaccardError(name='jaccard error', precision=5)

]


class _MultiOutputClassification(BasePrediction):
    def __init__(self, n_columns, y_pred=None, y_true=None, n_samples=None):
        self.n_columns = n_columns
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif n_samples is not None:
            if self.n_columns == 0:
                shape = (n_samples)
            else:
                shape = (n_samples, self.n_columns)
            self.y_pred = np.empty(shape, dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Inherits from the base class where the scores are averaged.
        Here, averaged predictions < 0.5 will be set to 0.0 and averaged
        predictions >= 0.5 will be set to 1.0 so that `y_pred` will consist
        only of 0.0s and 1.0s.
        """
        # call the combine from the BasePrediction
        combined_predictions = super(
            _MultiOutputClassification, cls
            ).combine(
                predictions_list=predictions_list,
                index_list=index_list
                )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            combined_predictions.y_pred[
                combined_predictions.y_pred < 0.5] = 0.0
            combined_predictions.y_pred[
                combined_predictions.y_pred >= 0.5] = 1.0

        return combined_predictions


def partial_multioutput(cls=_MultiOutputClassification, **kwds):
    # this class partially inititates _MultiOutputClassification with given
    # keywords
    class _PartialMultiOutputClassification(_MultiOutputClassification):
        __init__ = functools.partialmethod(cls.__init__, **kwds)
    return _PartialMultiOutputClassification


def make_multioutput(n_columns):
    return partial_multioutput(n_columns=n_columns)


class EstimatorGenre(SKLearnPipeline):
    """Choose predict method.

    Parameters
    ----------
    predict_method : {'auto', 'predict', 'predict_proba',
            'decision_function'}, default='auto'
        Prediction method to use. If 'auto', uses 'predict_proba' when
        estimator is a classifier and 'predict' otherwise.
    """
    def __init__(self, predict_method='auto'):
        super().__init__()
        self.predict_method = predict_method

    def test_submission(self, estimator_fitted, X):
        """Predict using a fitted estimator.

        Parameters
        ----------
        estimator_fitted : Estimator object
            A fitted scikit-learn estimator.
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The test data set.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes) or (n_samples)
        """
        methods = ('auto', 'predict', 'predict_proba', 'decision_function')
        n_samples = len(X)
        X = X.reset_index(drop=True)

        # get y corresponding to chosen X
        mask = np.zeros(n_samples, dtype=bool)
        mask[X.index] = True

        if self.predict_method not in methods:
            raise NotImplementedError(f"'method' should be one of: {methods} "
                                      f"Got: {self.predict_method}")
        if self.predict_method == 'auto':
            if is_classifier(estimator_fitted):
                y_pred = estimator_fitted.predict_proba(X)
            else:
                y_pred = estimator_fitted.predict(X)
        elif hasattr(estimator_fitted, self.predict_method):
            # call estimator with the `predict_method`
            est_predict = getattr(estimator_fitted, self.predict_method)
            y_pred = est_predict(X)
        else:
            raise NotImplementedError("Estimator does not support method: "
                                      f"{self.predict_method}.")

        if np.any(np.isnan(y_pred)):
            raise ValueError('NaNs found in the predictions.')

        y_pred_full = \
            np.full(fill_value=np.nan, shape=(mask.size, y_pred.shape[1]))
        y_pred_full[mask] = y_pred
        return y_pred_full

def make_workflow():
    # defines new workflow, where predict instead of predict_proba is called
    return EstimatorGenre(predict_method='predict')

problem_title = 'Spotify genre challenge'
n_genre = 25  # number of genre used in this challenge
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = make_multioutput(n_columns=n_genre)
# An object implementing the workflow
workflow = workflow = make_workflow()


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    
    return cv.split(X, y)


def _read_data(path, f_name_X, f_name_y):
    X_df = pd.read_csv(os.path.join(path, 'data', f_name_X), sep=';').drop(['Unnamed: 0'], axis=1)
    y_array = pd.read_csv(os.path.join(path, 'data', f_name_y), sep=';').drop(['Unnamed: 0'], axis=1).values
    
    return X_df, y_array


def get_train_data(path='.'):
    f_name_X = 'X_train.csv'
    f_name_y = 'y_train.csv'

    return _read_data(path, f_name_X, f_name_y)


def get_test_data(path='.'):
    f_name_X = 'X_test.csv'
    f_name_y = 'y_test.csv'

    return _read_data(path, f_name_X, f_name_y)



def reduce_metric(genres, metric, genre_index, reset_distance=False):
    """
    Args:
        genres (list) : the list of the genres we want to consider
        metric (matrix) : the matrix of the metric
        genre index (dict)
        reset_distance (bool) : whether or not reset the distance to [1, nb_class]
    """
    idx = [genre_index[i] for i in genres]
    new_metric = metric[idx][:,idx]
    if reset_distance:
        new_metric = np.array([new_metric[i].argsort() for i in range(len(new_metric))],dtype=np.float16)
    return new_metric
