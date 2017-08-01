"""This module implements regressor-based density ratio estimation."""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_random_state

from .base import DensityRatioMixin
from ..learning import as_classifier


class RegressorRatio(BaseEstimator, DensityRatioMixin):
    """Regressor-based density ratio estimator.
    
    This class approximates a density ratio `r(x) = p0(x) / p1(x)` with a regressor
    directly trained on the density ratio. It cannot be used in
    the likelihood-free setup.
    """

    def __init__(self, base_estimator, random_state=None):
        """Constructor.

        Parameters
        ----------
        * `base_estimator` [`BaseEstimator`]:
            A scikit-learn regressor.

        * `random_state` [integer or RandomState object]:
            The random seed.
        """
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X=None, y=None, sample_weight=None, **kwargs):
        """Fit the density ratio estimator.

        The density ratio estimator `r(x) = p0(x) / p1(x)` can be fit from data, using `fit(X, y)`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features), optional]:
            Training data.

        * `y` [array-like, shape=(n_samples,), optional]:
            Density ratio r(x) = p0(x) / p1(x).

        * `sample_weight` [array-like, shape=(n_samples,), optional]:
            The sample weights.

        Returns
        -------
        * `self` [object]:
            `self`.
        """

        # Build training data
        rng = check_random_state(self.random_state)
        
        if X is None or y is None:
            raise ValueError

        # Fit base estimator
        clf = clone(self.base_estimator)

        if sample_weight is None:
            self.regressor_ = clf.fit(X, y)
        else:
            self.regressor_ = clf.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X, log=False, **kwargs):
        """Predict the density ratio `r(x_i)` for all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `log` [boolean, default=False]:
            If true, return the log-ratio `log r(x) = log(p0(x)) - log(p1(x))`.

        Returns
        -------
        * `r` [array, shape=(n_samples,)]:
            The predicted ratio `r(X)`.
        """
        
        p = self.regressor_.predict(X)

        if log:
            return np.log(p[:, 0])
        else:
            return p[:]
