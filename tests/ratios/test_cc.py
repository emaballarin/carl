# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises

from carl.distributions import Normal
from carl.ratios import CalibratedClassifierRatio

from sklearn.linear_model import ElasticNetCV
from sklearn.naive_bayes import GaussianNB


def check_calibrated_classifier_ratio(clf, calibration, cv):
    # Passing distributions directly
    p0 = Normal(mu=0.0)
    p1 = Normal(mu=0.1)

    ratio = CalibratedClassifierRatio(base_estimator=clf,
                                      calibration=calibration, cv=cv)
    ratio.fit(numerator=p0, denominator=p1, n_samples=10000)

    reals = np.linspace(-1, 1, num=100).reshape(-1, 1)

    assert np.mean(np.abs(p0.pdf(reals) / p1.pdf(reals) -
                          ratio.predict(reals))) < 0.1
    assert np.mean(np.abs(-p0.nnlf(reals) + p1.nnlf(reals) -
                          ratio.predict(reals, log=True))) < 0.1

    # Passing X, y only
    X = np.vstack((p0.rvs(5000), p1.rvs(5000)))
    y = np.zeros(10000, dtype=np.int)
    y[5000:] = 1

    ratio = CalibratedClassifierRatio(base_estimator=clf,
                                      calibration=calibration, cv=cv)
    ratio.fit(X=X, y=y)

    reals = np.linspace(-1, 1, num=100).reshape(-1, 1)

    assert np.mean(np.abs(p0.pdf(reals) / p1.pdf(reals) -
                          ratio.predict(reals))) < 0.1
    assert np.mean(np.abs(-p0.nnlf(reals) + p1.nnlf(reals) -
                          ratio.predict(reals, log=True))) < 0.1


def test_calibrated_classifier_ratio():
    for clf, calibration, cv in [(ElasticNetCV(), "histogram", 3),
                                 (GaussianNB(), "kde", 3)]:
        yield check_calibrated_classifier_ratio, clf, calibration, cv
