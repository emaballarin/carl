"""This module implements some utility functions."""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
from .classifier import ClassifierRatio
import matplotlib.pyplot as plt


def plot_score_(ratio, axis, reals, labels):
    cal_num, cal_den = (ratio.classifier_.calibrators_[0].calibrator0,
                        ratio.classifier_.calibrators_[0].calibrator1)
    axis.plot(reals, cal_num.pdf(reals.reshape(-1, 1)),
              label="p(s_num), num~{0}".format(labels[0]))
    axis.plot(reals, cal_den.pdf(reals.reshape(-1, 1)),
              label="p(s_den), den~{0}".format(labels[1]))
    axis.legend(frameon=False)


def plot_scores(classifier_ratios, num_den_labels=None, save_file=None):
    """
    Plot score plots for a list of classifier ratios,
    useful to check training and calibration quality

    Parameters
    ----------
    * `classifier_ratios` [list of `ClassifierRatio`]:
        List of ClassifierRatio to plot.

    * `num_den_labels` [list of (num_label, den_label) tuples]:
        List of numeratod and denominator labels for each
        ratio in `classifier_ratios`

    """

    if len(classifier_ratios) == 0:
        raise ValueError
    for classifier in classifier_ratios:
        if not isinstance(classifier, ClassifierRatio):
            raise ValueError

    num_ratios = len(classifier_ratios)
    if num_den_labels is None:
        num_den_labels = [(l, 0) for l in
                          range(len(classifier_ratios))]
    else:
        # Ensure format for numbers
        def rounds(n): return map(lambda x: round(x, 1), n)
        num_den_labels = [(rounds(x[0]), rounds(x[1])) for x in num_den_labels]

    reals = np.linspace(0, 1)

    if num_ratios <= 3:
        f, axarr = plt.subplots(1, num_ratios, sharex=True, sharey=True,
                                figsize=(12, 5))
        for k, (ratio, pos) in enumerate(zip(classifier_ratios,
                                             num_den_labels)):
            plot_score_(ratio, axarr[k], reals, labels=pos)
    else:
        f, axarr = plt.subplots((num_ratios // 3) + (num_ratios % 3 != 0), 3,
                                sharex=True, sharey=True)
        for k, (ratio, pos) in enumerate(zip(classifier_ratios, 
                                             num_den_labels)):
            plot_score_(ratio, axarr[k // 3, k % 3], reals, labels=pos)
    f.subplots_adjust(hspace=0.05, wspace=0.05)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()
        plt.clf()
