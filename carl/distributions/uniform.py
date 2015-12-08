# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from theano.gof import graph

from . import DistributionMixin


class Uniform(DistributionMixin):
    def __init__(self, random_state=None, low=0.0, high=1.0):
        super(Uniform, self).__init__(random_state=random_state,
                                      low=low, high=high)

        # pdf
        self.pdf_ = T.switch(T.or_(T.lt(self.X, self.low),
                                   T.ge(self.X, self.high)),
                             0., 1. / (self.high - self.low))
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nnlf_ = T.switch(T.or_(T.lt(self.X, self.low),
                                   T.ge(self.X, self.high)),
                              np.inf, T.log(self.high - self.low))
        self.make_(self.nnlf_, "nnlf")

        # cdf
        self.cdf_ = T.switch(T.lt(self.X, self.low), 0.,
                             T.switch(T.lt(self.X, self.high),
                                      (self.X - self.low) / (self.high - self.low),
                                      1))
        self.make_(self.cdf_, "cdf")
