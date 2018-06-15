from __future__ import division
from sys import exit
from math import sqrt
import math
from numpy import array
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import util


class HoltWinter:
    """Triple Exponential Smoothin, a.k.a 
      Holter-Winter's method with multiplicative seasonality.

      lt = alpha*(y[t] - s[t-m]) + (1-alpha)*(l[t-1] + b[t-1])
      bt = beta*(l[t] - l[t-1]) + (1-beta)*b[t-1]
      st = gamma*(y[t] - l[t-1] - b[t-1]) + (1-gamma) * s[t-m]

      yhat = lt + bt + s[t+1-m]
      
      https://www.otexts.org/fpp/7/5
    """
    def __init__(self, name='Holt-Winter'):
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5

        self.season = 12
        self.name = name
        return

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return

    def set_season(self, k):
        """set the number of seasons"""
        self.season = k
        return

    def get_season(self):
        return self.season

    def get_info(self):
        template = "%s season=%s, [%.3f, %.3f, %.3f]" 
        msg = template % (self.name,
                          self.season,
                          self.alpha,
                          self.beta,
                          self.gamma)
        return msg

    def set_parameters(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        return

    def get_boundary(self):
        """the bounds of the parameters when optimizing"""
        bd = (0, 1)
        result = [bd, bd, bd]
        return result

    def get_parameters(self):
        result = []
        result.append(self.alpha)
        result.append(self.beta)
        result.append(self.gamma)
        return result

    def calc_init_values(self, y, k):
        a0 = sum(y[0:k])/float(k)

        b0 = 0.0
        for i in range(k):
            b0 += (y[i+k] - y[i])
        b0 /= (k*k)

        s0 = {}
        for i in range(k):
            s0[i] = y[i] / a0
        return a0, b0, s0

    def predict(self, y):
        # 1. set the initial values
        a0, b0, s = self.calc_init_values(y, self.season)
        y0 = (a0 + b0)*s[0]
        yh = [y0]

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        
        # 2. rolling predict
        for i in range(len(y)):
            idx = i % self.season
            at = alpha * (y[i]/s[idx]) + (1-alpha) * (a0 + b0)
            bt = beta * (at - a0) + (1-beta)*b0
            st = gamma * (y[i]/(a0+b0)) + (1-gamma) * s[idx]
            # TODO: verify these two
            yt = (a0 + b0)*st
            #yt = (at + bt)*st

            yh.append(yt)
            a0 = at
            b0 = bt
            s[idx] = st

        #3. calcuate the error
        err = util.calc_rmse(y, yh, len(y))
        return err, yh

