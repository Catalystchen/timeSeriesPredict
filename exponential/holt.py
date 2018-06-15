from __future__ import division
from sys import exit
from math import sqrt
import math
from numpy import array
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import util


class Holt:
    """Double Exponential Smoothing, a.k.a 
      Holter's linear trend method. It does not handle season. 

      lt = alpha*y[t] + (1-alpha)*(l[t-1] + b[t-1])
      bt = beta*(l[t] - l[t-1]) + (1-beta)*b[t-1]

      yhat = lt + bt
      
      https://www.otexts.org/fpp/7/2
    """
    def __init__(self, name='Holt'):
        self.alpha = 0.5
        self.beta = 0.5
        self.name = name
        return

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return

    def set_season(self, k):
        """it does not handle season component.
        """
        return

    def get_season(self):
        """it does not have season component"""
        return -1

    def get_info(self):
        template = "%s [%.3f, %.3f]" 
        msg = template % (self.name,
                          self.alpha,
                          self.beta)
        return msg

    def set_parameters(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        return

    def get_boundary(self):
        """the bounds of the parameters when optimizing"""
        bd = (0, 1)
        result = [bd, bd]
        return result

    def get_parameters(self):
        result = []
        result.append(self.alpha)
        result.append(self.beta)
        return result

    def calc_init_values(self, y):
        l0 = y[0]
        b0 = y[1] - y[0]
        return l0, b0

    def predict(self, y):
        # 1. set the initial values
        l0, b0 = self.calc_init_values(y)
        y0 = (l0 + b0)
        yh = [y0]

        alpha = self.alpha
        beta = self.beta
        
        # 2. rolling predict
        for i in range(len(y)):
            lt = alpha * y[i] + (1-alpha) * (l0 + b0)
            bt = beta * (lt - l0) + (1-beta)*b0
            yt = (lt + bt)

            yh.append(yt)
            l0 = lt
            b0 = bt

        #3. calcuate the error
        err = util.calc_rmse(y, yh, len(y))
        return err, yh
