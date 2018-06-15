from __future__ import division
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import util


class HoltWinterDamp:
    """Triple Exponential Smoothin with damped, a.k.a 
      Holter-Winter's method with damped trend and 
      multiplicative seasonality.

      https://www.otexts.org/fpp/7/5
    """
    def __init__(self, name='Holt-Winder-D'):
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        self.damp = 0.85

        self.season = 12
        self.name = name
        return

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return    

    def set_damp(self, d):
        """damp should be in the range of [0, 0.98]"""
        self.damp = d

    def set_season(self, k):
        """set the number of seasons"""
        self.season = k
        return

    def get_season(self):
        return self.season

    def get_info(self):
        template = "%s season=%s, [%.3f, %.3f, %.3f, %.3f]" 
        msg = template % (self.name,
                          self.season,
                          self.alpha,
                          self.beta,
                          self.gamma,
                          self.damp)
        return msg

    def set_parameters(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.damp = params[3]
        return

    def get_parameters(self):
        result = []
        result.append(self.alpha)
        result.append(self.beta)
        result.append(self.gamma)
        result.append(self.damp)
        return result

    def get_boundary(self):
        """the bounds of the parameters when optimizing"""
        bd = (0,1)
        result = [bd, bd, bd, (0, 0.99)]
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
        damp = self.damp
        
        # 2. rolling predict
        for i in range(len(y)):
            idx = i % self.season
            at = alpha * (y[i]/s[idx]) + (1-alpha) * (a0 + b0)
            bt = beta * (at - a0) + (1-beta)*damp*b0
            st = gamma * (y[i]/(a0 + damp*b0)) + (1-gamma) * s[idx]
            # TODO: verify these two
            #yt = (at + damp*bt)*st
            yt = (a0 + damp*b0)*st

            yh.append(yt)
            a0 = at
            b0 = bt
            s[idx] = st

        #3. calcuate the error
        err = util.calc_rmse(y, yh, len(y))
        return err, yh

    def fit(self, y):
        inits = self.get_parameters()
        boundary = self.get_boundary()

        myargs = (self, y)
        success, params = _optimizer(_hwd_rmse, inits, myargs, boundary)
        if not success:
            return False
        self.set_parameters(params)
        return True 


def _optimizer(fun, x0, margs, bounds):
    """
    Two methods for Bound-Constrained minimization:
       'TNC' and 'L-BFGS-B'.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    options = {
        'maxiter':200,
        'disp':False,
    }
    result = minimize(fun,
                      x0 = x0,
                      bounds = bounds,
                      #method = 'TNC',
                      method = 'L-BFGS-B',
                      options = options, 
                      args = margs)
    
    if not result.success:
        print("Failed to optimize: " + result.message)
    return result.success, result.x

def _hwd_rmse(params, *args):
    model = args[0]
    y = args[1]

    model.set_parameters(params)
    rmse, _ = model.predict(y)
    return rmse

def auto_optimize(y, model, maxk):
    err = 1000000.0
    sz = int(0.7 * len(y))
    train = y[0:sz]
    if maxk > sz/2:
        maxk = int(sz/2)
    
    k = 1
    params = model.get_parameters()
    for nk in range(1, maxk):
        model.set_season(nk)
        success = model.fit(train)
        if not success:
            print("failed to optimize for season=%s"% (nk))
            continue
        nerr, _ = model.predict(y)
        if nerr < err:
            err = nerr
            k = nk
            params = model.get_parameters()
    msg = "final: k=%s, error=%.3f, params=%s" % (k, err, params)
    print(msg)
    model.set_parameters(params)
    model.set_season(k)
    return

def main():
    fname = "AirPassengers.csv" 
    #fname = "shampoo-sales.csv"
    #fname = "daily_birth.csv"
    y = util.load_series("../data/" + fname)
    model = HoltWinterDamp()
    #model.set_damp(0.837)
    #model.set_damp(0.937)
    maxk = 13
    auto_optimize(y, model, maxk)
    err, yh = model.predict(y)
    msg = "season=%s, error=%.3f" % (model.get_season(), err)
    print(msg + "\n" + model.get_info())
    util.plot_figs(y, yh, msg)
    return

if __name__ == "__main__":
    main()