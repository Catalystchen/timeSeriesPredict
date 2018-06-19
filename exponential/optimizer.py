from __future__ import division
from scipy.optimize import minimize
import util


class SimpleOptimizer:
    """This optimizer is used to find the parameters for Holt-Winter models"""
    def __init__(self, maxSeason):
        self.max_season = maxSeason
        return

    def set_maxseason(self, maxseason):
        """set the maxValue of season when training"""
        self.max_season = maxseason
        return 

    def optimize_k(self, model, ytrain, yvalidate):
        err = 100000000.0

        k = 1
        maxk = self.max_season
        params = model.get_parameters()
        for nk in range(1, maxk):
            model.set_season(nk)
            success = self.optimize(model, ytrain)
            if not success:
                print("Failed to optimize for season=%s" % (nk))
                continue
            
            nerr, _ = model.predict(yvalidate)
            #msg = "k=%d, nerr=%.4f" % (nk, nerr)
            #print(msg)
            if nerr < err:
                err = nerr
                k = nk
                params = model.get_parameters()

        msg = "final: error=%.3f, season=%d, params=%s" % (err, k, params)        
        print(msg)

        model.set_season(k)
        model.set_parameters(params)
        return

    def optimize(self, model, y):
        inits = model.get_parameters()
        boundary = model.get_boundary()

        myargs = (model, y)
        success, params = _optimizer(_rmse_func, inits, myargs, boundary)
        if not success:
            return False
        model.set_parameters(params)
        return True


def _rmse_func(params, *args):
    model = args[0]
    y = args[1]   
    model.set_parameters(params)

    rmse, _ = model.predict(y)
    return rmse

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
        print("Failed to optimize: %s" % (result.message))
    return result.success, result.x
