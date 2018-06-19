from __future__ import division

import matplotlib.pyplot as plt
import util

from holtwinter import HoltWinter
from holtwinterd import HoltWinterDamp
from holt import Holt
from holtwinterAdd import HoltWinterAdd
from holtwinterDAdd import HoltWinterDampAdd
from optimizer import SimpleOptimizer

def split_data(y):
    sz = int(0.7 * len(y))
    train = y[0:sz]
    validate = y
    return train, validate

def simple_test(y):
    ytrain, yvalidate = split_data(y)

    maxk = 36
    optimizer = SimpleOptimizer(maxk) 
    model = HoltWinterDamp()
    #model = HoltWinter()
    #model = HoltWinterAdd()
    #model = HoltWinterDampAdd()
    #model = Holt()

    optimizer.optimize_k(model, ytrain, yvalidate)
    #model.set_season(24)
    #optimizer.optimize(model, ytrain)

    err, yh = model.predict(y)
    msg = "season=%s, error=%.4f" % (model.get_season(), err)
    print(msg + "\n" + model.get_info())

    data = []
    data.append(('real', y))
    data.append(('predict', yh))
    util.plot_figs(data, msg)
    return

def compare_models(y):
    ytrain, yvalidate = split_data(y)

    maxk = 13
    optimizer = SimpleOptimizer(maxk) 

    modelH = Holt()
    modelHW = HoltWinter()
    modelHWD = HoltWinterDamp()

    optimizer.optimize(modelH, ytrain)
    optimizer.optimize_k(modelHW, ytrain, yvalidate)
    optimizer.optimize_k(modelHWD, ytrain, yvalidate)

    errH, yH = modelH.predict(y)
    errHW, yHW = modelHW.predict(y)
    errHWD, yHWD = modelHWD.predict(y)

    results = {}
    results['Holt'] = (errH, modelH, yH)
    results['Holt-Winter'] = (errHW, modelHW, yHW)
    results['Holt-Winter-D'] = (errHWD, modelHWD, yHWD)

    for name in results:
        err, model, yh = results[name]
        msg = "model:%s err=%.4f" % (model.get_info(), err)
        print(msg)
        title = "model: %s, rmse=%.4f" % (name, err)
        data = []
        data.append(('real', y))
        data.append(('predict', yh))
        util.plot_figs(data, title)

    return

def main():
    fname = "AirPassengers.csv" 
    #fname = "shampoo-sales.csv"
    #fname = "daily_birth.csv"
    #fname = "denver.csv"
    y = util.load_series("../data/" + fname)
    simple_test(y)
    #compare_models(y)
    return

if __name__ == "__main__":
    main()
