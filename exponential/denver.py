
from __future__ import division

import matplotlib.pyplot as plt
import math

from holtwinter import HoltWinter
from holtwinterd import HoltWinterDamp
from holt import Holt
from holtwinterAdd import HoltWinterAdd
from holtwinterDAdd import HoltWinterDampAdd
from optimizer import SimpleOptimizer

import util

def split_data(y):
    sz = int(0.8 * len(y))
    train = y[0:sz]
    validate = y
    return train, validate

def simple_test(y, ytest):
    ytrain, yvalidate = split_data(y)

    maxk = 25
    optimizer = SimpleOptimizer(maxk) 
    model = HoltWinterDamp()
    #model = HoltWinter()
    #model = HoltWinterAdd()
    #model = HoltWinterDampAdd()
    #model = Holt()

    optimizer.optimize_k(model, ytrain, yvalidate)
    #model.set_season(24)
    #optimizer.optimize(model, ytrain)

    err, yh = model.predict(ytest)
    rerr = 100 * err
    msg = "season=%s, error=%.4f" % (model.get_season(), rerr)
    print(msg + "\n" + model.get_info())

    data = []
    data.append(('real', ytest))
    data.append(('predict', yh))
    title = "%s, rmse=%.4f" % (model.get_name(), rerr)
    util.plot_figs(data, title)
    return


def main():
    fname_train = "../data/my-denver-train.csv"
    fname_test = "../data/my-denver-test.csv"

    ytrain = util.load_series(fname_train)
    ytest = util.load_series(fname_test)

    simple_test(ytrain, ytest)
    return

if __name__ == "__main__":
    main()
