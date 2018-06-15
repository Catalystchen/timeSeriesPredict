from __future__ import division
import matplotlib.pyplot as plt
import math

def calc_rmse(y, yh, n):
    """
            idx = i % self.season
            lt = alpha * (y[i]/s[idx]) + (1-alpha) * (l[t-1] + b[t-1])
            bt = beta * (l[t] - l[t-1]) + (1-beta)*damp*b[t-1]
            st = gamma * (y[i]/(l[t-1] + damp*b[t-1])) + (1-gamma) * s[idx]

            yhat = (at + damp*bt)*st

    """
    err = 0.0
    for i in range(n):
        delta = yh[i] - y[i]
        delta = delta * delta
        err += delta
    
    rmse = math.sqrt(err/n)
    return rmse

def plot_figs(y, yhat, title):
    plt.plot(y, label='real')
    plt.plot(yhat, label='predict')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
    return


def float_str(alist):
    result = "["
    for i in alist:
        result += "%.2f, " % (i)
    result += "]"
    return result


def load_series(fname):
    fhd = open(fname, 'r')
    line = fhd.readline()
    print("header:" + line)
    result = []
    for line in fhd:
        items = line.strip().split(",")
        result.append(float(items[1]))

    fhd.close()
    return result