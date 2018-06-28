from __future__ import division
import matplotlib.pyplot as plt


def plot_figs(dat, title):
    if len(dat) < 1:
        print("Failed to plot empty data")
        return
    
    for item in dat:
        legend, data = item
        plt.plot(data, label=legend)

    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
    return

def write_csv(dat, fname):
    fhdl = open(fname, 'w')
    fhdl.write("#index, hourly-temprature\n")
    for i in range(len(dat)):
        line = "%d, %.4f\n" % (i, dat[i])
        fhdl.write(line)
    fhdl.close()
    return

def write_time_csv(dat, fname):
    fhdl = open(fname, 'w')
    fhdl.write("#index, hourly-temprature\n")
    for i in range(len(dat)):
        pair = dat[i]
        line = "%s, %.4f\n" % (pair[0], pair[1])
        fhdl.write(line)
    fhdl.close()
    return
