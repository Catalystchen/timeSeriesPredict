
import util

def load_data(fname):
    y = []
    yh = []
    fin = open(fname, 'r')
    line = fin.readline()

    for line in fin:
        items = line.strip().split(",")
        v1 = float(items[0])
        v2 = float(items[1])
        y.append(v1)
        yh.append(v2)
    fin.close()
    return y, yh

def main():
    fname = "./data/compare.dat"
    y, yh = load_data(fname)
    #yh = y[:]
    #yh.insert(0, y[0])
    n = len(y)
    #n = 1000
    rmse = util.calc_rmse(y, yh, n)

    dat = []
    dat.append(("real", y))
    dat.append(("predict", yh))
    util.plot_figs(dat, "Base, rmse=%.4f"%(rmse))
    return

if __name__ == "__main__":
    main()
