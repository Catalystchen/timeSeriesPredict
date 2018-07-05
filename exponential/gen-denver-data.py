
MAX_VALUE = 100.0
AVG_VALUE = 0.0

def norm_value(value):
    result = (value - AVG_VALUE) / MAX_VALUE
    return result


def recover_value(value):
    result = value * MAX_VALUE + AVG_VALUE
    return result


def load_values(fname_in):
    result = []
    fin = open(fname_in, 'r')
    # discard the first line
    line = fin.readline()

    for line in fin:
        items = line.strip().split(",")
        dtime = items[0].strip()
        value = float(items[1])
        nvalue = norm_value(value)

        result.append((dtime, value, nvalue))
    fin.close()
    return result


def generate_data(values, fname_prefix, fraction=0.7):
    fname_train = "%s-train.csv" % (fname_prefix)
    fname_test = "%s-test.csv" % (fname_prefix)

    tsize = int(fraction * len(values))

    fout_train = open(fname_train, 'w')
    fout_test = open(fname_test, 'w')

    fout = fout_train
    for i in range(len(values)):
        dtime, value, nvalue = values[i]
        nline = "%s, %.4f, %.4f" % (dtime, nvalue, value)
        fout.write(nline + '\n')
        if i == tsize:
           fout = fout_test

    fout_test.close()
    fout_train.close()
    return


def main():
    fname = "../data/denver.csv"
    fname_prefix = "../data/my-denver"

    values = load_values(fname)
    generate_data(values, fname_prefix, fraction=0.8)
    return

if __name__ == "__main__":
    main()
