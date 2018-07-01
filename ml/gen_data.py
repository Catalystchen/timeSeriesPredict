from datetime import datetime


MAX_VALUE = 50
AVG_VALUE = 273.15
TIME_FORMAT = '%Y-%m-%d %H:00:00'


def test_time():
    times = ["2012-10-01 17:00:00",
             "2012-10-01 20:00:00"]

    for item in times:
        dt = datetime.strptime(item, TIME_FORMAT)
        print("month=%d, hour=%d"%(dt.month, dt.hour))         
    return


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


def generate_features(values):
    deltas = []
    for i in range(1, len(values)):
        deltas.append(values[i] - values[i-1])
    
    tmp1 = ", ".join(format(x, "0.7f") for x in values)
    tmp2 = ", ".join(format(x, "0.7f") for x in deltas)
    return tmp1 + ", " + tmp2


def get_hourly_features(values, idx):
    n = 10
    prevalues = []

    for i in range(idx-n, idx):
        prevalues.append(values[i][2])

    return generate_features(prevalues)


def get_daily_features(values, idx):
    n = 5
    preday = idx - 24
    prevalues = []
    for i in range(preday-n, preday+n):
        prevalues.append(values[i][2])
    return generate_features(prevalues)


def generate_header():
    header = "#datetime, month, day, hour, original, y, "

    #hourly
    n = 10
    for i in range(n):
        header += ("y-%d, " % (n-i))

    for i in range(n-1):
        header += ("d-%d, " % (n-i))    

    #daily
    for i in range(n):
        header += ("yy-%d, " % (n-i))

    for i in range(n-1):
        header += ("dd-%d, " % (n-i))    

    header = header.strip().strip(',')
    return header

def generate_train_test_set(fname_in, fname_out_prefix, fraction=0.7):
    #1. load all the values
    values = load_values(fname_in)

    #2. calculate the size of train and test
    train_size = int(fraction * len(values)) 

    fname_out_train = "%s-train.csv" % (fname_out_prefix)
    fname_out_test = "%s-test.csv" % (fname_out_prefix)
    fout_train = open(fname_out_train, 'w')
    fout_test = open(fname_out_test, 'w')
    header = generate_header()
    fout_train.write(header+'\n')
    fout_test.write(header + '\n')

    #3. write the train and test
    fout = fout_train
    for idx in range(48, len(values)):
        dtime, value, nvalue = values[idx]
        dto = datetime.strptime(dtime, TIME_FORMAT)

        hourly = get_hourly_features(values, idx)
        daily = get_daily_features(values, idx)

        nline = "%s, %d, %d, %d, %.4f, %.7f, %s, %s" % (dtime, dto.month, dto.day, dto.hour,
                                                        value, nvalue,
                                                        hourly, daily)
        fout.write(nline + '\n')    
        if idx == train_size:
            fout = fout_test

    fout_train.close()
    fout_test.close()
    return

def main():
    fname_in = "./data/denver.csv"
    fname_out = "./data/denver-features"

    generate_train_test_set(fname_in, fname_out, 0.80)
    return


if __name__ == "__main__":
    main()
