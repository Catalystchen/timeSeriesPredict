# ============================================================
# Historical Hourly Weather Data 2012-2017
# Hourly weather data for 30 US & Canadian Cities + 6 Israeli Cities
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data#temperature.csv
# ============================================================
import util

def load_city_dict(line):
    """ line format:
    datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
    """
    line = line.strip()
    items = line.split(",")
    mydict = {}

    for i in range(len(items)):
        cname = items[i].strip()
        mydict[cname] = i

    print(mydict)
    return mydict

def load(fname, cityName):
    fhdl = open(fname, 'r')
    header = fhdl.readline()
    mydict = load_city_dict(header)

    idx = mydict[cityName]
    msg = "city:%s, index=%d" % (cityName, idx)
    print(msg)

    result = []
    i = -1 
    for line in fhdl:
        i = i + 1
        line = line.strip()
        items = line.split(",")
        if idx < len(items):
            value = items[idx]
            if len(value) > 1:
                result.append(float(value))
                continue

        print("missing value %d" % (i))
        if len(result) > 44000:
            break
        if len(result) > 0:
            value = result[len(result)-1]
            result.append(value)

    fhdl.close()

    print("%d Vs. %d data points" % (len(result), i))
    return result


def get_city_time_value(fname, cityName):
    fhdl = open(fname, 'r')
    header = fhdl.readline()
    mydict = load_city_dict(header)

    idx = mydict[cityName]
    msg = "city:%s, index=%d" % (cityName, idx)
    print(msg)

    result = []
    i = -1 
    for line in fhdl:
        i = i + 1
        items = line.strip().split(",")
        dtime = items[0].strip()
        if idx < len(items):
            value = items[idx]
            if len(value) > 1:
                value = float(value)
                pair = (dtime, value)
                result.append(pair)
                continue

        print("missing value %d" % (i))
        if len(result) > 44000:
            break

        if len(result) > 0:
            value = result[len(result)-1][1]
            pair = (dtime, value)
            result.append(pair)

    fhdl.close()    
    print("%d Vs. %d data points" % (len(result), i))
    return result


def test1():
    fname = "./data/temperature.csv"
    city = "Denver"
    #city = "New York"
    #city = "Kansas City"
    #city = "Seattle"
    data = load(fname, city)
    util.write_csv(data, "./data/denver.csv")

    dat = [(city, data)]
    util.plot_figs(dat, "temprature")
    return


def test2():
    fname = "./data/temperature.csv"
    city = "Denver"
    #city = "New York"

    data = get_city_time_value(fname, city)
    ofname = "./data/%s.csv" % ('denver')
    util.write_time_csv(data, ofname)

    values = [x[1]-273 for x in data]
    dat = [(city, values)]
    util.plot_figs(dat, "temprature")
    return

def main():
    test2()
    return

if __name__ == "__main__":
    main()
