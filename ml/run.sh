#########################################################################
# File Name: run.sh
# Author: songbin Liu
# mail: songbinliu@hotmail.com
# Created Time: Wed Jun 27 23:52:24 2018
#########################################################################
#!/bin/bash

#0. Download 'temperature.csv' from kaggle
 # https://www.kaggle.com/selfishgene/historical-hourly-weather-data#temperature.csv
 # put it to ./data/temperature.csv

#1. Extract the data for one city
# by default, it will extract the data for 'Denver', output=./data/denver.csv
python extract.py


#2. Generate training/test data sets
# output will be ./data/denver-features-test.csv and denver-features-train.csv
python gen_data.py


#3. Train the model
python main.py

