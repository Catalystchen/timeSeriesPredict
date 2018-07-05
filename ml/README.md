# machine learning for time series Prediction

This project shows how to do time series data prediction with machine learning models.
The data is `Historical Hourly Weather Data 2012-2017` from [Kaggle](https://www.kaggle.com/selfishgene/historical-hourly-weather-data#temperature.csv).

Two regression models of [Tensorflow](https://www.tensorflow.org) are used in this project: 
* [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor)  
* [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor)

These two models are used to do one step ahead prediction.


# Prediction performance
Three models are used to evaluate the performance:

* The `Baseline` method is to use the value of last step as the predicted value of next step;
* The `DNN` model is a simple shallow nerual networks with three hidden layers: 32 x 16 x 16;
* The `Linear` model has the same feature set as the `DNN` model.

Following table gives the inital `RMSE` of the three models without much tune:

|method| train-RMSE | test-RMSE|
|-|-|-|
|Baseline |-| 1.6046 |
|DNN | 1.3849 | 1.1665 | 
|Linear | 1.3946 |**1.1378** |
|Holt-Winter-D| 1.9 | 1.6045 |  <!-- [0.816, 0.695, 1.000, 0.793] -->
|Holt-Winter| 1.9 | 1.6063 | <!-- [0.503, 0.577, 1.000] -->

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42130796-2e2696fa-7cbd-11e8-99c9-c84ccd720131.png">
 <p align="center"> Figure-1 DNN model for Hourly temperature prediction. </p>
</div>

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42300962-e7ea2a36-7fe0-11e8-9cd3-c8a54b1233b0.png">
  <p align="center"> Figure-2 Holt-Winter-D model for Hourly temperature prediction. </p>
</div>

# Train the models

### 0. Download the dataset
Download `temperature.csv` from [Kaggle project](https://www.kaggle.com/selfishgene/historical-hourly-weather-data#temperature.csv);
and put it to `./data/temperature.csv`

### 1. Extract the data for one city
```bash
python extract.py
```
By default, it will extract the data for `Denver`, output=./data/denver.csv


### 2. Generate training/test data sets
```bash
python gen_data.py
```
This will genereate features for the training data and test data. Currently, the features are:
* the values of last `10` time steps;
* the deltas of the the last `10` time steps;
* the `10` values of previous days aroud current time;
* the deltas the previous days values;
* the month of the data;
* the day of the data;
* the hour of the data;

By default, output will be `./data/denver-features-test.csv` and `./data/denver-features-train.csv`

### 3. Train the model
```bash
python main.py
```
 `DNNRegressor` or `LinearRegressor` can be selected in `main.py` manually before the training. 
 
 ### 4. calculate the RMSE and draw the data
 ```bash
 python compare.py
 ```
 

