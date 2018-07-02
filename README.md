# timeSeriesPredict
This project uses `Statistical` and `Machine Learning` methods for time series data prediction. As suggested in the paper [`Statistical and Machine Learning forecasting methods: Concerns and ways forward`](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889), both types of methods are effective. So they are implemented in this project to do one step ahead prediction.



## Statistical methods
[`Exponential smoothing`](https://en.wikipedia.org/wiki/Exponential_smoothing) methods are implemented for time series data prediction.
More specifically, the `Holt-Winters` methods, which take into account the moving average, trend, and seasonal characteristics.

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42143859-64bc8026-7d85-11e8-9563-0d391921126e.png">
  <p align="center"> Figure-1 Monthly AirPassenger prediction. </p>
</div>


## Machine learning methods
Inspired by the statistical methods, for machine learning methods, we can draw features based on recent moving averae, trend, and multiple
seasonal characteristics. (Exponential smoothing methods can only handle one season)

For example, for on-line service workload, we can assume it has mulitple seasons: hourly, daily, weekly, monthly, and yearly. Thus we can use these seasonal values to predict future values. Let the model to learn how much each season will contribute.

<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42143980-73f0b1ba-7d86-11e8-8267-7910b022dbdd.png">
<p align="center"> Figure-2 Hourly temperature prediction.</p>
</p>

Two machine learning methods: **Linear regression** and **neural-networks regression** are applied (using [Tensorflow](https://www.tensorflow.org)) to do the prediction.

