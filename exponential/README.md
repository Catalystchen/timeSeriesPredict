# Exponential smoothing methods for prediction

Three variations of Exponential smoothing methods are implemented:

* **Holt method**: a.k.a Double exponential smoothing, which can handle trend;
* **Holt-Winter method**: a.k.a Triple exponential smoothing, which can handle both trend and season;
* **Damped Holt-Winter method**: the Holt-Winter method with damped trend;

[It](https://www.otexts.org/fpp/7/5) is said:
"A method that is often the single most accurate forecasting method for seasonal data is the Holt-Winters method with a damped trend and multiplicative seasonality" 


## Performance on the AirPassenger Dataset


|method| RMSE|
|-|-|
|Baseline | 33.5931 |
|Holt  | 34.3254 | 
|Holt-Winter  |17.3487 |
|Holt-Winter-Damp | **17.1779**|

* **Note 1**: Baseline is the method to use previous data to predict current data;
* **Note 2**: 70% data is used to select the parameters, and 30% data is used for testing;
* **Note 3**: *Holt-Winter* and *Holt-Winter-Damp* are using the multiplicative seasonality;
* **Note 4**: the **RMSE** is calculated on the whole dataset (training + testing data).

### Figures

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42187305-9111e34e-7e1d-11e8-8c92-850c08cabfb9.png">
  <p align="center"> Figure-1 Holt-Winter-Damp model for AirPassenger prediction. </p>
</div>

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42187576-8a26375a-7e1e-11e8-87f3-54cf147e25fc.png">
  <p align="center"> Figure-2 Holt-Winter model for AirPassenger prediction. </p>
</div>


<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/27221807/42187554-7830c42a-7e1e-11e8-9da7-a4fcd4fbf9c0.png">
  <p align="center"> Figure-3 Holt model for AirPassenger prediction. </p>
</div>
