# Exponential smoothing methods for prediction

Three variations of Exponential smoothing methods are implemented:

* **Holt method**: a.k.a Double exponential smoothing, which can handle trend;
* **Holt-Winter method**: a.k.a Triple exponential smoothing, which can handle both trend and season;
* **Damped Holt-Winter method**: the Holt-Winter method with damped trend;

[It](https://www.otexts.org/fpp/7/5) is said:
"A method that is often the single most accurate forecasting method for seasonal data is the Holt-Winters method with a damped trend and multiplicative seasonality." 


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

### The algorithms
The implementation of these Exponential smoothing methods are based on the book [`Forecasting: principles and practice`](https://www.otexts.org/fpp) by *Rob J Hyndman* and *George Athana­sopou­los*.

The training process to find the best values for parameters *alpha*, *Beta* and *Gamma* is implemented by using `L-BFGS-B` method in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). [Brute force method](https://github.com/songbinliu/timeSeriesPredict/blob/7b91c99ccc9ff02ecd2d198ddcd0e1701de0ceb2/exponential/optimizer.py#L23) is used to find the best value for **season**.  More details about the training process can be found in [`optimizer.py`](./optimizer.py)



##### Holt model
<div align="center">
<img width="600" src="https://user-images.githubusercontent.com/27221807/42188045-554d8b1c-7e20-11e8-9f6d-796d1024efeb.png">
</div>

This is the **additive** Holt model. Multiplicative method can be found in the [online book](https://www.otexts.org/fpp/7/2).

##### Holt-Winter model
<div align="center">
<img width="600" src="https://user-images.githubusercontent.com/27221807/42188165-cbf53616-7e20-11e8-965f-720f564e9e13.png">
</div>
This is the **additive** Holt-Winter model. Multiplicative method can be found in the [online book](https://www.otexts.org/fpp/7/5).


##### Holt-Winter-Damp model
<div align="center">
<img width="600" src="https://user-images.githubusercontent.com/27221807/42188299-544e1316-7e21-11e8-9210-0edc48753340.png">
</div>
This is the equations of the **Multiplicative** Holt-Winter damped model. Multiplicative method can be found in the [online book](https://www.otexts.org/fpp/7/5).
