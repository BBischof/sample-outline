# Capstone Exercise

Below is a suggested capstone exercise for lesson 4.2. Note that a notebook version of this exercise can be found [here](https://github.com/BBischof/sample-outline/blob/master/Capstone%20Exercise%20as%20Notebook.ipynb)

## Exercise

Now that we've seen both ARIMA and SARIMA models, let's add in the seasonal parameters and see how our key performance metrics change.

Since we've previously fit an ARIMA `(1,1,1)` model to the Mauna Kea CO2 data, we've saved the residuals to `y_res`. We've set that data to y,  that model to `mod`, and the results to `mle`.

The dataset `y_res` has been preloaded, and from our previous exercises, remember that we fit an ARIMA (1,1,1) model to this dataset. Some other useful things to remember is that we build a convenient ACF/PACF function, `cf2()`.

## Instructions

  * Investigate the ACF/PACF of `y_res` and observe the 12 month seasonality.
  * Define the `sdiff_res_12` series and determine seasonal parameters from it's ACF/PACF.
  * Fit SARIMA `(1,1,1)x(1,1,1,12)` —save model to `s_mod`, fit to `s_mle`— and examine the summary and diagnostics.
  * Plot the seasonal residue's ACF/PACF to see how much better they look.
  * Compare the AIC values for the `res` and `s_res`.

## Code

```python
cf2(y_res)

sdiff_res_12 = sm.tsa.statespace.tools.diff(y_res, k_seasonal_diff=1, k_seasons=12)
sdiff_res_12.plot(figsize=(15, 6))
plt.show()
cf2(sdiff_res_12)

s_mod = sm.tsa.statespace.SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
s_mle = s_mod.fit()
s_res = s_mle.resid[-500:]
print(s_mle.summary().tables[1])
s_mle.plot_diagnostics(figsize=(15, 12))
plt.show()

cf2(s_res)

print('ARIMA (1,1,1) - AIC:{}, SARIMA (1,1,1)x(1,1,1,12) - AIC:{}'.format(mle.aic, s_mle.aic))
```

## Helper functions, packages, and variable definitions

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())

mod = sm.tsa.statespace.SARIMAX(y, order=(1,1,1))
mle = mod.fit()
y_res = mle.resid[-500:]

def cf2(series, lag_num=40):
    fig = plt.figure(figsize=(12,8))
    ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_acf(series, lags=lag_num, ax=ax1)
    fig = sm.graphics.tsa.plot_pacf(series, lags=lag_num, ax=ax2)
```
