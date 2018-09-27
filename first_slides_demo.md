# Introduction to Autoregressive models(AR)

## What is AutoRegression?

### Part 1

AutoRegression refers to the process of regressing the time-dependent variable of interest as a _linear combination of past values of the variable._

Remember that $y\_{t-2}$ refers to the lag-2 value of our time-dependent variable.

An AR(p) is written as:

$y\_t = c + \beta\_{1}y\_{t-1} + \beta\_{2}y\_{t-2} + \ldots + \beta\_{p}y\_{t-p} + e\_t$

with $e\_{t}$ white noise.

### Script

Auto means "self", and regression, you probably know from other DataCamp courses; when we put those together, it should not surprise you that, autoregression refers to expressing the time-dependent variable as a linear combination of past values of that variable.

Remember that our subscripts indicate the lags from our current variable, and observe the usual multilinear regression formula with features given by lags.

If you're familiar with regression, you'll notice the usual constant term, but maybe the white noise will feel new. That's ok, it's just a time-dependent baseline value that is always present in real data.

## Identifying autoregressive behavior in data

### Part 1

In practice, we'll use the ACF and PACF to identify AR models; the data may follow an AR(p) model if:

- the ACF is exponentially decaying or sinusoidal,
_and_
- the PACF has a significant spike at lag p, but none beyond.

Let's generate some data from an AR(1) model, add some noise, and look at the ACF/PACF e.g.:

### Script

It's rarely possible to guess a timeseries' underlying model from the plot alone, but with our trusty ACF and PACF plots, the signature of an AR model is able to be uncovered.

If the ACF is exponentially decaying or sinusoidal, and the PACF has a significant spike at lag p, but none beyond—you've got yourself an AR model of order p.

Let's take a look at a nice picture on the next slide.

## Identifying autoregressive behavior in data (example)

### Part 1

```
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object = ArmaProcess(ar1, ma1)
simulated_ar_data = AR_object.generate_sample(nsample=1000)
noise = np.random.normal(0,0.2,1000)
cf2(simulated_ar_data + noise)
```

![cf plots](https://github.com/BBischof/sample-outline/blob/master/cf_plots_ar1_sim.png)

### Script

In the code block, we set our phi value to -0.9, and use StatsModels' ArmaProcess to generate our AR(1) series. To keep things interesting, we also make some noise and add that before plotting our ACF/PACF plots.

Remember, the code for making those plots has been wrapped up into a function called cf2.

From our plots we see a lovely exponential decay in the ACF, and the sharp dropoff from significance in the 1-lag of the PACF.

I know what you're thinking, and you're right this is an AR(1) model based on the criterium on the previous slide.

