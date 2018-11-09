# Machine Learning Engineer Nanodegree

# Capstone Project

Predicting Bitcoin Prices  
Lukasz Tymoszczuk  
November 8th, 2018

## Definition

### Project Overview

Bitcoin is decentralised crypto-currency which nobody physically owns. It exists only virtually and having security by design requires no intermediary institution to secure the transactions. Due to its nature, it is free from any bank and government actions. The most recent popularity raises a question when to buy (1).

High popularity causes that the currency is very volatile and unpredictable. Hype created by bitcoin is visible in social media. There are even interesting examples how sentiment analysis can be used to predict Bitcoin prices (2).

In this project I simplify the Bitcoin Price model and I assume that a lot of information about future prices can be deducted from the historical data, taken from kaggle.com (3). The model will use only the past performance in order to predict the future prices.

### Problem Statement

Bitcoin prices are time series data with `Close Price` and 'Volume' values. The goal is to create a model which used on unseen data predict prices as close as possible to actual prices. Such model even if not perfect, can be used as a better way than for example random decisions to buy or sell Bitcoins.

The steps for building and testing the model:

1. Preprocess the historical data
2. Use LSTM (Long-Short Term Memory) type of RNN (Recurrent Neural Network) to build a model
3. Test the model and tweak the parameters, layers in order to come up with a smallest error RMSE (Root Mean Square Error) calculated on predicted prices vs actual prices.
4. Once the model is tweaked, it will be compared (RMSE) with:
	- Random walk
	- Linear regression
	- Moving Average
5. It is expected that LSTM model will have the smallest RMSE among the above models.
6. The winner model can be then used on Production - in real-life trading, for example to determine if a long or short position should be opened tomorrow. 

### Metrics

Root square mean error for all days between predicted and actual prices will show us how close the model can predict actual prices:

```
RMSE = for all days (i): sqrt(sum((predicted_price_day_i - actual_price_day_i)^2))
```

The experiment will test if:

```
RMSE_LSTM = min(RMSE_LSTM, RMSE_RANDOM_WALK, RMSE_LINEAR_REGRESSION, RMSE_MOVING_WINDOW)
```

## Analysis

### Data Exploration

`coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv` contains 1819074 time series data. Every minute a transaction with following features is recorded:

- Open - The very first price in dollars for this time period
- High - The highest price in dollars for this time period
- Low - The lowest price in dollars for this time period
- Close - The last recorded price in dollars in this time period
- Volume (BTC & Currency) - How many bitcoins have been bought / sold in this time period

The data quality is good enough. There are no missing values for raw data (for any columns), but there are some missing days in the time series data. The number of missing days is not significant so it can be easily ignored. 

### Exploratory Visualization

**Fig. 1** The input close price data presents Bitcoin price between 01/12/2014 and 27/06/2018. It contains the 17th December 2017 peak of $19891.99

<img src="/bitcoin_prices.png" width=500; height=400>

**Fig. 2** The volume of Bitcoin transactions

<img src="/bitcoin_volume.png" width=500; height=400>


### Algorithms and Techniques



### Benchmark

## Methodology

### Data Preprocessing

### Implementation

### Refinement

## Results

### Model Evaluation and Validation

### Justification

## Conclusion

### Free-Form Visualization

### Reflection

### Improvemnet

## Resources

1. https://www.quora.com/What-is-bitcoin-Why-is-it-so-popular
2. https://hackernoon.com/sentiment-analysis-in-cryptocurrency-9abb40005d15
3. https://www.kaggle.com/mczielinski/bitcoin-historical-data

