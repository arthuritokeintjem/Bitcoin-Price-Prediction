# Bitcoin Price Prediction 📈

## 📋 Project Overview

This project focuses on developing and evaluating various time series forecasting models to predict the daily closing price of Bitcoin (BTC). Given Bitcoin's high volatility and significant market interest, accurate price prediction is crucial for investors and traders to make informed decisions, manage risk, and optimize investment strategies. This project explores statistical models like ARIMA and deep learning models like GRU and LSTM.

---

## 🎯 Problem Statement

* How accurately can we predict the future price of Bitcoin using historical price data?
* Which time series forecasting model (ARIMA, GRU, LSTM) performs best for Bitcoin price prediction given its volatile and non-linear nature?
* What are the key data preparation steps and feature engineering techniques suitable for Bitcoin price data?

---

## 📊 Dataset

The dataset used for this project consists of historical daily Bitcoin (BTC) price data, typically including features like:
* **Date**
* **Open**
* **High**
* **Low**
* **Close** (this is the target variable for prediction)
* **Volume**

The data was sourced using the `yfinance` library or a similar financial data provider (as detailed in the notebook `notebook.ipynb`).

---

## 🛠️ Methodologies & Models

Three main time series forecasting models were implemented and evaluated:

1.  **ARIMA (Autoregressive Integrated Moving Average):**
    * A classical statistical model that captures linear trends and seasonality in time series data.
    * Pros: Interpretable, good for linear patterns.
    * Cons: Struggles with non-linearities, requires data stationarity.

2.  **GRU (Gated Recurrent Unit):**
    * A type of Recurrent Neural Network (RNN) with gating mechanisms (reset and update gates) to manage information flow.
    * Pros: More efficient than LSTM, good performance on many sequential tasks.
    * Cons: May not capture very long-term dependencies as well as LSTM.

3.  **LSTM (Long Short-Term Memory):**
    * An advanced RNN architecture designed to learn long-term dependencies and complex non-linear patterns using input, forget, and output gates.
    * Pros: Excellent for complex sequences and long-term dependencies, robust to vanishing gradients.
    * Cons: Computationally more expensive, may require more data and tuning.

---

## ⚙️ Project Workflow

1.  **Data Collection:** Fetching historical Bitcoin price data.
2.  **Exploratory Data Analysis (EDA):** Visualizing price trends, volatility, and other statistical properties.
3.  **Data Preprocessing:**
    * Handling any missing values.
    * Feature selection (focusing on the 'Close' price).
    * Data normalization/scaling (e.g., MinMaxScaler) to prepare data for neural networks.
    * Creating sequences/windows of data for training deep learning models.
4.  **Model Development & Training:**
    * Implementing ARIMA, GRU, and LSTM models.
    * Splitting data into training and testing sets.
    * Training the models on the training data.
    * Hyperparameter tuning was considered essential for optimizing model performance.
5.  **Model Evaluation:**
    * Predicting prices on the test set.
    * Evaluating models using **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**.
    * Comparing model performance to identify the best-performing model.

---

## 📈 Results & Evaluation

The models were evaluated based on their RMSE and MAPE on the test data:

| Model          | RMSE (USD) | MAPE (%) |
|----------------|------------|----------|
| ARIMA          | 21937.81   | 18.60    |
| GRU            | 15195.96   | 15.99    |
| **LSTM** | **2168.12**| **2.02** |

**Key Findings:**
* The **LSTM model significantly outperformed** both ARIMA and GRU, achieving the lowest RMSE (2168.12 USD) and MAPE (2.02%).
* This suggests LSTM's superior ability to capture the complex, non-linear dynamics and long-term dependencies inherent in Bitcoin price data.
* ARIMA, being a linear model, struggled the most with Bitcoin's volatility.
* GRU showed better performance than ARIMA but was not as effective as LSTM.

---

## 💡 Conclusion

The LSTM model demonstrated the highest accuracy for predicting Bitcoin's daily closing price within this project's scope. This highlights the power of deep learning architectures for modeling highly volatile financial time series. Future work could involve more extensive hyperparameter optimization, incorporating external factors (e.g., market sentiment, news), or exploring hybrid models.

---

## 💻 Technologies Used

* **Programming Language:** Python 3
* **Key Libraries:**
    * Pandas (Data manipulation)
    * NumPy (Numerical operations)
    * Matplotlib & Seaborn (Data visualization)
    * Scikit-learn (Data preprocessing, metrics)
    * Statsmodels (ARIMA model)
    * TensorFlow & Keras (GRU, LSTM models)
    * yfinance (Data acquisition - *if applicable*)
