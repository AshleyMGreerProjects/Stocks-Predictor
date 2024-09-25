# Stock Price Predictor

1. **Project Overview**
2. **Data Description**
3. **Models Used (Linear Regression and LSTM)**
4. **Results and Accuracy Analysis**
5. **Code Explanation**
6. **How to Run the Project**
7. **Visualizations**

---

This project is built to predict stock prices using two primary machine learning models: **Linear Regression** and **Long Short-Term Memory (LSTM)**. It includes various stocks such as **AAPL (Apple), MSFT (Microsoft), MMM (3M),** and **NVDA (NVIDIA)**.

#### Key Features:
- **Multiple Models:** Linear Regression and LSTM for price prediction.
- **Stock Data:** Fetches historical stock prices via the `yfinance` library.
- **Prediction Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R² Score.
- **Plotting and Visualization:** Visualizes both actual vs. predicted stock prices.

---

### Data Description

The historical stock data consists of the following features:
- **Date**
- **Open Price**
- **High Price**
- **Low Price**
- **Close Price (Target Variable)**
- **Volume**

Additional technical indicators such as:
- **Moving Averages (MA)**
- **Relative Strength Index (RSI)**
- **Bollinger Bands**
- **MACD (Moving Average Convergence Divergence)**

---

### Models Used

1. **Linear Regression:**
   A simple statistical model that assumes a linear relationship between the input features (past stock prices, technical indicators) and the target variable (future stock price).

2. **LSTM (Long Short-Term Memory):**
   A more complex recurrent neural network model, designed for time series data like stock prices. It excels at capturing temporal dependencies, making it effective for longer-term trends.

---

### Results and Accuracy Analysis

#### 1. **Linear Regression**

- **AAPL:**
  - Predicted Price: $226.45
  - MSE: 4.1814
  - MAE: 1.2962
  - R²: 0.9989

- **MSFT:**
  - Predicted Price: $433.76
  - MSE: 15.5096
  - MAE: 2.6464
  - R²: 0.9987

- **MMM:**
  - Predicted Price: $135.48
  - MSE: 3.5259
  - MAE: 1.2173
  - R²: 0.9967

- **NVDA:**
  - Predicted Price: $116.14
  - MSE: 2.3596
  - MAE: 0.7863
  - R²: 0.9975

#### 2. **LSTM**

- **AAPL:**
  - Predicted Price: $198.20
  - MSE: 149.8215
  - MAE: 8.7171
  - R²: 0.9602

- **MSFT:**
  - Predicted Price: $376.68
  - MSE: 770.3661
  - MAE: 20.9653
  - R²: 0.9335

- **MMM:**
  - Predicted Price: $132.73
  - MSE: 25.8778
  - MAE: 3.9326
  - R²: 0.9755

- **NVDA:**
  - Predicted Price: $115.52
  - MSE: 5.2791
  - MAE: 1.2497
  - R²: 0.9944

#### **Model Comparison:**
- Linear Regression outperforms LSTM for short-term predictions in terms of accuracy, as indicated by lower MSE and higher R² scores. However, LSTM can capture longer-term patterns, which makes it better suited for time-series data with high volatility or trends over time.

---

### Code Explanation

1. **Data Loading:**
   - The stock data is fetched using `yfinance`. The features are then processed and split into training and testing sets.
   
2. **Feature Engineering:**
   - Moving Averages, RSI, MACD, and Bollinger Bands are added as features for the models to capture stock market trends.

3. **Linear Regression:**
   - The `scikit-learn` library is used to implement a basic linear regression model that predicts stock prices based on the engineered features.

   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

4. **LSTM:**
   - The LSTM model is built using the `TensorFlow` and `Keras` libraries. The sequential nature of stock prices is captured through the LSTM layers, making it suitable for time-series forecasting.

   ```python
   model = Sequential()
   model.add(LSTM(50, activation='relu', input_shape=input_shape))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse')
   ```

5. **Performance Evaluation:**
   - The models are evaluated using MSE, MAE, MAPE, and R² score.

   ```python
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   ```

6. **Visualization:**
   - The actual vs. predicted stock prices are plotted for each stock symbol using `matplotlib`.

---

### How to Run the Project

1. **Install the required libraries:**
   ```bash
   pip install yfinance pandas scikit-learn tensorflow matplotlib
   ```

2. **Run the script:**
   ```bash
   python machine_learning_market_predictor.py
   ```

3. **Input the stock symbol and select a model:**
   - The app will predict the next day's stock price and display error metrics.

---

### Visualizations are in the folders!
