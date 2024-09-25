import os
import sys
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QDialog, QVBoxLayout, QPlainTextEdit,
    QInputDialog, QFileDialog, QLabel, QComboBox, QPushButton, QLineEdit, QGridLayout,
    QTabWidget, QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QHBoxLayout
)

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
)
from sklearn.svm import SVR

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for caching data and saving models
CACHE_DIR = "cache"
MODEL_DIR = "models"

# Expanded FEATURE_COLUMNS with additional technical indicators
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI',
    'MACD', 'Signal', 'BB_Middle', 'BB_Upper', 'BB_Lower',
    'Stochastic_Oscillator', 'ATR', 'ADX', 'CCI', 'Williams_%R', 'OBV'
] + [f'Close_lag_{lag}' for lag in range(1, 6)]

AVAILABLE_MODELS = {
    "Linear Regression": LinearRegression,
    "Random Forest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Support Vector Regressor": SVR,
    "LSTM": "LSTM"  # Placeholder for LSTM
}

# Ensure cache and model directories exist
for directory in [CACHE_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Step 1: Fetch Financial Data using yfinance
def scrape_data(stock_symbol, start_date=None, end_date=None):
    stock_cache_dir = os.path.join(CACHE_DIR, stock_symbol)
    os.makedirs(stock_cache_dir, exist_ok=True)
    cache_file = os.path.join(stock_cache_dir, f"{stock_symbol}.csv")
    if os.path.exists(cache_file) and not start_date and not end_date:
        df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
    else:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("No data found for this stock symbol and date range.")
        df.to_csv(cache_file)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df

# Step 1a: Feature Engineering with Expanded Technical Indicators
def add_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    df['RSI'] = compute_RSI(df)
    df['MACD'], df['Signal'] = compute_MACD(df)
    
    # New Indicators
    df = compute_bollinger_bands(df)
    df = compute_stochastic_oscillator(df)
    df = compute_atr(df)
    df = compute_adx(df)
    df = compute_cci(df)
    df = compute_williams_r(df)
    df = compute_obv(df)

    for lag in range(1, 6):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    df.dropna(inplace=True)
    return df

def compute_RSI(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_MACD(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    MACD = exp1 - exp2
    Signal = MACD.ewm(span=9, adjust=False).mean()
    return MACD, Signal

def compute_bollinger_bands(df, window=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window).std() * num_std)
    df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window).std() * num_std)
    return df

def compute_stochastic_oscillator(df, window=14):
    low_min = df['Low'].rolling(window).min()
    high_max = df['High'].rolling(window).max()
    df['Stochastic_Oscillator'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return df

def compute_atr(df, window=14):
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close_prev'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_Close_prev'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_Close_prev', 'Low_Close_prev']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window).mean()
    df.drop(['High_Low', 'High_Close_prev', 'Low_Close_prev', 'TR'], axis=1, inplace=True)
    return df

def compute_adx(df, window=14):
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close_prev'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_Close_prev'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_Close_prev', 'Low_Close_prev']].max(axis=1)

    df['DM_plus'] = df['High'].diff()
    df['DM_minus'] = df['Low'].diff()
    df['DM_plus'] = np.where((df['DM_plus'] > df['DM_minus']) & (df['DM_plus'] > 0), df['DM_plus'], 0.0)
    df['DM_minus'] = np.where((df['DM_minus'] > df['DM_plus']) & (df['DM_minus'] > 0), df['DM_minus'], 0.0)

    TR_n = df['TR'].rolling(window=window).sum()
    DM_plus_n = df['DM_plus'].rolling(window=window).sum()
    DM_minus_n = df['DM_minus'].rolling(window=window).sum()

    df['DI_plus_n'] = 100 * (DM_plus_n / TR_n)
    df['DI_minus_n'] = 100 * (DM_minus_n / TR_n)

    df['DX'] = (abs(df['DI_plus_n'] - df['DI_minus_n']) / (df['DI_plus_n'] + df['DI_minus_n'])) * 100
    df['ADX'] = df['DX'].rolling(window=window).mean()

    df.drop(['High_Low', 'High_Close_prev', 'Low_Close_prev', 'TR', 'DM_plus', 'DM_minus', 'DI_plus_n', 'DI_minus_n', 'DX'], axis=1, inplace=True)
    return df

def compute_cci(df, window=20):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    TP_MA = TP.rolling(window).mean()
    MD = TP.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (TP - TP_MA) / (0.015 * MD)
    return df

def compute_williams_r(df, window=14):
    high_max = df['High'].rolling(window).max()
    low_min = df['Low'].rolling(window).min()
    df['Williams_%R'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
    return df

def compute_obv(df):
    df['Direction'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    df['Direction'] = np.where(df['Close'] == df['Close'].shift(1), 0, df['Direction'])
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    df.drop(['Direction'], axis=1, inplace=True)
    return df

# Hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }
    rf = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,
        cv=tscv,
        scoring='r2',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    logger.info(f"Best Random Forest Params: {random_search.best_params_}")

    return random_search.best_estimator_

# Hyperparameter tuning for GradientBoostingRegressor using RandomizedSearchCV
def tune_gradient_boosting(X_train, y_train):
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(100, 1000, num=10)],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gb = GradientBoostingRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=50,
        cv=tscv,
        scoring='r2',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    logger.info(f"Best Gradient Boosting Params: {random_search.best_params_}")

    return random_search.best_estimator_

# Hyperparameter tuning for Support Vector Regressor using RandomizedSearchCV
def tune_svr(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto'] + [1e-3, 1e-4],
        'epsilon': [0.1, 0.2, 0.5, 0.3],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svr = SVR()
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=svr,
        param_distributions=param_grid,
        n_iter=50,
        cv=tscv,
        scoring='r2',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    logger.info(f"Best SVR Params: {random_search.best_params_}")

    return random_search.best_estimator_

# Step 2: Machine Learning Model for Stock Price Prediction
def train_model(df, model_name, hyperparams=None):
    stock_model_dir = os.path.join(MODEL_DIR, df.name, model_name)
    os.makedirs(stock_model_dir, exist_ok=True)

    model_file = os.path.join(
        stock_model_dir, f"{df.name}_{model_name}.h5"
    ) if model_name == 'LSTM' else os.path.join(
        stock_model_dir, f"{df.name}_{model_name}.pkl"
    )
    scaler_file = os.path.join(stock_model_dir, f"{df.name}_{model_name}_scaler.pkl")

    df = add_technical_indicators(df)
    df['Prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[FEATURE_COLUMNS]
    y = df['Prediction']

    # Split data
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler to prevent overlapping
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    if model_name == 'LSTM':
        # Save preprocessed data in cache for LSTM
        lstm_data_dir = os.path.join(CACHE_DIR, df.name, 'lstm_data')
        os.makedirs(lstm_data_dir, exist_ok=True)
        np.save(os.path.join(lstm_data_dir, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(lstm_data_dir, 'y_train.npy'), y_train.values)
        np.save(os.path.join(lstm_data_dir, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(lstm_data_dir, 'y_test.npy'), y_test.values)

        # Reshape data for LSTM [samples, timesteps, features]
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        model = train_lstm(
            X_train_lstm, y_train.values,
            X_test_lstm, y_test.values,
            input_shape, model_file
        )
        predictions = model.predict(X_test_lstm).flatten()
    else:
        if os.path.exists(model_file):
            # Load existing model to prevent overlapping
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        else:
            if model_name == 'XGBoost':
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=42
                )
                if hyperparams:
                    model.set_params(**hyperparams)
                model.fit(X_train_scaled, y_train)
            elif model_name == 'Random Forest':
                if hyperparams:
                    model = RandomForestRegressor(random_state=42, **hyperparams)
                else:
                    model = tune_random_forest(X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train)
            elif model_name == 'Gradient Boosting':
                if hyperparams:
                    model = GradientBoostingRegressor(random_state=42, **hyperparams)
                else:
                    model = tune_gradient_boosting(X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train)
            elif model_name == 'Support Vector Regressor':
                if hyperparams:
                    model = SVR(**hyperparams)
                else:
                    model = tune_svr(X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train)
            else:
                ModelClass = AVAILABLE_MODELS.get(model_name)
                if not ModelClass:
                    raise ValueError(f"Model {model_name} is not supported.")
                if hyperparams:
                    model = ModelClass(**hyperparams)
                else:
                    model = ModelClass()
                model.fit(X_train_scaled, y_train)

            # Save model to prevent overlapping
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, mae, mape, r2, scaler

# LSTM Model Training (Unchanged)
def train_lstm(X_train, y_train, X_val, y_val, input_shape, model_file):
    # Check if the model already exists to prevent overlapping
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[es],
            verbose=1
        )
        model.save(model_file)
    return model

# Step 3: Plotting
def plot_data(df, predictions, stock_symbol, model_name):
    actual_prices = df['Close'].tail(len(predictions))
    r2 = r2_score(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    mae = mean_absolute_error(actual_prices, predictions)
    mape = mean_absolute_percentage_error(actual_prices, predictions) * 100

    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-len(predictions):], actual_prices, label='Actual Prices', color='blue')
    plt.plot(df.index[-len(predictions):], predictions, label='Predicted Prices', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{model_name} Prediction vs Actual Prices for {stock_symbol}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Display metrics
    print(f"Model: {model_name}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

# Step 4: Feature Importance Visualization
def plot_feature_importance(model, feature_names, model_name, stock_symbol):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
        plt.title(f'Feature Importances for {model_name} - {stock_symbol}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        QMessageBox.information(None, "Feature Importance", f"Feature importance not available for {model_name}.")

# GUI Application using PyQt6
class StockPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Market Predictor")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()

    def setup_ui(self):
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with tabs
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Prediction Tab
        self.prediction_tab = QWidget()
        self.tabs.addTab(self.prediction_tab, "Prediction")
        self.setup_prediction_tab()

        # Data Tab
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data")
        self.setup_data_tab()

        # Model Comparison Tab
        self.model_comparison_tab = QWidget()
        self.tabs.addTab(self.model_comparison_tab, "Model Comparison")
        self.setup_model_comparison_tab()

    def setup_prediction_tab(self):
        layout = QVBoxLayout(self.prediction_tab)

        # Stock Symbol Input
        stock_layout = QHBoxLayout()
        self.label_stock = QLabel("Enter Stock Symbol(s) (e.g., AAPL, MSFT):")
        self.stock_symbol_input = QLineEdit()
        self.stock_symbol_input.setText("AAPL")
        stock_layout.addWidget(self.label_stock)
        stock_layout.addWidget(self.stock_symbol_input)
        layout.addLayout(stock_layout)

        # Model Selection
        model_layout = QHBoxLayout()
        self.label_model = QLabel("Select Prediction Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        model_layout.addWidget(self.label_model)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Model Parameter Customization (Dynamic)
        self.param_customization_widget = QWidget()
        self.param_layout = QFormLayout()
        self.param_customization_widget.setLayout(self.param_layout)
        layout.addWidget(self.param_customization_widget)

        self.model_combo.currentTextChanged.connect(self.update_model_parameters)
        self.update_model_parameters(self.model_combo.currentText())

        # Buttons
        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_stock_price)
        self.plot_importance_button = QPushButton("Plot Feature Importance")
        self.plot_importance_button.clicked.connect(self.plot_feature_importance_action)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.plot_importance_button)
        layout.addLayout(button_layout)

        # Result Display
        self.result_display = QPlainTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        # Stock Symbol Input
        stock_layout = QHBoxLayout()
        self.label_data_stock = QLabel("Enter Stock Symbol(s) (e.g., AAPL, MSFT):")
        self.data_stock_input = QLineEdit()
        self.data_stock_input.setText("AAPL")
        stock_layout.addWidget(self.label_data_stock)
        stock_layout.addWidget(self.data_stock_input)
        layout.addLayout(stock_layout)

        # Date Range Selection
        date_layout = QHBoxLayout()
        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("Start Date (YYYY-MM-DD)")
        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("End Date (YYYY-MM-DD)")
        date_layout.addWidget(QLabel("Start Date:"))
        date_layout.addWidget(self.start_date_input)
        date_layout.addWidget(QLabel("End Date:"))
        date_layout.addWidget(self.end_date_input)
        layout.addLayout(date_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.fetch_data_button = QPushButton("Fetch Data")
        self.fetch_data_button.clicked.connect(self.fetch_data)
        self.show_last_30_button = QPushButton("Show Last 30 Days")
        self.show_last_30_button.clicked.connect(self.show_last_30_days)
        button_layout.addWidget(self.fetch_data_button)
        button_layout.addWidget(self.show_last_30_button)
        layout.addLayout(button_layout)

        # Data Display
        self.data_display = QPlainTextEdit()
        self.data_display.setReadOnly(True)
        layout.addWidget(self.data_display)

    def setup_model_comparison_tab(self):
        layout = QVBoxLayout(self.model_comparison_tab)

        # Stock Symbol Input
        stock_layout = QHBoxLayout()
        self.label_compare_stock = QLabel("Enter Stock Symbol(s) (e.g., AAPL, MSFT):")
        self.compare_stock_input = QLineEdit()
        self.compare_stock_input.setText("AAPL")
        stock_layout.addWidget(self.label_compare_stock)
        stock_layout.addWidget(self.compare_stock_input)
        layout.addLayout(stock_layout)

        # Compare Button
        self.compare_models_button = QPushButton("Compare Models")
        self.compare_models_button.clicked.connect(self.compare_models)
        layout.addWidget(self.compare_models_button)

        # Comparison Result Display
        self.comparison_display = QPlainTextEdit()
        self.comparison_display.setReadOnly(True)
        layout.addWidget(self.comparison_display)

    def update_model_parameters(self, model_name):
        # Clear previous parameters
        for i in reversed(range(self.param_layout.count())):
            self.param_layout.itemAt(i).widget().setParent(None)

        # Define parameter inputs based on the selected model
        if model_name == "Random Forest":
            self.param_n_estimators = QSpinBox()
            self.param_n_estimators.setRange(10, 1000)
            self.param_n_estimators.setValue(100)
            self.param_max_depth = QSpinBox()
            self.param_max_depth.setRange(1, 100)
            self.param_max_depth.setValue(10)
            self.param_max_depth.setSpecialValueText("None")
            self.param_min_samples_split = QSpinBox()
            self.param_min_samples_split.setRange(2, 100)
            self.param_min_samples_split.setValue(2)
            self.param_min_samples_leaf = QSpinBox()
            self.param_min_samples_leaf.setRange(1, 100)
            self.param_min_samples_leaf.setValue(1)
            self.param_max_features = QComboBox()
            self.param_max_features.addItems(['auto', 'sqrt', 'log2'])

            self.param_layout.addRow("Number of Estimators:", self.param_n_estimators)
            self.param_layout.addRow("Max Depth:", self.param_max_depth)
            self.param_layout.addRow("Min Samples Split:", self.param_min_samples_split)
            self.param_layout.addRow("Min Samples Leaf:", self.param_min_samples_leaf)
            self.param_layout.addRow("Max Features:", self.param_max_features)

        elif model_name == "Gradient Boosting":
            self.param_n_estimators = QSpinBox()
            self.param_n_estimators.setRange(10, 1000)
            self.param_n_estimators.setValue(100)
            self.param_learning_rate = QDoubleSpinBox()
            self.param_learning_rate.setRange(0.001, 1.0)
            self.param_learning_rate.setSingleStep(0.01)
            self.param_learning_rate.setValue(0.1)
            self.param_max_depth = QSpinBox()
            self.param_max_depth.setRange(1, 100)
            self.param_max_depth.setValue(3)
            self.param_subsample = QDoubleSpinBox()
            self.param_subsample.setRange(0.1, 1.0)
            self.param_subsample.setSingleStep(0.1)
            self.param_subsample.setValue(1.0)

            self.param_layout.addRow("Number of Estimators:", self.param_n_estimators)
            self.param_layout.addRow("Learning Rate:", self.param_learning_rate)
            self.param_layout.addRow("Max Depth:", self.param_max_depth)
            self.param_layout.addRow("Subsample:", self.param_subsample)

        elif model_name == "XGBoost":
            self.param_n_estimators = QSpinBox()
            self.param_n_estimators.setRange(10, 1000)
            self.param_n_estimators.setValue(100)
            self.param_max_depth = QSpinBox()
            self.param_max_depth.setRange(1, 100)
            self.param_max_depth.setValue(10)
            self.param_learning_rate = QDoubleSpinBox()
            self.param_learning_rate.setRange(0.001, 1.0)
            self.param_learning_rate.setSingleStep(0.01)
            self.param_learning_rate.setValue(0.01)
            self.param_subsample = QDoubleSpinBox()
            self.param_subsample.setRange(0.1, 1.0)
            self.param_subsample.setSingleStep(0.1)
            self.param_subsample.setValue(0.8)

            self.param_layout.addRow("Number of Estimators:", self.param_n_estimators)
            self.param_layout.addRow("Max Depth:", self.param_max_depth)
            self.param_layout.addRow("Learning Rate:", self.param_learning_rate)
            self.param_layout.addRow("Subsample:", self.param_subsample)

        elif model_name == "Support Vector Regressor":
            self.param_C = QDoubleSpinBox()
            self.param_C.setRange(0.1, 1000.0)
            self.param_C.setSingleStep(1.0)
            self.param_C.setValue(100.0)
            self.param_gamma = QDoubleSpinBox()
            self.param_gamma.setRange(0.001, 1.0)
            self.param_gamma.setSingleStep(0.001)
            self.param_gamma.setValue(0.1)
            self.param_epsilon = QDoubleSpinBox()
            self.param_epsilon.setRange(0.001, 1.0)
            self.param_epsilon.setSingleStep(0.001)
            self.param_epsilon.setValue(0.1)

            self.param_layout.addRow("C:", self.param_C)
            self.param_layout.addRow("Gamma:", self.param_gamma)
            self.param_layout.addRow("Epsilon:", self.param_epsilon)

        elif model_name == "Linear Regression":
            # No hyperparameters to customize
            info_label = QLabel("No hyperparameters to customize for Linear Regression.")
            self.param_layout.addRow(info_label)

        elif model_name == "LSTM":
            # LSTM hyperparameters can be added here if needed
            info_label = QLabel("LSTM parameters are fixed for this application.")
            self.param_layout.addRow(info_label)

    def get_model_hyperparameters(self, model_name):
        if model_name == "Random Forest":
            hyperparams = {
                'n_estimators': self.param_n_estimators.value(),
                'max_depth': self.param_max_depth.value() if self.param_max_depth.value() != 0 else None,
                'min_samples_split': self.param_min_samples_split.value(),
                'min_samples_leaf': self.param_min_samples_leaf.value(),
                'max_features': self.param_max_features.currentText()
            }
            return hyperparams
        elif model_name == "Gradient Boosting":
            hyperparams = {
                'n_estimators': self.param_n_estimators.value(),
                'learning_rate': self.param_learning_rate.value(),
                'max_depth': self.param_max_depth.value(),
                'subsample': self.param_subsample.value()
            }
            return hyperparams
        elif model_name == "XGBoost":
            hyperparams = {
                'n_estimators': self.param_n_estimators.value(),
                'max_depth': self.param_max_depth.value(),
                'learning_rate': self.param_learning_rate.value(),
                'subsample': self.param_subsample.value()
            }
            return hyperparams
        elif model_name == "Support Vector Regressor":
            hyperparams = {
                'C': self.param_C.value(),
                'gamma': self.param_gamma.value(),
                'epsilon': self.param_epsilon.value()
            }
            return hyperparams
        else:
            return None

    def predict_stock_price(self):
        try:
            stock_symbols = self.stock_symbol_input.text().strip().upper()
            if not stock_symbols:
                raise ValueError("Please enter at least one valid stock symbol.")
            stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]

            model_name = self.model_combo.currentText()
            if model_name not in AVAILABLE_MODELS:
                raise ValueError("Selected model is not supported.")

            hyperparams = self.get_model_hyperparameters(model_name)

            results = ""
            for stock_symbol in stock_symbols:
                df = scrape_data(stock_symbol)
                df.name = stock_symbol
                model, mse, mae, mape, r2, scaler = train_model(df, model_name, hyperparams)

                latest_features = df[FEATURE_COLUMNS].tail(1)
                latest_features_scaled = scaler.transform(latest_features)

                if model_name == 'LSTM':
                    latest_features_scaled = latest_features_scaled.reshape((latest_features_scaled.shape[0], 1, latest_features_scaled.shape[1]))
                    predicted_price = model.predict(latest_features_scaled)[0][0]
                else:
                    predicted_price = model.predict(latest_features_scaled)[0]

                result_message = (
                    f"Stock Symbol: {stock_symbol}\n"
                    f"Predicted Next Day Close Price: ${predicted_price:.2f}\n"
                    f"Mean Squared Error (MSE): {mse:.4f}\n"
                    f"Mean Absolute Error (MAE): {mae:.4f}\n"
                    f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
                    f"R² Score: {r2:.4f}\n\n"
                )
                results += result_message

                # Plotting predictions
                split_index = int(len(df) * 0.8)
                plot_features = df[FEATURE_COLUMNS].iloc[split_index:]
                plot_features_scaled = scaler.transform(plot_features)

                if model_name == 'LSTM':
                    plot_features_scaled = plot_features_scaled.reshape((plot_features_scaled.shape[0], 1, plot_features_scaled.shape[1]))
                    plot_predictions = model.predict(plot_features_scaled).flatten()
                else:
                    plot_predictions = model.predict(plot_features_scaled)

                plot_data(df, plot_predictions, stock_symbol, model_name)

            self.result_display.setPlainText(results)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_feature_importance_action(self):
        try:
            stock_symbols = self.stock_symbol_input.text().strip().upper()
            model_name = self.model_combo.currentText()
            if not stock_symbols:
                raise ValueError("Please enter at least one valid stock symbol.")
            stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]

            for stock_symbol in stock_symbols:
                df = scrape_data(stock_symbol)
                df.name = stock_symbol
                hyperparams = self.get_model_hyperparameters(model_name)
                model, _, _, _, _, scaler = train_model(df, model_name, hyperparams)

                if model_name == 'LSTM':
                    QMessageBox.information(self, "Feature Importance", f"Feature importance not available for LSTM on {stock_symbol}.")
                    continue

                plot_feature_importance(model, FEATURE_COLUMNS, model_name, stock_symbol)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def fetch_data(self):
        try:
            stock_symbols = self.data_stock_input.text().strip().upper()
            if not stock_symbols:
                raise ValueError("Please enter at least one valid stock symbol.")
            stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]

            start_date = self.start_date_input.text().strip()
            end_date = self.end_date_input.text().strip()

            combined_df = pd.DataFrame()
            for stock_symbol in stock_symbols:
                df = scrape_data(stock_symbol, start_date if start_date else None, end_date if end_date else None)
                df = df.rename(columns=lambda x: f"{stock_symbol}_{x}")
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = combined_df.join(df, how='outer')

            self.data_display.setPlainText(combined_df.tail(100).to_string())

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_last_30_days(self):
        try:
            stock_symbols = self.data_stock_input.text().strip().upper()
            if not stock_symbols:
                raise ValueError("Please enter at least one valid stock symbol.")
            stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]

            combined_df = pd.DataFrame()
            for stock_symbol in stock_symbols:
                df = scrape_data(stock_symbol)
                df_last_30 = df.tail(30)
                df_last_30 = df_last_30.rename(columns=lambda x: f"{stock_symbol}_{x}")
                if combined_df.empty:
                    combined_df = df_last_30
                else:
                    combined_df = combined_df.join(df_last_30, how='outer')

            self.data_display.setPlainText(combined_df.to_string())

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def compare_models(self):
        try:
            stock_symbols = self.compare_stock_input.text().strip().upper()
            if not stock_symbols:
                raise ValueError("Please enter at least one valid stock symbol.")
            stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]

            results = ""
            for stock_symbol in stock_symbols:
                df = scrape_data(stock_symbol)
                df.name = stock_symbol

                model_results = {}
                for model_name in AVAILABLE_MODELS.keys():
                    hyperparams = self.get_model_hyperparameters(model_name)
                    model, mse, mae, mape, r2, scaler = train_model(df, model_name, hyperparams)
                    model_results[model_name] = {
                        'MSE': mse,
                        'MAE': mae,
                        'MAPE': mape,
                        'R2': r2
                    }

                result_message = f"Stock Symbol: {stock_symbol}\n"
                for model, metrics in model_results.items():
                    result_message += (
                        f"  {model}:\n"
                        f"    MSE: {metrics['MSE']:.4f}\n"
                        f"    MAE: {metrics['MAE']:.4f}\n"
                        f"    MAPE: {metrics['MAPE']:.2f}%\n"
                        f"    R² Score: {metrics['R2']:.4f}\n"
                    )
                result_message += "\n"
                results += result_message

            self.comparison_display.setPlainText(results)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# Entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())
