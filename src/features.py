import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    """Add technical indicators to stock dataframe"""

    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Bollinger Bands
    boll = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = boll.bollinger_hband()
    df['BB_Low'] = boll.bollinger_lband()

    # Volume SMA (pandas instead of broken ta.volume.volume_sma)
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()

    df = df.dropna()
    return df


def prepare_features(df, sequence_length=60, test_size=0.2):
    """Prepare sequences for training LSTM"""

    df = add_technical_indicators(df)

    feature_cols = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD',
                    'MACD_Signal', 'BB_High', 'BB_Low', 'Volume_SMA']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])  # predict closing price

    X, y = np.array(X), np.array(y)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'sequence_length': sequence_length
    }

def prepare_prediction_data(df, scaler, feature_cols, sequence_length=60):
    """Prepare data for making predictions"""
    df_features = add_technical_indicators(df)
    df_clean = df_features[feature_cols].dropna()
    
    if len(df_clean) < sequence_length:
        print(f"Insufficient data for prediction: {len(df_clean)} rows needed: {sequence_length}")
        return None
    
    # Scale the data
    scaled_data = scaler.transform(df_clean)
    
    # Get the last sequence for prediction
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(feature_cols))
    
    return last_sequence, df_clean.iloc[-1]['Close']