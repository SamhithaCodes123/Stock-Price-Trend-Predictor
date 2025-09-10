import os
import sys
import pickle
import numpy as np
import pandas as pd
from .data import load_stock_data
from .features import prepare_prediction_data, add_technical_indicators
from .model import StockPredictor
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Path settings
# ------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)


class StockPricePredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.base_model = None
        self.feature_info = None
        self.load_feature_info()
        
    def load_feature_info(self):
        """Load feature information"""
        feature_info_path = os.path.join(self.models_dir, "feature_info.pkl")
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'rb') as f:
                self.feature_info = pickle.load(f)
        else:
            print("Feature info not found!")
    
    def load_base_model(self):
        """Load base LSTM model"""
        if self.base_model is None and self.feature_info:
            base_model_path = os.path.join(self.models_dir, "base_model.pth")
            if os.path.exists(base_model_path):
                input_size = len(self.feature_info['feature_cols'])
                self.base_model = StockPredictor(input_size=input_size)
                self.base_model.load_model(base_model_path)
                print("Base model loaded successfully")
            else:
                print("Base model not found!")
    
    def get_model_for_stock(self, symbol):
        """Get the best available model for a stock (fine-tuned or base)"""
        # Check for fine-tuned model
        fine_tuned_path = os.path.join(
            self.models_dir, "fine_tuned", f"{symbol.replace('.NS', '')}.pth"
        )
        
        if os.path.exists(fine_tuned_path):
            input_size = len(self.feature_info['feature_cols'])
            model = StockPredictor(input_size=input_size)
            model.load_model(fine_tuned_path)
            return model, "fine_tuned"
        else:
            # Use base model
            if self.base_model is None:
                self.load_base_model()
            return self.base_model, "base"
    
    def predict_next_day_price(self, symbol, data_dir="data"):
        """Predict next day closing price for a stock"""
        if self.feature_info is None:
            print("Feature info not loaded!")
            return None
        
        # Load stock data
        df = load_stock_data(symbol, data_dir)
        if df is None or len(df) < self.feature_info['sequence_length']:
            print(f"Insufficient data for {symbol}")
            return None
        
        # Get appropriate model
        model, model_type = self.get_model_for_stock(symbol)
        if model is None:
            print(f"No model available for {symbol}")
            return None
        
        # Prepare prediction data
        pred_data = prepare_prediction_data(
            df, 
            self.feature_info['scaler'],
            self.feature_info['feature_cols'],
            self.feature_info['sequence_length']
        )
        
        if pred_data is None:
            return None
        
        last_sequence, current_price = pred_data
        
        # Make prediction (normalized)
        predicted_normalized = model.predict(last_sequence)
        
        # Denormalize prediction
        # Find the 'Close' column index
        close_idx = self.feature_info['feature_cols'].index('Close')
        
        # Create dummy array for inverse transform
        dummy = np.zeros((1, len(self.feature_info['feature_cols'])))
        dummy[0, close_idx] = predicted_normalized[0, 0]
        
        # Inverse transform
        predicted_price = self.feature_info['scaler'].inverse_transform(dummy)[0, close_idx]
        
        # Calculate percentage change
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_abs': predicted_price - current_price,
            'model_type': model_type,
            'symbol': symbol.replace('.NS', '')
        }
    
    def get_prediction_confidence(self, symbol, data_dir="data", n_predictions=10):
        """Get prediction confidence using multiple forward passes"""
        results = []
        
        for _ in range(n_predictions):
            pred = self.predict_next_day_price(symbol, data_dir)
            if pred:
                results.append(pred['predicted_price'])
        
        if not results:
            return None
        
        mean_pred = np.mean(results)
        std_pred = np.std(results)
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'confidence_interval_95': (mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred)
        }
    
    def backtest_model(self, symbol, days=30, data_dir="data"):
        """Simple backtest for model accuracy"""
        df = load_stock_data(symbol, data_dir)
        if df is None or len(df) < days + self.feature_info['sequence_length']:
            return None
        
        # Use last 'days' for testing
        test_df = df.iloc[:-days].copy()
        actual_prices = df.iloc[-days:]['Close'].values
        predictions = []
        
        for i in range(days):
            # Prepare data up to day i
            current_df = pd.concat([test_df, df.iloc[-days+i:-days+i+1]])
            
            pred_data = prepare_prediction_data(
                current_df,
                self.feature_info['scaler'],
                self.feature_info['feature_cols'],
                self.feature_info['sequence_length']
            )
            
            if pred_data is None:
                continue
            
            # Get model and predict
            model, _ = self.get_model_for_stock(symbol)
            if model is None:
                continue
            
            last_sequence, _ = pred_data
            predicted_normalized = model.predict(last_sequence)
            
            # Denormalize
            close_idx = self.feature_info['feature_cols'].index('Close')
            dummy = np.zeros((1, len(self.feature_info['feature_cols'])))
            dummy[0, close_idx] = predicted_normalized[0, 0]
            predicted_price = self.feature_info['scaler'].inverse_transform(dummy)[0, close_idx]
            
            predictions.append(predicted_price)
        
        if not predictions:
            return None
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual_prices = actual_prices[:len(predictions)]
        
        mae = np.mean(np.abs(predictions - actual_prices))
        rmse = np.sqrt(np.mean((predictions - actual_prices)**2))
        mape = np.mean(np.abs((predictions - actual_prices) / actual_prices)) * 100
        
        return {
            'predictions': predictions,
            'actual': actual_prices,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy_direction': np.mean(
                np.sign(predictions[1:] - actual_prices[:-1]) == 
                np.sign(actual_prices[1:] - actual_prices[:-1])
            ) * 100
        }

def predict_stock_price(symbol, models_dir="models", data_dir="data"):
    """Standalone function to predict stock price"""
    predictor = StockPricePredictor(models_dir)
    return predictor.predict_next_day_price(symbol, data_dir)

if __name__ == "__main__":
    # Example usage
    predictor = StockPricePredictor()
    
    # Test prediction
    result = predictor.predict_next_day_price("RELIANCE.NS")
    if result:
        print(f"Stock: {result['symbol']}")
        print(f"Current Price: ₹{result['current_price']:.2f}")
        print(f"Predicted Price: ₹{result['predicted_price']:.2f}")
        print(f"Change: {result['price_change']:+.2f}% (₹{result['price_change_abs']:+.2f})")
        print(f"Model: {result['model_type']}")
    
    # Test backtest
    backtest = predictor.backtest_model("RELIANCE.NS", days=10)
    if backtest:
        print(f"\nBacktest Results:")
        print(f"MAPE: {backtest['mape']:.2f}%")
        print(f"Direction Accuracy: {backtest['accuracy_direction']:.2f}%")