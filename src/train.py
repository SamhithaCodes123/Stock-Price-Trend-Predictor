# import os
# import pandas as pd
# import numpy as np
# import pickle
# from data import get_nifty50_stocks, load_stock_data
# from features import prepare_features
# from model import StockPredictor
# import warnings
# warnings.filterwarnings('ignore')

# def train_base_model(data_dir="data", models_dir="models", min_data_length=500):
#     """Train base LSTM model on all available stocks"""
#     print("Training base model...")
    
#     if not os.path.exists(models_dir):
#         os.makedirs(models_dir)
    
#     stocks = get_nifty50_stocks()
#     all_X_train, all_y_train = [], []
#     all_X_test, all_y_test = [], []
#     feature_info = None
    
#     successful_stocks = 0
    
#     for stock in stocks[:10]:  # Limit to first 10 for faster training
#         print(f"Processing {stock}...")
        
#         df = load_stock_data(stock, data_dir)
#         if df is None or len(df) < min_data_length:
#             print(f"Insufficient data for {stock}")
#             continue
        
#         # Prepare features
#         data = prepare_features(df)
#         if data is None:
#             continue
        
#         # Store feature info from first successful stock
#         if feature_info is None:
#             feature_info = {
#                 'scaler': data['scaler'],
#                 'feature_cols': data['feature_cols'],
#                 'sequence_length': data['sequence_length']
#             }
        
#         # Accumulate training data
#         all_X_train.append(data['X_train'])
#         all_y_train.append(data['y_train'])
#         all_X_test.append(data['X_test'])
#         all_y_test.append(data['y_test'])
        
#         successful_stocks += 1
#         print(f"Added {stock}: Train={len(data['X_train'])}, Test={len(data['X_test'])}")
    
#     if successful_stocks == 0:
#         print("No sufficient data found for training!")
#         return None
    
#     # Combine all data
#     X_train_combined = np.vstack(all_X_train)
#     y_train_combined = np.hstack(all_y_train)
#     X_test_combined = np.vstack(all_X_test)
#     y_test_combined = np.hstack(all_y_test)
    
#     print(f"Combined training data: {X_train_combined.shape}")
#     print(f"Combined test data: {X_test_combined.shape}")
    
#     # Initialize and train model
#     input_size = X_train_combined.shape[2]
#     predictor = StockPredictor(input_size=input_size, hidden_size=64, num_layers=2)
    
#     # Train model
#     train_losses, test_losses = predictor.train_model(
#         X_train_combined, y_train_combined,
#         X_test_combined, y_test_combined,
#         epochs=50, batch_size=64
#     )
    
#     # Evaluate
#     metrics = predictor.evaluate(X_test_combined, y_test_combined)
#     print(f"Base Model Performance: {metrics}")
    
#     # Save model
#     model_path = os.path.join(models_dir, "base_model.pth")
#     predictor.save_model(model_path, feature_info)
    
#     # Save feature info separately
#     with open(os.path.join(models_dir, "feature_info.pkl"), 'wb') as f:
#         pickle.dump(feature_info, f)
    
#     print(f"Base model training completed using {successful_stocks} stocks")
#     return predictor, feature_info, metrics

# def fine_tune_stock_model(stock_symbol, base_model_path, feature_info, 
#                          data_dir="data", models_dir="models"):
#     """Fine-tune model for specific stock"""
#     print(f"Fine-tuning model for {stock_symbol}...")
    
#     # Load data
#     df = load_stock_data(stock_symbol, data_dir)
#     if df is None or len(df) < 500:
#         print(f"Insufficient data for {stock_symbol}")
#         return None
    
#     # Prepare features using same scaler and feature columns
#     data = prepare_features(df)
#     if data is None:
#         return None
    
#     # Load base model
#     input_size = len(feature_info['feature_cols'])
#     predictor = StockPredictor(input_size=input_size, hidden_size=64, num_layers=2, lr=0.0001)
#     predictor.load_model(base_model_path)
    
#     # Fine-tune with lower learning rate and fewer epochs
#     train_losses, test_losses = predictor.train_model(
#         data['X_train'], data['y_train'],
#         data['X_test'], data['y_test'],
#         epochs=20, batch_size=32
#     )
    
#     # Evaluate
#     metrics = predictor.evaluate(data['X_test'], data['y_test'])
#     print(f"Fine-tuned {stock_symbol} Performance: {metrics}")
    
#     # Save fine-tuned model
#     fine_tuned_dir = os.path.join(models_dir, "fine_tuned")
#     if not os.path.exists(fine_tuned_dir):
#         os.makedirs(fine_tuned_dir)
    
#     model_path = os.path.join(fine_tuned_dir, f"{stock_symbol.replace('.NS', '')}.pth")
#     predictor.save_model(model_path, feature_info)
    
#     return predictor, metrics

# def train_all_models(data_dir="data", models_dir="models"):
#     """Complete training pipeline"""
#     print("Starting complete training pipeline...")
    
#     # Train base model
#     base_model, feature_info, base_metrics = train_base_model(data_dir, models_dir)
    
#     if base_model is None:
#         print("Base model training failed!")
#         return
    
#     # Fine-tune for selected high-cap stocks
#     priority_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
#     base_model_path = os.path.join(models_dir, "base_model.pth")
#     fine_tuned_results = {}
    
#     for stock in priority_stocks:
#         try:
#             result = fine_tune_stock_model(stock, base_model_path, feature_info, data_dir, models_dir)
#             if result:
#                 fine_tuned_results[stock] = result[1]  # Store metrics
#         except Exception as e:
#             print(f"Failed to fine-tune {stock}: {e}")
    
#     print("\n" + "="*50)
#     print("TRAINING COMPLETED")
#     print("="*50)
#     print(f"Base Model Performance: {base_metrics}")
#     print(f"Fine-tuned models: {len(fine_tuned_results)}")
#     for stock, metrics in fine_tuned_results.items():
#         print(f"  {stock}: MAPE = {metrics['MAPE']:.2f}%")
    
#     return base_metrics, fine_tuned_results

# if __name__ == "__main__":
#     # Run complete training
#     train_all_models()

import os
import pandas as pd
import numpy as np
import pickle
from src.data import get_nifty50_stocks, load_stock_data
from src.features import prepare_features
from src.model import StockPredictor
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Path settings
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_base_model(data_dir=DATA_DIR, models_dir=MODELS_DIR, min_data_length=500):
    """Train base LSTM model on all available stocks"""
    print("Training base model...")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    stocks = get_nifty50_stocks()
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    feature_info = None
    successful_stocks = 0

    for stock in stocks[:10]:  # limit to 10 for faster testing
        print(f"Processing {stock}...")

        df = load_stock_data(stock, data_dir)
        if df is None or len(df) < min_data_length:
            print(f"Insufficient data for {stock}")
            continue

        # Prepare features
        data = prepare_features(df)
        if data is None:
            continue

        # Store feature info from first successful stock
        if feature_info is None:
            feature_info = {
                'scaler': data['scaler'],
                'feature_cols': data['feature_cols'],
                'sequence_length': data['sequence_length']
            }

        # Accumulate training data
        all_X_train.append(data['X_train'])
        all_y_train.append(data['y_train'])
        all_X_test.append(data['X_test'])
        all_y_test.append(data['y_test'])

        successful_stocks += 1
        print(f"Added {stock}: Train={len(data['X_train'])}, Test={len(data['X_test'])}")

    if successful_stocks == 0:
        print("No sufficient data found for training!")
        return None

    # Combine all data
    X_train_combined = np.vstack(all_X_train)
    y_train_combined = np.hstack(all_y_train)
    X_test_combined = np.vstack(all_X_test)
    y_test_combined = np.hstack(all_y_test)

    print(f"Combined training data: {X_train_combined.shape}")
    print(f"Combined test data: {X_test_combined.shape}")

    # Initialize and train model
    input_size = X_train_combined.shape[2]
    predictor = StockPredictor(input_size=input_size, hidden_size=64, num_layers=2)

    train_losses, test_losses = predictor.train_model(
        X_train_combined, y_train_combined,
        X_test_combined, y_test_combined,
        epochs=50, batch_size=64
    )

    # Evaluate
    metrics = predictor.evaluate(X_test_combined, y_test_combined)
    print(f"Base Model Performance: {metrics}")

    # Save model + feature info
    model_path = os.path.join(models_dir, "base_model.pth")
    predictor.save_model(model_path, feature_info)

    with open(os.path.join(models_dir, "feature_info.pkl"), 'wb') as f:
        pickle.dump(feature_info, f)

    print(f"Base model training completed using {successful_stocks} stocks")
    return predictor, feature_info, metrics


def fine_tune_stock_model(stock_symbol, base_model_path, feature_info,
                         data_dir=DATA_DIR, models_dir=MODELS_DIR):
    """Fine-tune model for specific stock"""
    print(f"Fine-tuning model for {stock_symbol}...")

    df = load_stock_data(stock_symbol, data_dir)
    if df is None or len(df) < 500:
        print(f"Insufficient data for {stock_symbol}")
        return None

    data = prepare_features(df)
    if data is None:
        return None

    input_size = len(feature_info['feature_cols'])
    predictor = StockPredictor(input_size=input_size, hidden_size=64, num_layers=2, lr=0.0001)
    predictor.load_model(base_model_path)

    train_losses, test_losses = predictor.train_model(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        epochs=20, batch_size=32
    )

    metrics = predictor.evaluate(data['X_test'], data['y_test'])
    print(f"Fine-tuned {stock_symbol} Performance: {metrics}")

    fine_tuned_dir = os.path.join(models_dir, "fine_tuned")
    if not os.path.exists(fine_tuned_dir):
        os.makedirs(fine_tuned_dir)

    model_path = os.path.join(fine_tuned_dir, f"{stock_symbol.replace('.NS', '')}.pth")
    predictor.save_model(model_path, feature_info)

    return predictor, metrics


def train_all_models(data_dir=DATA_DIR, models_dir=MODELS_DIR):
    """Complete training pipeline"""
    print("Starting complete training pipeline...")

    base = train_base_model(data_dir, models_dir)
    if base is None:
        print("Base model training failed!")
        return
    base_model, feature_info, base_metrics = base

    base_model_path = os.path.join(models_dir, "base_model.pth")
    fine_tuned_results = {}
    priority_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

    for stock in priority_stocks:
        try:
            result = fine_tune_stock_model(stock, base_model_path, feature_info, data_dir, models_dir)
            if result:
                fine_tuned_results[stock] = result[1]
        except Exception as e:
            print(f"Failed to fine-tune {stock}: {e}")

    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Base Model Performance: {base_metrics}")
    print(f"Fine-tuned models: {len(fine_tuned_results)}")
    for stock, metrics in fine_tuned_results.items():
        print(f"  {stock}: MAPE = {metrics['MAPE']:.2f}%")

    return base_metrics, fine_tuned_results


if __name__ == "__main__":
    train_all_models()