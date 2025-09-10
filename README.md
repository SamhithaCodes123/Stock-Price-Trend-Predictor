# 📈 NIFTY 50 Stock Price Predictor

An AI-powered stock market prediction system using LSTM neural networks to predict next-day closing prices for NIFTY 50 stocks.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.1.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **Dynamic NIFTY 50 Handling**: Automatically updates stock list when composition changes
- **Hybrid LSTM Models**: Base model for all stocks + fine-tuned models for key stocks
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, ATR
- **Interactive Dashboard**: Streamlit web app with real-time predictions
- **Automated Pipeline**: Scheduled data updates and model retraining
- **Comprehensive Analysis**: Jupyter notebook with EDA and visualizations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Feature Eng.   │    │  LSTM Models    │
│                 │    │                 │    │                 │
│ • Yahoo Finance │───▶│ • Technical     │───▶│ • Base Model    │
│ • NIFTY 50 List │    │   Indicators    │    │ • Fine-tuned    │
│ • Auto Updates  │    │ • Normalization │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Predictions   │    │   Inference     │
│                 │    │                 │    │                 │
│ • Stock Charts  │◀───│ • Next-day      │◀───│ • Model Loading │
│ • Predictions   │    │   Price         │    │ • Data Prep     │
│ • Backtesting   │    │ • Confidence    │    │ • Prediction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd stock-trend-predictor
pip install -r requirements.txt
```

### 2. Download Data & Train Models

```bash
# Download historical data for all NIFTY 50 stocks
python src/data.py

# Train base model and fine-tuned models
python src/train.py
```

### 3. Run the Web App

```bash
streamlit run app.py
```

### 4. Explore Data Analysis

```bash
jupyter notebook Stock_Analysis_EDA.ipynb
```

## 📁 Project Structure

```
stock-trend-predictor/
├── src/
│   ├── data.py           # Data fetching and management
│   ├── features.py       # Technical indicators and feature engineering
│   ├── model.py          # LSTM model architecture
│   ├── train.py          # Training pipeline
│   └── inference.py      # Prediction pipeline
├── models/
│   ├── base_model.pth    # Base LSTM model
│   ├── feature_info.pkl  # Feature preprocessing info
│   └── fine_tuned/       # Fine-tuned models for key stocks
│       ├── RELIANCE.pth
│       └── TCS.pth
├── data/                 # Stock data CSV files
│   ├── RELIANCE.csv
│   └── TCS.csv
├── app.py               # Streamlit web application
├── update_pipeline.py   # Automated update script
├── Stock_Analysis_EDA.ipynb  # Jupyter notebook for analysis
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Technical Details

### LSTM Model Architecture

```python
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        # 2-layer LSTM with dropout
        # Dense layer for final prediction
```

### Features Used

- **Price Data**: Open, High, Low, Close, Volume
- **Moving Averages**: SMA (5, 10, 20), EMA (12, 26)
- **Momentum**: RSI (14), MACD, MACD Signal
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume SMA, Volume ratios
- **Price Ratios**: High-Low %, Close-Open %

### Hybrid Approach

1. **Base Model**: Trained on pooled data from all NIFTY 50 stocks
2. **Fine-tuned Models**: Specialized models for high-cap stocks (RELIANCE, TCS, HDFCBANK, etc.)
3. **Smart Selection**: System automatically chooses the best available model for predictions

## 📊 Performance Metrics

The models are evaluated using:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **Direction Accuracy**: Percentage of correct trend predictions

## 🔄 Automation

### Scheduled Updates

```bash
# Update data only
python update_pipeline.py --data-only

# Full update with model retraining if needed
python update_pipeline.py

# Force model retraining
python update_pipeline.py --force-retrain
```

### Cron Job Setup

```bash
# Add to crontab for daily updates at 6 PM
0 18 * * * /path/to/python /path/to/update_pipeline.py --data-only
```

## 📈 Usage Examples

### Making Predictions

```python
from src.inference import StockPricePredictor

predictor = StockPricePredictor()
result = predictor.predict_next_day_price("RELIANCE.NS")

print(f"Current: ₹{result['current_price']:.2f}")
print(f"Predicted: ₹{result['predicted_price']:.2f}")
print(f"Change: {result['price_change']:+.2f}%")
```

### Backtesting

```python
backtest = predictor.backtest_model("RELIANCE.NS", days=30)
print(f"MAPE: {backtest['mape']:.2f}%")
print(f"Direction Accuracy: {backtest['accuracy_direction']:.2f}%")
```

## 🎨 Web Interface Features

- **Stock Selection**: Dropdown with current NIFTY 50 stocks
- **Interactive Charts**: Candlestick charts with technical indicators
- **Predictions**: Next-day price with confidence levels
- **Backtesting**: Historical model performance
- **Real-time Updates**: Automatic data refresh capability

## ⚠️ Important Notes

### Disclaimers

- **Educational Purpose**: This project is for learning and research only
- **Not Financial Advice**: Do not use predictions for actual trading decisions
- **Market Risks**: Stock markets are inherently unpredictable
- **Past Performance**: Historical results don't guarantee future performance

### Limitations

- Predictions are based on technical analysis only
- Fundamental factors and news events are not considered
- Market conditions can change rapidly
- Model accuracy may vary across different stocks and market conditions

## 🐛 Troubleshooting

### Common Issues

1. **Yahoo Finance API Errors**
   ```bash
   # Update yfinance package
   pip install --upgrade yfinance
   ```

2. **Model Loading Errors**
   ```bash
   # Retrain models
   python src/train.py
   ```

3. **Data Loading Issues**
   ```bash
   # Clear data and re-download
   rm -rf data/
   python src/data.py
   ```

## 🔮 Future Enhancements

- [ ] Integration with fundamental analysis
- [ ] Sentiment analysis from news and social media
- [ ] Portfolio optimization features
- [ ] Mobile app development
- [ ] Real-time streaming predictions
- [ ] Integration with trading platforms
- [ ] Alternative model architectures (Transformer, GRU)

## 📚 Resources

- [Yahoo Finance API Documentation](https://python-yahoofinance.readthedocs.io/)
- [PyTorch LSTM Guide](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data
- NSE for NIFTY 50 index composition
- PyTorch team for the deep learning framework
- Streamlit team for the web app framework

---

⭐ **If you found this project helpful, please give it a star!** ⭐