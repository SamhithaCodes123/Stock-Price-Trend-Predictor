import yfinance as yf
import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
from nsetools import Nse
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Path settings
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------
# Get NIFTY 50 stock list
# ------------------------
def get_nifty50_stocks():
    """Fetch current NIFTY 50 stock list from NSE website"""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/',
    }

    fallback_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
        'MARUTI.NS', 'AXISBANK.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS',
        'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'TITAN.NS', 'NESTLEIND.NS',
        'POWERGRID.NS', 'NTPC.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'M&M.NS',
        'DIVISLAB.NS', 'TECHM.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'IOC.NS',
        'GRASIM.NS', 'DRREDDY.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'ADANIENT.NS',
        'CIPLA.NS', 'HEROMOTOCO.NS', 'APOLLOHOSP.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS',
        'JSWSTEEL.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'ADANIPORTS.NS', 'TATAMOTORS.NS',
        'TATACONSUM.NS', 'BAJAJ-AUTO.NS', 'UPL.NS', 'BPCL.NS', 'TATASTEEL.NS'
    ]

    try:
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            stocks = [
                item['symbol'] + '.NS'
                for item in data['data']
                if not item['symbol'].startswith("NIFTY")   # exclude index entries
            ]
            print("✅ NSE API list used")
            return stocks[:50]

        print("⚠️ NSE API failed, using fallback list")
        return fallback_stocks

    except Exception as e:
        print(f"⚠️ Error fetching NSE list: {e}")
        print("⚠️ Using fallback list")
        return fallback_stocks

# ------------------------
# Download stock data
# ------------------------
def download_stock_data(symbol, start_date="2000-01-01", end_date=None):
    """Download OHLCV data for a stock"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        df.reset_index(inplace=True)   # ensure Date is a column
        df['Symbol'] = symbol.replace('.NS', '')

        return df

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

# ------------------------
# Update pipeline
# ------------------------
def update_data_pipeline(data_dir=DATA_DIR):
    """Update all NIFTY 50 stock data"""
    stocks = get_nifty50_stocks()
    successful_downloads = 0

    print(f"Updating data for {len(stocks)} stocks...")

    for stock in stocks:
        try:
            filepath = os.path.join(data_dir, f"{stock.replace('.NS', '')}.csv")

            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath, parse_dates=["Date"])
                last_date = existing_df["Date"].max()
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                existing_df = pd.DataFrame()
                start_date = "2000-01-01"

            new_df = download_stock_data(stock, start_date)

            if new_df is not None and not new_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset="Date", keep="last", inplace=True)
                combined_df.sort_values("Date", inplace=True)
                combined_df.to_csv(filepath, index=False)
                print(f"✅ {stock}: {len(new_df)} new records")

            successful_downloads += 1

        except Exception as e:
            print(f"❌ Failed {stock}: {e}")

    print(f"✅ Successfully updated {successful_downloads}/{len(stocks)} stocks")
    return successful_downloads

# ------------------------
# Load stock data
# ------------------------

def load_stock_data(symbol, data_dir=DATA_DIR):
    """Load stock data from CSV file"""
    filepath = os.path.join(data_dir, f"{symbol.replace('.NS', '')}.csv")

    if not os.path.exists(filepath):
        print(f"Data file not found for {symbol}. Downloading...")
        df = download_stock_data(symbol)
        if df is not None:
            return df
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    update_data_pipeline()