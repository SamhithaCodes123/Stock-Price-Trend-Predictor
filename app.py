import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import get_nifty50_stocks, load_stock_data, update_data_pipeline
from src.inference import StockPricePredictor
from src.features import add_technical_indicators

# Page config
st.set_page_config(
    page_title="NIFTY 50 Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_list():
    """Get cached stock list"""
    return get_nifty50_stocks()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_cached_stock_data(symbol):
    """Load cached stock data"""
    return load_stock_data(symbol)

def plot_stock_chart(df, symbol, days=30):
    """Create interactive stock price chart"""
    # Get recent data
    recent_df = df.tail(days).copy()
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=recent_df.index,
        open=recent_df['Open'],
        high=recent_df['High'],
        low=recent_df['Low'],
        close=recent_df['Close'],
        name=symbol
    ))
    
    # Add moving averages
    recent_df_with_indicators = add_technical_indicators(recent_df)
    
    if 'SMA_20' in recent_df_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=recent_df_with_indicators.index,
            y=recent_df_with_indicators['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'EMA_12' in recent_df_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=recent_df_with_indicators.index,
            y=recent_df_with_indicators['EMA_12'],
            mode='lines',
            name='EMA 12',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title=f'{symbol} - Last {days} Days',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_technical_indicators(df, days=30):
    """Plot technical indicators"""
    recent_df = add_technical_indicators(df.tail(days))
    
    fig = go.Figure()
    
    # RSI
    if 'RSI' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['RSI'],
            mode='lines',
            name='RSI',
            yaxis='y2'
        ))
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought", yref='y2')
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold", yref='y2')
    
    # MACD
    if 'MACD' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['MACD'],
            mode='lines',
            name='MACD',
            yaxis='y3'
        ))
    
    fig.update_layout(
        title='Technical Indicators',
        xaxis_title='Date',
        template='plotly_white',
        height=400,
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
        yaxis3=dict(title="MACD", overlaying="y", side="left"),
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 NIFTY 50 Stock Price Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("### AI-Powered Next-Day Price Prediction using LSTM Neural Networks")
    
    # Sidebar
    st.sidebar.header("🎯 Stock Selection")
    
    # Check if models exist
    models_exist = (os.path.exists("models/base_model.pth") and 
                   os.path.exists("models/feature_info.pkl"))
    
    if not models_exist:
        st.error("⚠️ Models not found! Please run the training script first.")
        st.code("python src/train.py")
        st.stop()
    
    # Load stock list
    try:
        stocks = get_stock_list()
        stock_names = [s.replace('.NS', '') for s in stocks]
        
        selected_stock_name = st.sidebar.selectbox(
            "Choose a NIFTY 50 Stock:",
            stock_names,
            index=0
        )
        
        selected_stock = selected_stock_name + '.NS'
        
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        st.stop()
    
    # Sidebar options
    st.sidebar.header("📊 Display Options")
    chart_days = st.sidebar.slider("Chart Days", 15, 90, 30)
    show_indicators = st.sidebar.checkbox("Show Technical Indicators", True)
    show_backtest = st.sidebar.checkbox("Show Backtest Results", False)
    
    # Data refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        with st.spinner("Updating data..."):
            update_data_pipeline()
        st.sidebar.success("Data refreshed!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {selected_stock_name} Stock Analysis")
        
        # Load stock data
        try:
            df = load_cached_stock_data(selected_stock)
            if df is None or df.empty:
                st.error(f"No data available for {selected_stock_name}")
                st.stop()
            
            # Display current info
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # Current price display
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("💰 Current Price", f"₹{current_price:.2f}")
            
            with col_b:
                st.metric("📊 Change", f"₹{change:+.2f}", f"{change_pct:+.2f}%")
            
            with col_c:
                st.metric("📈 High", f"₹{df['High'].iloc[-1]:.2f}")
            
            with col_d:
                st.metric("📉 Low", f"₹{df['Low'].iloc[-1]:.2f}")
            
            # Stock chart
            fig_chart = plot_stock_chart(df, selected_stock_name, chart_days)
            st.plotly_chart(fig_chart, use_container_width=True)
            
            # Technical indicators
            if show_indicators:
                st.subheader("🔍 Technical Indicators")
                fig_indicators = plot_technical_indicators(df, chart_days)
                st.plotly_chart(fig_indicators, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    with col2:
        st.subheader("🔮 AI Prediction")
        
        # Initialize predictor
        try:
            predictor = StockPricePredictor("models")
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction_result = predictor.predict_next_day_price(selected_stock)
            
            if prediction_result:
                # Prediction display
                pred_price = prediction_result['predicted_price']
                pred_change = prediction_result['price_change']
                pred_change_abs = prediction_result['price_change_abs']
                model_type = prediction_result['model_type']
                
                # Prediction box
                if pred_change >= 0:
                    trend_emoji = "📈"
                    color = "green"
                else:
                    trend_emoji = "📉" 
                    color = "red"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>{trend_emoji} Tomorrow's Prediction</h3>
                    <h2>₹{pred_price:.2f}</h2>
                    <p style="font-size: 1.2rem;">
                        <span style="color: {'lightgreen' if pred_change >= 0 else 'lightcoral'};">
                            {pred_change:+.2f}% (₹{pred_change_abs:+.2f})
                        </span>
                    </p>
                    <p><small>Model: {model_type.replace('_', ' ').title()}</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown("#### 📋 Prediction Details")
                
                st.info(f"**Current Price:** ₹{current_price:.2f}")
                st.info(f"**Predicted Price:** ₹{pred_price:.2f}")
                
                if pred_change >= 0:
                    st.success(f"**Expected Gain:** +₹{pred_change_abs:.2f} ({pred_change:+.2f}%)")
                else:
                    st.error(f"**Expected Loss:** ₹{pred_change_abs:.2f} ({pred_change:+.2f}%)")
                
                st.info(f"**Model Type:** {model_type.replace('_', ' ').title()}")
                
                # Confidence indicator
                confidence_level = min(100, max(60, 90 - abs(pred_change) * 2))
                st.progress(confidence_level/100)
                st.caption(f"Confidence Level: {confidence_level:.0f}%")
                
            else:
                st.error("❌ Could not generate prediction. Please check if sufficient data is available.")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    # Backtest section
    if show_backtest and prediction_result:
        st.subheader("🎯 Model Performance (Backtest)")
        
        with st.spinner("Running backtest..."):
            backtest_result = predictor.backtest_model(selected_stock, days=20)
        
        if backtest_result:
            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            
            with col_bt1:
                st.metric("MAPE", f"{backtest_result['mape']:.2f}%")
            
            with col_bt2:
                st.metric("Direction Accuracy", f"{backtest_result['accuracy_direction']:.1f}%")
            
            with col_bt3:
                st.metric("RMSE", f"₹{backtest_result['rmse']:.2f}")
            
            with col_bt4:
                st.metric("MAE", f"₹{backtest_result['mae']:.2f}")
            
            # Backtest chart
            backtest_df = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now().date(), periods=len(backtest_result['actual'])),
                'Actual': backtest_result['actual'],
                'Predicted': backtest_result['predictions']
            })
            
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(
                x=backtest_df['Date'],
                y=backtest_df['Actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue')
            ))
            fig_backtest.add_trace(go.Scatter(
                x=backtest_df['Date'],
                y=backtest_df['Predicted'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red')
            ))
            
            fig_backtest.update_layout(
                title='Backtest: Predicted vs Actual Prices',
                yaxis_title='Price (₹)',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_backtest, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚠️ <strong>Disclaimer:</strong> 
        Stock predictions are not investment advice. Past performance does not guarantee future results.</p>
        <p>📊 Powered by Neural Networks & Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()