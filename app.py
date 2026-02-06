import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from scipy.signal import argrelextrema
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Stock Analyst", layout="wide")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
    comp_input = st.text_input("Competitor Ticker (Optional)").upper()
    analyze_btn = st.button("Run Deep Analysis")

# --- CORE LOGIC ---
class StockAnalyzer:
    def __init__(self, ticker_symbol, competitor_symbol=None):
        self.ticker = yf.Ticker(ticker_symbol)
        self.symbol = ticker_symbol
        self.comp_symbol = competitor_symbol
        self.info = self.ticker.info
        self.history = self.ticker.history(period="1y")

    def calculate_technicals(self):
        df = self.history.copy()
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # Support/Resistance
        res_idx = argrelextrema(df.Close.values, np.greater_equal, order=20)[0]
        sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
        
        return {
            "Price": df['Close'].iloc[-1],
            "RSI": df['RSI'].iloc[-1],
            "Resistances": [df.Close.iloc[i] for i in res_idx[-3:]],
            "Supports": [df.Close.iloc[i] for i in sup_idx[-3:]]
        }

    def generate_report(self, api_key):
        genai.configure(api_key=api_key)
        
        # 1. AUTO-DISCOVERY: Find the latest supported Flash model
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods 
            and 'flash' in m.name.lower()
        ]
        
        # Pick the most recent one (usually the last in the list) or use the alias
        # We use 'gemini-flash-latest' as the primary, but fallback to discovery
        target_model = "gemini-flash-latest" if "models/gemini-flash-latest" in available_models else available_models[-1]
        
        model = genai.GenerativeModel(target_model)
        
        # ... rest of your prompt and content generation ...
        tech = self.calculate_technicals()
        news = [n.get('title') for n in self.ticker.news[:8]]
        
        prompt = f"Analyze {self.symbol}..." # (Your existing prompt here)
        
        return model.generate_content(prompt).text

# --- APP UI ---
st.title("🤖 AI-Powered Stock Intelligence")

if analyze_btn:
    if not api_key:
        st.error("Please provide a Gemini API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input}..."):
                analyzer = StockAnalyzer(ticker_input, comp_input)
                
                # Layout: Two columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📈 {ticker_input} Price Action")
                    fig = go.Figure(data=[go.Candlestick(x=analyzer.history.index,
                                    open=analyzer.history['Open'],
                                    high=analyzer.history['High'],
                                    low=analyzer.history['Low'],
                                    close=analyzer.history['Close'])])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Key Metrics")
                    metrics = analyzer.info
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Current Price", f"${metrics.get('currentPrice', 'N/A')}")
                    m_col1.metric("Forward P/E", metrics.get('forwardPE', 'N/A'))
                    m_col2.metric("Market Cap", f"{metrics.get('marketCap', 0):,}")
                    m_col2.metric("Target Price", f"${metrics.get('targetMeanPrice', 'N/A')}")

                st.divider()
                st.subheader("🧠 Gemini Intelligence Report")
                report = analyzer.generate_report(api_key)
                st.markdown(report)
                
        except Exception as e:
            st.error(f"Error fetching data for {ticker_input}: {e}")
