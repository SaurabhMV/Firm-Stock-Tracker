import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Pro Stock Analyst", layout="wide")

# --- SIDEBAR: AUTO-MODEL DISCOVERY ---
with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password")
    ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
    analyze_btn = st.button("Generate Deep Research")

def get_analyst_data(ticker):
    """Fetches Analyst recommendations and targets."""
    info = ticker.info
    current = info.get('currentPrice', 1)
    target = info.get('targetMeanPrice', current)
    upside = ((target - current) / current) * 100
    
    # Summary of consensus (e.g., 'buy', 'hold')
    rec = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
    return {"Target": target, "Upside": upside, "Consensus": rec}

def generate_pro_report(symbol, info, tech, news, api_key):
    genai.configure(api_key=api_key)
    # Using the 'latest' alias as discussed
    model = genai.GenerativeModel("gemini-2.5-flash") 
    
    prompt = f"""
    Act as a Hedge Fund Strategy Lead. Analyze {symbol} for a long-term investor.
    
    CONTEXT:
    - Business: {info.get('longBusinessSummary')[:1000]}
    - Fundamentals: P/E {info.get('forwardPE')}, Margin {info.get('profitMargins')}, Debt/Equity {info.get('debtToEquity')}
    - Technicals: RSI {tech['RSI']:.2f}, Supports {tech['Supports']}
    - Recent Headlines: {news}

    YOUR GOAL: Provide a balanced investment thesis.
    
    STRUCTURE:
    1. THE BULL CASE: Why buy now? (Key growth drivers)
    2. THE BEAR CASE: What could go wrong? (Regulatory, competition, macro)
    3. VALUATION: Is it overvalued based on its P/E and growth?
    4. RISK RATING: Rate from 1 (Low) to 10 (Speculative).
    5. FINAL VERDICT: A direct Buy/Hold/Sell suggestion with a 'Confidence Score' (0-100%).
    """
    return model.generate_content(prompt).text

# --- APP UI ---
st.title("🏛️ Professional Equity Research Dashboard")

if analyze_btn:
    if not api_key:
        st.error("API Key Required")
    else:
        with st.spinner("Synthesizing market data..."):
            ticker = yf.Ticker(ticker_input)
            analyst = get_analyst_data(ticker)
            
            # --- TOP ROW: KPI CARDS ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${ticker.info.get('currentPrice')}")
            c2.metric("Target Price", f"${analyst['Target']:.2f}", f"{analyst['Upside']:.1f}% Upside")
            c3.metric("Wall St. Consensus", analyst['Consensus'])
            c4.metric("Market Cap", f"{ticker.info.get('marketCap', 0):,}")

            # --- TABS FOR BETTER UX ---
            tab1, tab2, tab3 = st.tabs(["📊 Charts & Data", "🧠 AI Thesis", "📰 News Sentiment"])

            with tab1:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    hist = ticker.history(period="1y")
                    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    st.write("**Key Financial Health**")
                    st.json({
                        "Profit Margin": ticker.info.get("profitMargins"),
                        "Revenue Growth": ticker.info.get("revenueGrowth"),
                        "Return on Equity": ticker.info.get("returnOnEquity")
                    })

            with tab2:
                # Mock technicals for the prompt
                tech_data = {"RSI": 55, "Supports": [140, 142], "Price": ticker.info.get('currentPrice')}
                news_titles = [n.get('title') for n in ticker.news[:5]]
                
                report = generate_pro_report(ticker_input, ticker.info, tech_data, news_titles, api_key)
                st.markdown(report)

            with tab3:
                st.subheader("Latest Market Buzz")
                for n in ticker.news[:5]:
                    st.write(f"🔗 [{n['title']}]({n['link']})")
