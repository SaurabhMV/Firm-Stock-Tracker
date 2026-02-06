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
    
    # --- DYNAMIC DISCOVERY ---
    try:
        # Get all models that support generating content and are "Flash" models
        models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods 
            and 'flash' in m.name.lower()
        ]
        # Sort to ensure we get the highest version number (e.g., Gemini 3 over 2.5)
        models.sort()
        # Use the newest one found, or fallback to the universal alias
        target_model = models[-1] if models else "models/gemini-flash-latest"
    except Exception:
        target_model = "models/gemini-flash-latest"

    model = genai.GenerativeModel(target_model)
    
    # --- PREPARE PROMPT ---
    prompt = f"""
    Act as a Hedge Fund Strategy Lead. Analyze {symbol} for a long-term investor.
    Context: {info.get('longBusinessSummary')[:1000]}
    Technicals: RSI {tech['RSI']:.2f}, Supports {tech['Supports']}
    News: {news}
    Provide: 1. Bull Case, 2. Bear Case, 3. Risk Rating (1-10), 4. Final Buy/Hold/Sell Verdict.
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

            import re

            def sanitize_link(link):
                """Ensures the link is a valid absolute URL."""
                if not link:
                    return "#"
                # If it starts with a tracking redirect, you could try to strip it,
                # but usually just ensuring it's absolute helps.
                if link.startswith('/'):
                    return f"https://finance.yahoo.com{link}"
                return link
            
            # ... inside your Tab 3 logic ...
            with tab3:
            # Line 116: Everything below MUST be indented 4 spaces
            st.subheader("📰 Market-Moving News Feed")
            
            news_items = ticker.news[:8]
            if not news_items:
                st.info("No recent news found for this ticker.")
            else:
                # Note: Nested blocks (like for loops) need another level of indent!
                for n in news_items:
                    title = n.get('title', 'No Title')
                    raw_link = n.get('link', '#')
                    # Ensure link is absolute
                    clean_link = raw_link if raw_link.startswith('http') else f"https://finance.yahoo.com{raw_link}"
                    
                    # 2. Get Thumbnail (looking for the highest quality resolution)
                    thumbnail_url = None
                    if 'thumbnail' in n and 'resolutions' in n['thumbnail']:
                        # Usually, the last resolution in the list is the highest quality
                        thumbnail_url = n['thumbnail']['resolutions'][-1].get('url')
        
                    # 3. UI: Two Columns (Small image on left, text on right)
                    col_img, col_txt = st.columns([1, 4])
                    
                    with col_img:
                        if thumbnail_url:
                            st.image(thumbnail_url, use_container_width=True)
                        else:
                            st.write("🖼️") # Fallback icon if no image
                    
                    with col_txt:
                        st.markdown(f"**[{title}]({clean_link})**")
                        # Show source and time
                        pub = n.get('publisher', 'Finance')
                        st.caption(f"Source: {pub}")
                        st.link_button("Read Article", clean_link, size="small")
                    
                    st.divider()
