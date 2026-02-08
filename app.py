import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import google.generativeai as genai
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- APP CONFIG ---
st.set_page_config(page_title="Firm Stock Tracker", page_icon="📈")

# Android "Add to Home Screen" support
st.markdown("""
    <head>
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="application-name" content="Stock Tracker">
    </head>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS & API KEY ---
with st.sidebar:
    st.title("Settings ⚙️")
    
    # MANUAL API KEY OPTION
    api_key_manual = st.text_input(
        "Gemini API Key", 
        type="password", 
        placeholder="Paste your key here...",
        help="Entering a key here overrides st.secrets."
    )
    
    # Fallback logic: Priority given to manual input
    api_key = api_key_manual or st.secrets.get("GOOGLE_API_KEY", "")
    
    # Model selection
    model_options = {
        "Gemini 3 Pro (Smartest)": "gemini-3-pro-preview",
        "Gemini 3 Flash (Fast & Smart)": "gemini-3-flash-preview",
        "Gemini 2.5 Pro (Balanced)": "gemini-2.5-pro",
        "Gemini 2.5 Flash (Lightweight)": "gemini-2.5-flash"
    }
    
    selected_model_display = st.selectbox(
        "Choose Gemini Model:",
        options=list(model_options.keys()),
        index=1
    )
    selected_model_id = model_options[selected_model_display]
    
    if not api_key:
        st.warning("⚠️ No API Key detected. Please enter one above.")
    else:
        st.success("✅ API Key loaded.")

# --- MAIN UI INPUTS ---
ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
analyze_btn = st.button("Generate Deep Research")

# --- HELPER FUNCTIONS ---

def get_analyst_data(ticker):
    """Fetches Analyst recommendations and targets safely."""
    info = ticker.info
    current = info.get('currentPrice')
    target = info.get('targetMeanPrice')
    upside = ((target - current) / current) * 100 if (current and target) else 0
    rec = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
    return {"Target": target, "Upside": upside, "Consensus": rec}

def calculate_technicals(history):
    """Calculates RSI and Support/Resistance levels."""
    df = history.copy()
    if df.empty:
        return {"RSI": 50, "Supports": [], "Price": 0}
        
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [df.Close.iloc[i] for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    
    return {"RSI": df['RSI'].iloc[-1], "Supports": supports, "Price": df['Close'].iloc[-1]}

def generate_pro_report(symbol, info, tech, news, key, model_id):
    """Generates the AI analysis."""
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_id)
    prompt = f"""
    Act as a Hedge Fund Strategy Lead. Analyze {symbol} for a long-term investor.
    - Fundamentals: P/E {info.get('forwardPE')}, Margin {info.get('profitMargins')}
    - Technicals: RSI {tech.get('RSI', 50):.2f}, Key Support Levels: {tech.get('Supports')}
    - News Headlines: {news}
    Provide a Bull Case, Bear Case, and a Final Verdict.
    """
    return model.generate_content(prompt).text

# --- APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input}..."):
                ticker = yf.Ticker(ticker_input)
                history = ticker.history(period="1y")
                
                if history.empty:
                    st.error("Could not fetch data. Check the ticker symbol.")
                    st.stop()

                info = ticker.info
                analyst = get_analyst_data(ticker)
                tech_data = calculate_technicals(history)
                
                # Snapshot Metrics
                st.subheader(f"📊 {ticker_input} Snapshot")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"${info.get('currentPrice', 0):.2f}")
                c2.metric("Target Price", f"${analyst['Target'] or 'N/A'}", f"{analyst['Upside']:.1f}%")
                c3.metric("Consensus", analyst['Consensus'])
                c4.metric("RSI (14)", f"{tech_data['RSI']:.1f}")

                tab1, tab2, tab3 = st.tabs(["📈 Charts", "🧠 AI Thesis", "📰 News"])
                
                with tab1:
                    fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], 
                                    high=history['High'], low=history['Low'], close=history['Close'])])
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    news_titles = [n.get('title') for n in ticker.news[:5]]
                    report = generate_pro_report(ticker_input, info, tech_data, news_titles, api_key, selected_model_id)
                    st.markdown(report)
                
                with tab3:
                    for n in ticker.news[:5]:
                        st.write(f"**{n.get('title')}**")
                        st.caption(f"[Read Article]({n.get('link')})")

        except Exception as e:
            st.error(f"Error: {e}")
