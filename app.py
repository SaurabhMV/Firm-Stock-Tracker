import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import google.generativeai as genai
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- APP CONFIG & ANDROID RENAME FIX ---
st.set_page_config(page_title="Firm Stock Tracker", page_icon="📈")

# This helps Android recognize your custom name when "Adding to Home Screen"
st.markdown("""
    <head>
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="application-name" content="Stock Tracker">
    </head>
""", unsafe_allow_html=True)

# --- SIDEBAR MODEL SELECTION ---
with st.sidebar:
    st.title("Settings ⚙️")
    
    # Map friendly names to actual Google Model IDs
    model_options = {
        "Gemini 3 Pro (Smartest)": "gemini-3-pro-preview",
        "Gemini 3 Flash (Fast & Smart)": "gemini-3-flash-preview",
        "Gemini 2.5 Pro (Balanced)": "gemini-2.5-pro",
        "Gemini 2.5 Flash (Lightweight)": "gemini-2.5-flash"
    }
    
    selected_model_display = st.selectbox(
        "Choose Gemini Model:",
        options=list(model_options.keys()),
        index=1  # Defaults to Gemini 3 Flash
    )
    
    selected_model_id = model_options[selected_model_display]
    
    # Get API Key from secrets
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("⚠️ API Key not found in secrets!")
    
    st.info(f"Currently using: {selected_model_id}")

# --- HELPER FUNCTIONS ---

def get_analyst_data(ticker):
    """Fetches Analyst recommendations and targets safely."""
    info = ticker.info
    current = info.get('currentPrice')
    target = info.get('targetMeanPrice')
    
    if current and target:
        upside = ((target - current) / current) * 100
    else:
        upside = 0
        
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
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [df.Close.iloc[i] for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    
    return {
        "RSI": df['RSI'].iloc[-1], 
        "Supports": supports, 
        "Price": df['Close'].iloc[-1]
    }

def generate_pro_report(symbol, info, tech, news, api_key, model_id):
    """Generates the AI analysis using the sidebar-selected model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = f"""
    Act as a Hedge Fund Strategy Lead. Analyze {symbol} for a long-term investor.
    
    CONTEXT:
    - Business: {info.get('longBusinessSummary', 'N/A')[:1000]}
    - Fundamentals: P/E {info.get('forwardPE', 'N/A')}, Margin {info.get('profitMargins', 'N/A')}, Debt/Equity {info.get('debtToEquity', 'N/A')}
    - Technicals: RSI {tech.get('RSI', 50):.2f}, Key Support Levels: {tech.get('Supports')}
    - Recent Headlines: {news}

    STRUCTURE:
    1. THE BULL CASE
    2. THE BEAR CASE
    3. VALUATION CHECK
    4. FINAL VERDICT: Buy/Hold/Sell with Confidence Score.
    """
    return model.generate_content(prompt).text

# --- MAIN UI ---
ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
analyze_btn = st.button("Generate Deep Research")

# --- MAIN APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in Streamlit Secrets.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input} with {selected_model_display}..."):
                ticker = yf.Ticker(ticker_input)
                history = ticker.history(period="1y")
                
                if history.empty:
                    st.error(f"Could not fetch data for {ticker_input}. Check the symbol.")
                    st.stop()

                info = ticker.info
                analyst = get_analyst_data(ticker)
                tech_data = calculate_technicals(history)
                
                # --- TOP ROW: KPI METRICS ---
                st.subheader(f"📊 {ticker_input} Market Snapshot")
                c1, c2, c3, c4 = st.columns(4)
                
                current_price = info.get('currentPrice', history['Close'].iloc[-1])
                target_price = analyst['Target'] if analyst['Target'] else "N/A"
                
                c1.metric("Current Price", f"${current_price:.2f}")
                c2.metric("Target Price", f"${target_price}", f"{analyst['Upside']:.1f}% Upside" if isinstance(target_price, (int, float)) else None)
                c3.metric("Wall St. Consensus", analyst['Consensus'])
                c4.metric("RSI (14)", f"{tech_data['RSI']:.1f}")

                tab1, tab2, tab3 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News"])

                with tab1:
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        fig = go.Figure(data=[go.Candlestick(x=history.index, 
                                        open=history['Open'], high=history['High'], 
                                        low=history['Low'], close=history['Close'])])
                        fig.update_layout(title=f"{ticker_input} Price Action", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with col_b:
                        st.write("### Key Fundamentals")
                        fund_data = {
                            "Market Cap": info.get("marketCap"),
                            "Forward P/E": info.get("forwardPE"),
                            "Profit Margins": info.get("profitMargins")
                        }
                        st.table(pd.DataFrame.from_dict(fund_data, orient='index', columns=['Value']))

                with tab2:
                    st.subheader(f"🤖 {selected_model_display} Memo")
                    news_titles = [n.get('title') for n in ticker.news[:5]]
                    report = generate_pro_report(ticker_input, info, tech_data, news_titles, api_key, selected_model_id)
                    st.markdown(report)

                with tab3:
                    st.subheader("⚡ AI News & Events")
                    calendar = ticker.calendar
                    if calendar:
                        st.write("### Upcoming Calendar")
                        st.write(calendar)
                    
                    st.divider()
                    for n in ticker.news[:5]:
                        st.write(f"**{n.get('title')}**")
                        st.caption(f"Source: {n.get('publisher')} | [Link]({n.get('link')})")

        except Exception as e:
            st.error(f"An error occurred: {e}")
