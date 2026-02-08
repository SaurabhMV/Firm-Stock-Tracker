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

st.markdown("""
    <head>
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="application-name" content="Stock Tracker">
    </head>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.title("Settings ⚙️")
    
    # API KEY OPTION
    api_key_manual = st.text_input("Gemini API Key", type="password", placeholder="Paste key or leave for secrets...")
    api_key = api_key_manual or st.secrets.get("GOOGLE_API_KEY", "")
    
    # MODEL SELECTION
    model_options = {
        "Gemini 3 Pro (Smartest)": "gemini-3-pro-preview",
        "Gemini 3 Flash (Fast & Smart)": "gemini-3-flash-preview",
        "Gemini 2.5 Pro (Balanced)": "gemini-2.5-pro",
        "Gemini 2.5 Flash (Lightweight)": "gemini-2.5-flash"
    }
    
    selected_model_display = st.selectbox("Choose Gemini Model:", options=list(model_options.keys()), index=1)
    selected_model_id = model_options[selected_model_display]
    
    if api_key:
        st.success("✅ API Key Loaded")
    else:
        st.warning("⚠️ No API Key Detected")

# --- MAIN UI INPUTS (FIXED INDENTATION) ---
ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
analyze_btn = st.button("Generate Deep Research")

# --- HELPER FUNCTIONS ---

def get_analyst_data(ticker):
    info = ticker.info
    current = info.get('currentPrice')
    target = info.get('targetMeanPrice')
    upside = ((target - current) / current) * 100 if (current and target) else 0
    rec = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
    return {"Target": target, "Upside": upside, "Consensus": rec}

def calculate_technicals(history):
    df = history.copy()
    if df.empty: return {"RSI": 50, "Supports": [], "Price": 0}
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [df.Close.iloc[i] for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    return {"RSI": df['RSI'].iloc[-1], "Supports": supports, "Price": df['Close'].iloc[-1]}

def generate_pro_report(symbol, info, tech, news_data, key, model_id):
    """Generates the AI analysis with strict date awareness."""
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_id)
    
    # Get today's date for the prompt
    today = datetime.date.today().strftime('%B %d, %Y')
    
    prompt = f"""
    You are a Senior Wall Street Analyst. The current date is **{today}**.
    
    Generate a professional investment thesis for {symbol} ({info.get('longName')}).
    
    ### DATA SOURCE (Real-time):
    1. **Fundamentals:**
       - Price: ${info.get('currentPrice')}
       - P/E Ratio: {info.get('forwardPE', 'N/A')}
       - PEG Ratio: {info.get('pegRatio', 'N/A')} (Value < 1.0 is undervalued)
       - Analyst Target: ${info.get('targetMeanPrice', 'N/A')}
    
    2. **Technicals:**
       - RSI (14): {tech.get('RSI', 50):.2f}
       - Support Levels: {tech.get('Supports')}
    
    3. **Recent News Headlines (Chronological):**
    {news_data}

    ### INSTRUCTIONS:
    - **Prioritize the 'Recent News' provided above.** If a headline is from 2026, treat it as critical.
    - **Ignore outdated training data** (e.g., from 2023 or 2024) if it conflicts with the provided news.
    - If the RSI is > 70, mention "Overbought" risks. If < 30, mention "Oversold" opportunities.
    
    ### OUTPUT FORMAT (Markdown):
    **1. 🐂 The Bull Case** (Focus on growth drivers & recent positive earnings/news)
    **2. 🐻 The Bear Case** (Focus on risks, valuation concerns, or negative news)
    **3. ⚖️ Valuation Check** (Compare P/E and PEG to fair value. Is it cheap?)
    **4. 🏁 Final Verdict** (Buy/Hold/Sell with a specific timeframe, e.g., "12-month Buy").
    """
    
    return model.generate_content(prompt).text
    
# --- MAIN APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input} using {selected_model_display}..."):
                ticker = yf.Ticker(ticker_input)
                history = ticker.history(period="1y")
                
                if history.empty:
                    st.error("Could not fetch data.")
                    st.stop()

                info = ticker.info
                analyst = get_analyst_data(ticker)
                tech_data = calculate_technicals(history)
                
                # --- KPI METRICS ---
                st.subheader(f"📊 {ticker_input} Market Snapshot")
                c1, c2, c3, c4 = st.columns(4)
                curr = info.get('currentPrice', history['Close'].iloc[-1])
                c1.metric("Current Price", f"${curr:.2f}")
                c2.metric("Target Price", f"${analyst['Target'] or 'N/A'}", f"{analyst['Upside']:.1f}%")
                c3.metric("Consensus", analyst['Consensus'])
                c4.metric("RSI (14)", f"{tech_data['RSI']:.1f}")

                tab1, tab2, tab3 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News"])

# --- TAB 1: CHARTS & FINANCIALS (Bloomberg Style) ---
                with tab1:
                    # 1. SENTIMENT GAUGE & TOP KPI
                    st.write("### 🧭 Investment Sentiment")
                    
                    # Logic for Sentiment Score (0-100)
                    # RSI Score: Lower RSI (Oversold) is Bullish (Higher score)
                    rsi_val = tech_data['RSI']
                    rsi_score = 100 - rsi_val if rsi_val else 50 
                    
                    # Analyst Score
                    rec_map = {"Strong Buy": 100, "Buy": 75, "Hold": 50, "Sell": 25, "Strong Sell": 0, "N/A": 50}
                    analyst_score = rec_map.get(analyst['Consensus'], 50)
                    
                    # Upside Score
                    upside_val = analyst['Upside']
                    upside_score = np.clip((upside_val + 10) * 2, 0, 100) # Simple map
                    
                    final_sentiment = (rsi_score * 0.3) + (analyst_score * 0.4) + (upside_score * 0.3)

                    # Display Gauge
                    col_gauge, col_metrics = st.columns([1, 1])
                    
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = final_sentiment,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Bullishness Score", 'font': {'size': 18}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "white"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#FF4B4B"},   # Bearish (Red)
                                    {'range': [40, 60], 'color': "#FFAA00"},  # Neutral (Orange)
                                    {'range': [60, 100], 'color': "#00CC96"}  # Bullish (Green)
                                ],
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_metrics:
                        st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                        st.metric("P/E (Trailing)", f"{info.get('trailingPE', 'N/A')}")
                        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

                    st.divider()

# 2. INTERACTIVE PRICE CHART WITH S&P 500 COMPARISON
                    st.write("### 📈 Price Action vs. S&P 500")
                    
                    # Fetch S&P 500 data for comparison
                    spy = yf.Ticker("SPY")
                    spy_hist = spy.history(period="1y")

                    # Normalize both to 100 to show % change (Relative Strength)
                    stock_norm = (history['Close'] / history['Close'].iloc[0]) * 100
                    spy_norm = (spy_hist['Close'] / spy_hist['Close'].iloc[0]) * 100

                    fig = go.Figure()

                    # Add Main Candlestick Chart
                    fig.add_trace(go.Candlestick(
                        x=history.index, open=history['Open'], high=history['High'], 
                        low=history['Low'], close=history['Close'], name=f"{ticker_input} Price"
                    ))

                    # Add S&P 500 Line (on a secondary Y-axis or as a % overlay)
                    fig.add_trace(go.Scatter(
                        x=spy_hist.index, y=spy_hist['Close'], 
                        name="S&P 500 ($SPY)", 
                        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot'),
                        yaxis="y2"
                    ))

                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        height=500,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis=dict(title="Stock Price ($)"),
                        yaxis2=dict(
                            title="S&P 500 ($)",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Relative Performance Summary
                    stock_perf = ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]) * 100
                    spy_perf = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) / spy_hist['Close'].iloc[0]) * 100
                    diff = stock_perf - spy_perf
                    
                    if diff > 0:
                        st.success(f"🚀 {ticker_input} is outperforming the S&P 500 by {diff:.2f}% this year.")
                    else:
                        st.error(f"📉 {ticker_input} is underperforming the S&P 500 by {abs(diff):.2f}% this year.")
                        
                    # 3. SMART PEER COMPARISON
                    st.write("### 🏁 Smart Peer Comparison")
                    
                    # Mapping of sectors to relevant industry leaders
                    sector_peers = {
                        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "ORCL", "ADBE"],
                        "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
                        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "LLY", "MRK"],
                        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX"],
                        "Communication Services": ["META", "NFLX", "DIS", "TMUS", "VZ", "T"],
                        "Energy": ["XOM", "CVX", "SHEL", "BP", "TTE"],
                        "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM"],
                        "Industrials": ["BA", "GE", "CAT", "UPS", "HON", "LMT"]
                    }

                    # Detect current sector
                    current_sector = info.get('sector', "Technology") # Default to tech if unknown
                    
                    # Get relevant list or fallback to general tech giants
                    relevant_list = sector_peers.get(current_sector, ["AAPL", "MSFT", "GOOGL", "AMZN"])
                    
                    # Remove current ticker if it's in the peer list
                    if ticker_input in relevant_list:
                        relevant_list = [p for p in relevant_list if p != ticker_input]
                    
                    selected_peers = st.multiselect(
                        f"Compare with other {current_sector} leaders:", 
                        options=relevant_list + ["SPY", "QQQ"], # Added ETFs as bonus options
                        default=relevant_list[:3] # Pre-select top 3 rivals
                    )
                    
                    if selected_peers:
                        compare_list = []
                        for p in [ticker_input] + selected_peers:
                            try:
                                p_ticker = yf.Ticker(p)
                                p_info = p_ticker.info
                                compare_list.append({
                                    "Ticker": p,
                                    "P/E": p_info.get('trailingPE', 0),
                                    "Margin %": (p_info.get('profitMargins', 0) or 0) * 100,
                                    "Rev Growth %": (p_info.get('revenueGrowth', 0) or 0) * 100
                                })
                            except:
                                continue
                                
                        if compare_list:
                            df_compare = pd.DataFrame(compare_list).set_index("Ticker")
                            
                            # Visual Comparison Chart
                            fig_comp = go.Figure()
                            fig_comp.add_trace(go.Bar(x=df_compare.index, y=df_compare['P/E'], name='P/E Ratio', marker_color='#00CC96'))
                            fig_comp.update_layout(title=f"P/E Comparison: {current_sector}", template="plotly_dark", height=300)
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            st.dataframe(df_compare, use_container_width=True)
                            
                    # 4. RESTORED DETAILED FUNDAMENTALS (The "Full List")
                    st.write("### 📋 Detailed Financials")
                    detailed_data = {
                        "Forward P/E": info.get("forwardPE"),
                        "PEG Ratio": info.get("pegRatio"),
                        "Price to Book": info.get("priceToBook"),
                        "Total Cash": f"${info.get('totalCash', 0):,}",
                        "Total Debt": f"${info.get('totalDebt', 0):,}",
                        "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%",
                        "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0)*100:.2f}%",
                        "Free Cash Flow": f"${info.get('freeCashflow', 0):,}",
                        "52 Week High": f"${info.get('fiftyTwoWeekHigh')}",
                        "52 Week Low": f"${info.get('fiftyTwoWeekLow')}"
                    }
                    st.table(pd.DataFrame.from_dict(detailed_data, orient='index', columns=['Value']))
                    
# --- TAB 2: AI THESIS (UPDATED) ---
                with tab2:
                    st.write(f"### 🤖 Gemini {selected_model_display.split(' ')[1]} Analysis")
                    st.caption(f"Generated on {datetime.date.today().strftime('%B %d, %Y')}")
                    
                    # 1. Prepare News with Dates (Critical for Freshness)
                    news_context = ""
                    if ticker.news:
                        for n in ticker.news[:7]: # Analyze top 7 stories
                            # Convert unix timestamp to readable date
                            ts = n.get('providerPublishTime', 0)
                            pub_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            title = n.get('title')
                            news_context += f"- [{pub_date}] {title}\n"
                    else:
                        news_context = "No recent news available via API."

                    # 2. Check for API Key
                    if not api_key:
                        st.warning("⚠️ Please enter your Gemini API Key in the sidebar to generate the thesis.")
                        st.stop()

                    # 3. Generate Report
                    with st.spinner("Analyzing recent news, fundamentals, and technicals..."):
                        try:
                            # Pass the formatted 'news_context' instead of just titles
                            report = generate_pro_report(
                                ticker_input, 
                                info, 
                                tech_data, 
                                news_context, 
                                api_key, 
                                selected_model_id
                            )
                            st.markdown(report)
                            
                            # Disclaimer
                            st.divider()
                            st.caption("Disclaimer: AI-generated content may contain hallucinations. Always verify earnings dates and financial data manually.")
                            
                        except Exception as e:
                            st.error(f"AI Generation Failed: {e}")
                            
                with tab3:
                    st.subheader("⚡ AI Executive News Briefing")
                    news_items = ticker.news[:10]
                    if news_items:
                        news_text = "\n".join([f"- {n.get('title')}" for n in news_items])
                        genai.configure(api_key=api_key)
                        news_model = genai.GenerativeModel(selected_model_id)
                        news_prompt = f"Summarize sentiment and themes for {ticker_input} news:\n{news_text}"
                        st.success(news_model.generate_content(news_prompt).text)
                    
                    st.divider()
                    st.subheader("📅 Upcoming Events")
                    st.write(ticker.calendar)

        except Exception as e:
            st.error(f"Error: {e}")
