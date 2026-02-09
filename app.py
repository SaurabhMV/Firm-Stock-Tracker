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

# --- MAIN UI INPUTS ---
ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
analyze_btn = st.button("Generate Deep Research")
period = st.sidebar.selectbox("History", ["1mo", "6mo", "1y", "2y", "5y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1h", "1d", "1wk"], index=1)

if ticker_input:
    ticker_data = yf.Ticker(ticker_input)
    
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
    if df.empty or len(df) < 14: 
        return {"RSI": 50, "Supports": [], "Price": 0}
    
    # --- PRO RSI CALCULATION (Wilder's Smoothing) ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Support Level Logic
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [round(df.Close.iloc[i], 2) for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    
    return {
        "RSI": df['RSI'].iloc[-1], 
        "Supports": supports, 
        "Price": df['Close'].iloc[-1]
    }

def generate_pro_report(symbol, info, tech, news_data, key, model_id):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_id)
    today = datetime.date.today().strftime('%B %d, %Y')
    
    prompt = f"""
    You are a Senior Wall Street Analyst. The current date is **{today}**.
    Generate a professional investment thesis for {symbol} ({info.get('longName')}).
    ### DATA SOURCE (Real-time):
    1. Fundamentals: Price: ${info.get('currentPrice')}, P/E: {info.get('forwardPE')}, PEG: {info.get('pegRatio')}, Target: ${info.get('targetMeanPrice')}
    2. Technicals: RSI: {tech.get('RSI', 50):.2f}, Supports: {tech.get('Supports')}
    3. Recent News: {news_data}
    ...
    """
    return model.generate_content(prompt).text

def get_val(df, options):
    """Helper to find the first available row from a list of possible names."""
    for opt in options:
        if opt in df.index:
            return df.loc[opt]
    return None

def calculate_piotroski_score(ticker_obj):
    try:
        income = ticker_obj.financials
        balance = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        if income.empty or income.shape[1] < 2:
            return None, "Insufficient annual data (need 2+ years)."

        f_score = 0
        details = []

        # 1 & 2. Profitability (Net Income & Operating Cash Flow)
        ni = get_val(income, ['Net Income', 'Net Income Common Stockholders'])
        ocf = get_val(cashflow, ['Operating Cash Flow', 'Cash Flow From Operating Activities'])
        
        if ni is not None and ni.iloc[0] > 0: f_score += 1
        if ocf is not None and ocf.iloc[0] > 0: f_score += 1

        # 3. ROA Improvement
        assets = get_val(balance, ['Total Assets'])
        if all(x is not None for x in [ni, assets]):
            roa_curr = ni.iloc[0] / assets.iloc[0]
            roa_prev = ni.iloc[1] / assets.iloc[1]
            if roa_curr > roa_prev: f_score += 1
        
        # 4. Accruals (OCF > Net Income)
        if ni is not None and ocf is not None:
            if ocf.iloc[0] > ni.iloc[0]: f_score += 1

        # 5. Leverage (Long Term Debt)
        debt = get_val(balance, ['Long Term Debt', 'Total Non Current Liabilities Net Minority Interest'])
        if debt is not None and assets is not None:
            # Check if debt is decreasing relative to assets
            if (debt.iloc[0]/assets.iloc[0]) < (debt.iloc[1]/assets.iloc[1]): f_score += 1

        # 6. Liquidity (Current Ratio)
        ca = get_val(balance, ['Current Assets', 'Total Current Assets'])
        cl = get_val(balance, ['Current Liabilities', 'Total Current Liabilities'])
        if ca is not None and cl is not None:
            if (ca.iloc[0]/cl.iloc[0]) > (ca.iloc[1]/cl.iloc[1]): f_score += 1

        # 7. NO NEW SHARES (The fix for your error)
        shares = get_val(balance, ['Ordinary Share Capital', 'Common Stock', 'Share Capital', 'Issuance Of Capital Stock'])
        if shares is not None:
            # For share capital, lower or equal is a point
            if shares.iloc[0] <= shares.iloc[1]: f_score += 1
            details.append(("No Share Dilution", "✅" if shares.iloc[0] <= shares.iloc[1] else "❌"))
        else:
            details.append(("No Share Dilution", "⚠️ Row Not Found"))

        # 8 & 9. Efficiency (Gross Margin & Asset Turnover)
        rev = get_val(income, ['Total Revenue'])
        cogs = get_val(income, ['Cost Of Revenue'])
        if all(x is not None for x in [rev, cogs, assets]):
            # Margin check
            gm_curr = (rev.iloc[0] - cogs.iloc[0]) / rev.iloc[0]
            gm_prev = (rev.iloc[1] - cogs.iloc[1]) / rev.iloc[1]
            if gm_curr > gm_prev: f_score += 1
            # Turnover check
            if (rev.iloc[0]/assets.iloc[0]) > (rev.iloc[1]/assets.iloc[1]): f_score += 1

        return f_score, details 
    except Exception as e:
        return None, f"Error: {str(e)}"
        
def calculate_fundamental_score(ticker_obj):
    """
    Calculates a custom 6-point fundamental scorecard.
    Returns: (Total Score, List of individual metric results)
    """
    info = ticker_obj.info
    score = 0
    results = []

    # 1. Valuation: P/E Ratio (Lower is usually better for value)
    pe = info.get('forwardPE')
    if pe and pe < 25:
        score += 1
        results.append({"Metric": "P/E Ratio", "Value": f"{pe:.2f}", "Status": "✅ Healthy", "Note": "Below 25"})
    else:
        results.append({"Metric": "P/E Ratio", "Value": f"{pe if pe else 'N/A'}", "Status": "⚠️ High/NA", "Note": "Over 25"})

    # 2. Valuation: PEG Ratio (Price/Earnings to Growth - Under 1 is 'cheap')
    peg = info.get('pegRatio')
    if peg and peg < 1:
        score += 1
        results.append({"Metric": "PEG Ratio", "Value": f"{peg:.2f}", "Status": "✅ Undervalued", "Note": "Below 1.0"})
    else:
        results.append({"Metric": "PEG Ratio", "Value": f"{peg if peg else 'N/A'}", "Status": "❌ Overvalued", "Note": "Above 1.0"})

    # 3. Profitability: ROE (Return on Equity)
    roe = info.get('returnOnEquity')
    if roe and roe > 0.15:
        score += 1
        results.append({"Metric": "Return on Equity", "Value": f"{roe*100:.1f}%", "Status": "✅ Strong", "Note": "Above 15%"})
    else:
        results.append({"Metric": "Return on Equity", "Value": f"{roe*100 if roe else 'N/A'}%", "Status": "❌ Weak", "Note": "Below 15%"})

    # 4. Debt Health: Debt to Equity
    de = info.get('debtToEquity')
    if de and de < 100: # yfinance debtToEquity is often expressed as a percentage (100 = 1.0)
        score += 1
        results.append({"Metric": "Debt to Equity", "Value": f"{de/100:.2f}", "Status": "✅ Low Debt", "Note": "Below 1.0"})
    else:
        results.append({"Metric": "Debt to Equity", "Value": f"{de/100 if de else 'N/A'}", "Status": "⚠️ High Leverage", "Note": "Above 1.0"})

    # 5. Liquidity: Current Ratio
    cr = info.get('currentRatio')
    if cr and cr > 1.5:
        score += 1
        results.append({"Metric": "Current Ratio", "Value": f"{cr:.2f}", "Status": "✅ Liquid", "Note": "Above 1.5"})
    else:
        results.append({"Metric": "Current Ratio", "Value": f"{cr if cr else 'N/A'}", "Status": "❌ Tight", "Note": "Below 1.5"})

    # 6. Profitability: Operating Margins
    margin = info.get('operatingMargins')
    if margin and margin > 0.15:
        score += 1
        results.append({"Metric": "Operating Margin", "Value": f"{margin*100:.1f}%", "Status": "✅ High", "Note": "Above 15%"})
    else:
        results.append({"Metric": "Operating Margin", "Value": f"{margin*100 if margin else 'N/A'}%", "Status": "❌ Thin", "Note": "Below 15%"})

    return score, results

# --- MAIN APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input}..."):
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

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News", "⚖️ Fundamental Scorecard", "📈 SMA Strategy"])

                # --- TAB 1: CHARTS & FINANCIALS ---
                with tab1:
                    st.write("### 🧭 Investment Sentiment")
                    rsi_val = tech_data['RSI']
                    rsi_score = 100 - rsi_val if rsi_val else 50 
                    rec_map = {"Strong Buy": 100, "Buy": 75, "Hold": 50, "Sell": 25, "Strong Sell": 0, "N/A": 50}
                    analyst_score = rec_map.get(analyst['Consensus'], 50)
                    upside_score = np.clip((analyst['Upside'] + 10) * 2, 0, 100)
                    final_sentiment = (rsi_score * 0.3) + (analyst_score * 0.4) + (upside_score * 0.3)

                    col_gauge, col_metrics = st.columns([1, 1])
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number", value = final_sentiment,
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                                     'steps': [{'range': [0, 40], 'color': "#FF4B4B"},
                                               {'range': [40, 60], 'color': "#FFAA00"},
                                               {'range': [60, 100], 'color': "#00CC96"}]}))
                        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_metrics:
                        st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                        st.metric("P/E (Trailing)", f"{info.get('trailingPE', 'N/A')}")
                        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

                    st.divider()

                    # --- INTERACTIVE PRICE CHART WITH BOLLINGER BANDS ---
                    st.write("### 📈 Price Action & Technical Bands")
                    
                    # Bollinger Band Calculation (20-day SMA)
                    bb_window = 20
                    history['SMA20'] = history['Close'].rolling(window=bb_window).mean()
                    history['StdDev'] = history['Close'].rolling(window=bb_window).std()
                    history['Upper_BB'] = history['SMA20'] + (history['StdDev'] * 2)
                    history['Lower_BB'] = history['SMA20'] - (history['StdDev'] * 2)

                    spy = yf.Ticker("SPY")
                    spy_hist = spy.history(period="1y")

                    fig = go.Figure()

                    # Add Bollinger Bands (Added first so they are behind candlesticks)
                    fig.add_trace(go.Scatter(x=history.index, y=history['Upper_BB'], 
                                             line=dict(color='rgba(173, 216, 230, 0.2)', width=1), 
                                             showlegend=False, name="Upper Band"))
                    fig.add_trace(go.Scatter(x=history.index, y=history['Lower_BB'], 
                                             line=dict(color='rgba(173, 216, 230, 0.2)', width=1), 
                                             fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', 
                                             showlegend=False, name="Lower Band"))
                    fig.add_trace(go.Scatter(x=history.index, y=history['SMA20'], 
                                             line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash'), 
                                             name="20-Day SMA"))

                    # Add Main Candlestick Chart
                    fig.add_trace(go.Candlestick(
                        x=history.index, open=history['Open'], high=history['High'], 
                        low=history['Low'], close=history['Close'], name=f"{ticker_input}"
                    ))

                    # Add S&P 500 Line
                    fig.add_trace(go.Scatter(
                        x=spy_hist.index, y=spy_hist['Close'], 
                        name="S&P 500", 
                        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot'),
                        yaxis="y2"
                    ))

                    fig.update_layout(
                        template="plotly_dark", xaxis_rangeslider_visible=False, height=500,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis=dict(title="Stock Price ($)"),
                        yaxis2=dict(title="S&P 500 ($)", overlaying="y", side="right", showgrid=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance Summary
                    stock_perf = ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]) * 100
                    spy_perf = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) / spy_hist['Close'].iloc[0]) * 100
                    diff = stock_perf - spy_perf
                    if diff > 0: st.success(f"🚀 {ticker_input} outperforming S&P 500 by {diff:.2f}%")
                    else: st.error(f"📉 {ticker_input} underperforming S&P 500 by {abs(diff):.2f}%")
                        
                    # Peer Comparison (Keeping your logic intact)
                    st.write("### 🏁 Smart Peer Comparison")
                    sector_peers = {"Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"], "Financial Services": ["JPM", "BAC", "GS"], "Healthcare": ["JNJ", "PFE", "UNH"]}
                    current_sector = info.get('sector', "Technology")
                    relevant_list = sector_peers.get(current_sector, ["AAPL", "MSFT", "GOOGL"])
                    selected_peers = st.multiselect(f"Compare {current_sector} leaders:", options=relevant_list + ["SPY"], default=relevant_list[:2])
                    
                    if selected_peers:
                        compare_data = []
                        for p in [ticker_input] + selected_peers:
                            p_info = yf.Ticker(p).info
                            compare_data.append({"Ticker": p, "P/E": p_info.get('trailingPE', 0), "Margin %": (p_info.get('profitMargins', 0) or 0) * 100})
                        df_compare = pd.DataFrame(compare_data).set_index("Ticker")
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
                            
                # --- TAB 3: NEWS & EVENTS (FIXED & ERROR-PROOFED) ---
                with tab3:
                    st.write("### 📡 Market Intelligence")
                    
                    def safe_date(ts):
                        if ts and isinstance(ts, (int, float)):
                            return datetime.datetime.fromtimestamp(ts).strftime('%b %d, %Y')
                        return "Date TBD"

                    st.write("#### ✨ Executive News Summary")
                    news_items = ticker.news
                    
                    if news_items:
                        valid_news = [f"[{safe_date(n.get('providerPublishTime'))}] {n.get('title')}" for n in news_items[:8]]
                        news_summary_text = "\n".join(valid_news)

                        if api_key:
                            try:
                                with st.status("Analyzing Headlines...", expanded=False) as status:
                                    genai.configure(api_key=api_key)
                                    sent_model = genai.GenerativeModel(selected_model_id)
                                    sent_prompt = f"Summarize top 3 themes for {ticker_input} from these headlines:\n{news_summary_text}"
                                    res = sent_model.generate_content(sent_prompt)
                                    st.write(res.text)
                                    status.update(label="Analysis Complete", state="complete")
                            except Exception:
                                st.info("AI Summary temporarily unavailable.")
                    else:
                        st.info("No recent news found.")

                    st.divider()

                    # --- CORPORATE CALENDAR (FIXED FOR DICT/DATAFRAME) ---
                    st.write("#### 📅 Corporate Calendar")
                    ev1, ev2 = st.columns(2)
                    
                    cal = ticker.calendar
                    next_earnings = "TBD"
                    
                    # Fix for 'dict' object has no attribute 'empty'
                    if cal is not None:
                        try:
                            if isinstance(cal, pd.DataFrame) and not cal.empty:
                                next_earnings = cal.iloc[0, 0].strftime('%b %d, %Y')
                            elif isinstance(cal, dict):
                                e_dates = cal.get('Earnings Date') or cal.get('Earnings')
                                if e_dates: next_earnings = e_dates[0].strftime('%b %d, %Y')
                        except:
                            next_earnings = "Check Investor Relations"

                    with ev1:
                        st.markdown(f"""
                            <div style="border: 1px solid #333; padding: 15px; border-radius: 10px; background-color: #111; height: 120px;">
                                <p style="color: #888; margin:0; font-size: 12px;">NEXT EARNINGS</p>
                                <h3 style="margin: 5px 0; color: #00CC96;">{next_earnings}</h3>
                                <p style="margin:0; font-size: 13px;">Quarterly Results</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with ev2:
                        ex_date_raw = info.get('exDividendDate')
                        ex_str = safe_date(ex_date_raw)
                        st.markdown(f"""
                            <div style="border: 1px solid #333; padding: 15px; border-radius: 10px; background-color: #111; height: 120px;">
                                <p style="color: #888; margin:0; font-size: 12px;">DIVIDEND EX-DATE</p>
                                <h3 style="margin: 5px 0; color: #FFAA00;">{ex_str}</h3>
                                <p style="margin:0; font-size: 13px;">Income Distribution</p>
                            </div>
                        """, unsafe_allow_html=True)

                # --- TAB 4: FUNDAMENTAL SCORECARD (FIXED VARIABLE NAME) ---
                with tab4:
                    st.header("🛡️ Piotroski F-Score")
                    
                    with st.spinner("Analyzing Year-over-Year improvements..."):
                        f_score, f_result = calculate_piotroski_score(ticker)
                    
                    # Check if f_result is a list (Success) or a string (Error)
                    if isinstance(f_result, list):
                        col_f1, col_f2 = st.columns([1, 2])
                        with col_f1:
                            st.metric("Piotroski Score", f"{f_score} / 9")
                            if f_score >= 8: st.success("🎯 Elite Strength")
                            elif f_score >= 5: st.warning("⚖️ Average Health")
                            else: st.error("🚨 Financial Distress")
                        
                        with col_f2:
                            # Correctly calling the DataFrame constructor with a list of tuples
                            df_f = pd.DataFrame(f_result, columns=["Check", "Status"])
                            st.table(df_f)
                    else:
                        # If f_result is the error string
                        st.error(f"F-Score calculation failed: {f_result}")
                
                    st.divider()
                    # ... Rest of your Fundamental Scorecard code (P/E, PEG, etc.)
                    
                    st.header("Fundamental Scorecard")
                    st.write("This scorecard evaluates the stock across 6 key pillars of value and health.")
                
                    # FIXED: Changed 'ticker_data.info' to 'ticker.info'
                    f_info = ticker.info
                    
                    metrics = [
                        {"name": "P/E Ratio (Forward)", "key": "forwardPE", "op": "lt", "val": 25, "desc": "Target < 25"},
                        {"name": "PEG Ratio", "key": "pegRatio", "op": "lt", "val": 1.0, "desc": "Target < 1.0"},
                        {"name": "Return on Equity (ROE)", "key": "returnOnEquity", "op": "gt", "val": 0.15, "desc": "Target > 15%"},
                        {"name": "Debt to Equity", "key": "debtToEquity", "op": "lt", "val": 100, "desc": "Target < 100%"},
                        {"name": "Current Ratio", "key": "currentRatio", "op": "gt", "val": 1.5, "desc": "Target > 1.5"},
                        {"name": "Operating Margin", "key": "operatingMargins", "op": "gt", "val": 0.15, "desc": "Target > 15%"}
                    ]
                
                    f_score = 0
                    score_rows = []
                
                    for m in metrics:
                        val = f_info.get(m['key'])
                        status = "❌"
                        if val is not None:
                            passed = (m['op'] == 'lt' and val < m['val']) or (m['op'] == 'gt' and val > m['val'])
                            if passed:
                                f_score += 1
                                status = "✅"
                            disp = f"{val*100:.1f}%" if "%" in m['desc'] else f"{val:.2f}"
                        else:
                            disp = "N/A"
                        score_rows.append({"Status": status, "Metric": m['name'], "Value": disp, "Target": m['desc']})
                
                    sc1, sc2 = st.columns([1, 2])
                    with sc1:
                        st.metric("Fundamental Health", f"{f_score} / 6")
                        if f_score >= 5: st.success("Strong Fundamentals")
                        elif f_score >= 3: st.warning("Average Health")
                        else: st.error("Weak Fundamentals")
                
                    with sc2:
                        st.table(score_rows)

                with tab5:
                    st.header("18 vs 50 SMA Strategy + Volume Analysis")
                    
                    if ticker:
                        try:
                            # Fetch Data
                            df = yf.download(ticker.ticker, period=period, interval=interval, auto_adjust=True)
                            
                            if df.empty:
                                st.error("No data found for this ticker.")
                            else:
                                # Handle MultiIndex column names (Common in 2025/2026 yfinance versions)
                                if isinstance(df.columns, pd.MultiIndex):
                                    df.columns = df.columns.get_level_values(0)
                
                                # --- Calculations ---
                                df['SMA18'] = df['Close'].rolling(window=18).mean()
                                df['SMA50'] = df['Close'].rolling(window=50).mean()
                                df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
                                
                                # RSI Calculation
                                delta = df['Close'].diff()
                                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                                df['RSI'] = 100 - (100 / (1 + (gain / loss)))
                
                                # Signal Logic
                                df['Signal'] = 0.0
                                # We start logic after the 50th bar to ensure SMA50 is populated
                                if len(df) > 50:
                                    df.loc[df.index[50:], 'Signal'] = np.where(df['SMA18'][50:] > df['SMA50'][50:], 1.0, 0.0)
                                    df['Position'] = df['Signal'].diff()
                
                                # --- Plotting ---
                                fig = make_subplots(
                                    rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    subplot_titles=(f'{ticker} Price & SMAs', 'Volume + 20-MA', 'RSI Momentum'), 
                                    row_width=[0.2, 0.2, 0.6] 
                                )
                
                                # Row 1: Price & Strategy
                                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['SMA18'], line=dict(color='orange', width=2), name='18 SMA'), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='cyan', width=2), name='50 SMA'), row=1, col=1)
                
                                # Buy/Sell Markers
                                buy_pts = df[df['Position'] == 1]
                                fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['SMA18'], mode='markers',
                                                         marker=dict(symbol='triangle-up', size=12, color='lime'), name='BUY'), row=1, col=1)
                
                                sell_pts = df[df['Position'] == -1]
                                fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['SMA18'], mode='markers',
                                                         marker=dict(symbol='triangle-down', size=12, color='red'), name='SELL'), row=1, col=1)
                
                                # Row 2: Volume
                                vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
                                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=vol_colors), row=2, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['Vol_Avg'], line=dict(color='yellow', width=1.5), name='Vol 20-MA'), row=2, col=1)
                
                                # Row 3: RSI
                                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='magenta'), name='RSI'), row=3, col=1)
                                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                                fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                
                        except Exception as e:
                            st.error(f"Strategy Error: {e}")
                            
        except Exception as e:
            st.error(f"Error: {e}")
