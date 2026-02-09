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

def calculate_piotroski_score(ticker_obj):
    """Calculates the 9-point Piotroski F-Score (YoY Comparison)."""
    try:
        # Fetch data (Annual)
        income = ticker_obj.financials
        balance = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        # We need at least 2 years of data
        if income.shape[1] < 2: return None, "Insufficient data for YoY comparison."

        # Indices: 0 is current year, 1 is previous year
        f_score = 0
        details = []

        # --- PROFITABILITY (4 Points) ---
        # 1. Positive Net Income
        ni_current = income.loc['Net Income'].iloc[0]
        if ni_current > 0: f_score += 1
        details.append(("Net Income > 0", "✅" if ni_current > 0 else "❌"))

        # 2. Positive Operating Cash Flow
        ocf_current = cashflow.loc['Operating Cash Flow'].iloc[0]
        if ocf_current > 0: f_score += 1
        details.append(("Cash Flow > 0", "✅" if ocf_current > 0 else "❌"))

        # 3. ROA Improvement
        assets_curr = balance.loc['Total Assets'].iloc[0]
        assets_prev = balance.loc['Total Assets'].iloc[1]
        roa_curr = ni_current / assets_curr
        roa_prev = income.loc['Net Income'].iloc[1] / assets_prev
        if roa_curr > roa_prev: f_score += 1
        details.append(("ROA Improving", "✅" if roa_curr > roa_prev else "❌"))

        # 4. Accrual (OCF > Net Income)
        if ocf_current > ni_current: f_score += 1
        details.append(("Quality of Earnings (OCF > NI)", "✅" if ocf_current > ni_current else "❌"))

        # --- LEVERAGE & LIQUIDITY (3 Points) ---
        # 5. Lower Long-Term Debt Ratio
        try:
            lt_debt_curr = balance.loc['Long Term Debt'].iloc[0] / assets_curr
            lt_debt_prev = balance.loc['Long Term Debt'].iloc[1] / assets_prev
            if lt_debt_curr < lt_debt_prev: f_score += 1
            details.append(("Lower Debt Ratio", "✅" if lt_debt_curr < lt_debt_prev else "❌"))
        except: details.append(("Lower Debt Ratio", "⚠️ N/A"))

        # 6. Higher Current Ratio
        cr_curr = balance.loc['Current Assets'].iloc[0] / balance.loc['Current Liabilities'].iloc[0]
        cr_prev = balance.loc['Current Assets'].iloc[1] / balance.loc['Current Liabilities'].iloc[1]
        if cr_curr > cr_prev: f_score += 1
        details.append(("Better Liquidity", "✅" if cr_curr > cr_prev else "❌"))

        # 7. No New Shares (Dilution)
        shares_curr = balance.loc['Ordinary Share Capital'].iloc[0]
        shares_prev = balance.loc['Ordinary Share Capital'].iloc[1]
        if shares_curr <= shares_prev: f_score += 1
        details.append(("No Share Dilution", "✅" if shares_curr <= shares_prev else "❌"))

        # --- EFFICIENCY (2 Points) ---
        # 8. Higher Gross Margin
        gm_curr = (income.loc['Total Revenue'].iloc[0] - income.loc['Cost Of Revenue'].iloc[0]) / income.loc['Total Revenue'].iloc[0]
        gm_prev = (income.loc['Total Revenue'].iloc[1] - income.loc['Cost Of Revenue'].iloc[1]) / income.loc['Total Revenue'].iloc[1]
        if gm_curr > gm_prev: f_score += 1
        details.append(("Margin Improvement", "✅" if gm_curr > gm_prev else "❌"))

        # 9. Higher Asset Turnover
        at_curr = income.loc['Total Revenue'].iloc[0] / assets_curr
        at_prev = income.loc['Total Revenue'].iloc[1] / assets_prev
        if at_curr > at_prev: f_score += 1
        details.append(("Asset Efficiency", "✅" if at_curr > at_prev else "❌"))

        return f_score, details
    except Exception as e:
        return None, str(e)

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

                tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News", "⚖️ Fundamental Scorecard"])

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
                    st.header("🛡️ Piotroski F-Score (YoY Health)")
                    
                    with st.spinner("Calculating year-over-year balance sheet improvements..."):
                        f_score, f_details = calculate_piotroski_score(ticker)
                    
                    if f_score is not None:
                        col_f1, col_f2 = st.columns([1, 2])
                        with col_f1:
                            st.metric("Piotroski Score", f"{f_score} / 9")
                            if f_score >= 8: st.success("🎯 Elite Financial Strength")
                            elif f_score >= 5: st.warning("⚖️ Stable")
                            else: st.error("🚨 Warning: Deteriorating Health")
                        
                        with col_f2:
                            # Create a nice 2-column list for details
                            df_f = pd.DataFrame(f_details, columns=["Metric", "Status"])
                            st.table(df_f)
                    else:
                        st.error(f"F-Score calculation failed: {f_details}")
                
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
                            
        except Exception as e:
            st.error(f"Error: {e}")
