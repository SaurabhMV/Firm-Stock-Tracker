import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import google.generativeai as genai
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- CONFIG ---
st.set_page_config(page_title="Pro Stock Analyst", layout="wide")

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.title("⚙️ Setup")
    # Check if key is in secrets, otherwise use text input
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded from Secrets!")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    ticker_input = st.text_input("Stock Ticker", value="GOOG").upper()
    analyze_btn = st.button("Generate Deep Research")

# --- HELPER FUNCTIONS ---

def get_analyst_data(ticker):
    """Fetches Analyst recommendations and targets safely."""
    info = ticker.info
    current = info.get('currentPrice')
    target = info.get('targetMeanPrice')
    
    # Calculate upside only if data exists
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
        
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Support/Resistance
    # Find local minima (supports)
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [df.Close.iloc[i] for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    
    return {
        "RSI": df['RSI'].iloc[-1], 
        "Supports": supports, 
        "Price": df['Close'].iloc[-1]
    }

def get_best_model_name():
    """Dynamically finds the newest available Gemini Flash model."""
    try:
        # Get all models that support generating content
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Filter for 'flash' models and sort them to get the latest version
        flash_models = [m for m in models if 'flash' in m.lower()]
        flash_models.sort()
        
        if flash_models:
            return flash_models[-1]  # Returns the newest one (e.g., gemini-3.0-flash)
    except Exception:
        pass
    
    # Fallback if discovery fails
    return "models/gemini-flash-latest"

def sanitize_link(link):
    """
    Forces all links to be absolute URLs. 
    Fixes the 'redirect to Streamlit' issue.
    """
    if not link or link == "#":
        # Fallback: Just go to Yahoo Finance homepage if link is missing
        return "https://finance.yahoo.com"
    
    # 1. If it already has http/https, it's good.
    if link.lower().startswith("http"):
        return link
    
    # 2. If it starts with a slash (e.g., "/news/story..."), prepend domain
    if link.startswith("/"):
        return f"https://finance.yahoo.com{link}"
        
    # 3. If it's a relative path without slash (e.g., "news/story..."), prepend domain + slash
    return f"https://finance.yahoo.com/{link}"
    
def generate_pro_report(symbol, info, tech, news, api_key):
    """Generates the AI analysis using dynamic model discovery."""
    genai.configure(api_key=api_key)
    
    # --- DYNAMIC MODEL DISCOVERY (Fixes 404 Errors) ---
    try:
        # Find all models that support content generation
        models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods 
            and 'flash' in m.name.lower()
        ]
        # Sort to get the latest version (e.g., Gemini 1.5 or 2.0)
        models.sort()
        target_model = models[-1] if models else "models/gemini-1.5-flash"
    except Exception:
        target_model = "models/gemini-1.5-flash"

    model = genai.GenerativeModel(target_model)
    
    prompt = f"""
    Act as a Hedge Fund Strategy Lead. Analyze {symbol} for a long-term investor.
    
    CONTEXT:
    - Business: {info.get('longBusinessSummary', 'N/A')[:1000]}
    - Fundamentals: P/E {info.get('forwardPE', 'N/A')}, Margin {info.get('profitMargins', 'N/A')}, Debt/Equity {info.get('debtToEquity', 'N/A')}
    - Technicals: RSI {tech.get('RSI', 50):.2f}, Key Support Levels: {tech.get('Supports')}
    - Recent Headlines: {news}

    YOUR GOAL: Provide a balanced investment thesis.
    
    STRUCTURE:
    1. THE BULL CASE: Why buy now? (Key growth drivers)
    2. THE BEAR CASE: What could go wrong? (Risks)
    3. VALUATION CHECK: Is it overvalued?
    4. FINAL VERDICT: Buy/Hold/Sell with a Confidence Score (0-100%).
    """
    return model.generate_content(prompt).text

# --- MAIN APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Gemini API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker_input}..."):
                # 1. Fetch Data
                ticker = yf.Ticker(ticker_input)
                history = ticker.history(period="1y")
                
                if history.empty:
                    st.error(f"Could not fetch data for {ticker_input}. Check the ticker symbol.")
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

                # --- TABS FOR ANALYSIS ---
                tab1, tab2, tab3 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News"])

                # TAB 1: CHARTS
                with tab1:
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        fig = go.Figure(data=[go.Candlestick(x=history.index, 
                                        open=history['Open'], high=history['High'], 
                                        low=history['Low'], close=history['Close'])])
                        fig.update_layout(title=f"{ticker_input} 1-Year Price Action", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with col_b:
                        st.write("### Key Fundamentals")
                        fund_data = {
                            "Market Cap": info.get("marketCap"),
                            "Trailing P/E": info.get("trailingPE"),
                            "Forward P/E": info.get("forwardPE"),
                            "Revenue Growth": info.get("revenueGrowth"),
                            "Profit Margins": info.get("profitMargins")
                        }
                        st.table(pd.DataFrame.from_dict(fund_data, orient='index', columns=['Value']))

                # TAB 2: AI THESIS
                with tab2:
                    st.subheader("🤖 Gemini Investment Memo")
                    # Prepare news titles for the prompt
                    news_titles = [n.get('title') for n in ticker.news[:5]]
                    report = generate_pro_report(ticker_input, info, tech_data, news_titles, api_key)
                    st.markdown(report)

                # TAB 3: NEWS FEED (Fixed Indentation & Links)
                # TAB 3: NEWS FEED (Fixed Links)# --- TAB 3: AI NEWS BRIEFING ---
# --- TAB 3: AI NEWS BRIEFING ---
                with tab3:
                    st.subheader("⚡ AI Executive News Briefing")
                    
                    news_items = ticker.news[:10]
                    
                    if not news_items:
                        st.info("No recent news found for this ticker.")
                    else:
                        # --- PART 1: GENERATE AI SUMMARY ---
                        # Combine headlines into a single string
                        news_text = ""
                        for n in news_items:
                            # Convert timestamp to YYYY-MM-DD
                            ts = n.get('providerPublishTime', 0)
                            pub_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            news_text += f"- {pub_time}: {n.get('title')}\n"
                        
                        try:
                            # --- FIXED: Use the helper function to get the model ---
                            model_name = get_best_model_name() 
                            news_model = genai.GenerativeModel(model_name)
                            
                            news_prompt = f"""
                            You are a financial news anchor. Here are the latest headlines for {ticker_input}:
                            {news_text}
                            
                            Task:
                            1. Summarize the general sentiment (Bullish/Bearish/Neutral).
                            2. Group these stories into 3 key themes.
                            3. Be extremely concise (bullet points).
                            """
                            with st.spinner(f"Analyzing news with {model_name}..."):
                                news_summary = news_model.generate_content(news_prompt).text
                                st.success(news_summary)
                                
                        except Exception as e:
                            st.warning(f"Could not generate summary: {e}")

                        st.divider()

                        # --- PART 2: TIMELINE LIST ---
                        st.subheader("📜 Timeline of Events")
                        
                        for n in news_items:
                            title = n.get('title', 'No Title')
                            ts = n.get('providerPublishTime', 0)
                            readable_date = datetime.datetime.fromtimestamp(ts).strftime('%b %d, %I:%M %p')
                            publisher = n.get('publisher', 'Unknown Source')
                            
                            # Thumbnail Logic
                            thumbnail_url = None
                            if 'thumbnail' in n and 'resolutions' in n['thumbnail']:
                                resolutions = n['thumbnail'].get('resolutions', [])
                                if resolutions:
                                    thumbnail_url = resolutions[-1].get('url')

                            # Render Row
                            with st.container():
                                col_img, col_txt = st.columns([1, 5])
                                with col_img:
                                    if thumbnail_url:
                                        st.image(thumbnail_url, use_container_width=True)
                                    else:
                                        st.write("📰")
                                with col_txt:
                                    st.markdown(f"**{title}**")
                                    st.caption(f"🕒 {readable_date}  |  Source: {publisher}")
                                st.divider()
                                
        except Exception as e:
            st.error(f"An error occurred: {e}")
