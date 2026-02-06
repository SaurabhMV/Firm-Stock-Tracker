import streamlit as st
import yfinance as yf
import time
import requests
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Smart Interval Monitor", page_icon="⏲️", layout="wide")
st.title("⏲️ Responsive Stock Monitor")

# --- Sidebar ---
st.sidebar.header("Settings")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
chat_id = st.sidebar.text_input("Authorized Chat ID")
ticker_input = st.sidebar.text_input("Tickers", value="AAPL, TSLA, ZGLD.TO")
drop_threshold = st.sidebar.slider("Alert Threshold (%)", 1, 20, 5)

# --- RE-ADDED FEATURE: Check Interval ---
check_interval = st.sidebar.number_input(
    "Refresh Rate (seconds)", 
    min_value=30, 
    value=120, 
    help="How often to check stock prices. Bot remains responsive every 10s."
)

# --- State Management ---
if 'running' not in st.session_state: st.session_state.running = False
if 'last_update_id' not in st.session_state: st.session_state.last_update_id = 0
if 'last_fetch_time' not in st.session_state: st.session_state.last_fetch_time = 0
if 'results_cache' not in st.session_state: st.session_state.results_cache = []

# --- Helper Functions ---
def send_telegram(msg):
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try: requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
        except: pass

def fetch_data(tickers):
    results = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)
            df_3d = t.history(period='3d', interval='1h')
            df_today = t.history(period='1d', interval='1m')
            if not df_3d.empty and not df_today.empty:
                curr = df_today['Close'].iloc[-1]
                h3 = df_3d['High'].max()
                open_p = df_today['Open'].iloc[0]
                pullback = ((curr - h3) / h3) * 100
                day_chg = ((curr - open_p) / open_p) * 100
                results.append({"Ticker": symbol, "Price": curr, "Pullback": pullback, "DayChg": day_chg})
        except: continue
    return results

def check_bot_commands():
    """Polls Telegram for commands without blocking the UI."""
    if not bot_token or not chat_id: return False
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"offset": st.session_state.last_update_id + 1, "timeout": 1}
    try:
        r = requests.get(url, params=params).json()
        if r.get("result"):
            for update in r["result"]:
                st.session_state.last_update_id = update["update_id"]
                msg = update.get("message", {})
                if str(msg.get("chat", {}).get("id")) != str(chat_id).strip(): continue
                text = msg.get("text", "").lower()
                
                if "/start" in text:
                    st.session_state.running = True
                    send_telegram("🚀 *Authorized: Monitor Started*")
                    return True
                elif "/stop" in text:
                    st.session_state.running = False
                    send_telegram("🛑 *Authorized: Monitor Stopped*")
                    return True
                elif "/status" in text:
                    send_telegram("⌛ *Fetching current status...*")
                    data = fetch_data([t.strip().upper() for t in ticker_input.split(",")])
                    status_msg = "📊 *Current Status:*\n"
                    for i in data: status_msg += f"*{i['Ticker']}*: ${i['Price']:.2f} ({i['Pullback']:.2f}%)\n"
                    send_telegram(status_msg)
    except: pass
    return False

# --- Main App Execution ---
# Always check commands first
if check_bot_commands(): st.rerun()

status_txt = "🟢 RUNNING" if st.session_state.running else "🔴 STOPPED"
st.sidebar.markdown(f"**System Status:** {status_txt}")

if st.session_state.running:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    # 1. Timer Logic: Only fetch if interval has passed
    current_time = time.time()
    if (current_time - st.session_state.last_fetch_time) >= check_interval:
        with st.spinner("Updating prices..."):
            st.session_state.results_cache = fetch_data(tickers)
            st.session_state.last_fetch_time = current_time
            
            # 2. Alert Logic
            for item in st.session_state.results_cache:
                if item['Pullback'] <= -drop_threshold:
                    send_telegram(f"🚨 *{item['Ticker']} ALERT*\nDown `{abs(item['Pullback']):.2f}%` from 3D High.")

    # 3. Display Data
    if st.session_state.results_cache:
        st.dataframe(pd.DataFrame(st.session_state.results_cache), use_container_width=True, hide_index=True)
        st.caption(f"Last Price Sync: {time.strftime('%H:%M:%S')}. Next sync in {int(check_interval - (time.time() - st.session_state.last_fetch_time))}s.")

    # 4. Short Sleep for Bot Responsiveness
    time.sleep(10)
    st.rerun()
else:
    st.info("System Standby. Bot commands: `/start`, `/stop`, `/status`")
    time.sleep(10)
    st.rerun()
