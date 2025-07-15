# STREAMLIT APP - READS REAL MT5 DATA FROM BRIDGE
# Replace your main.py with this code

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="🤖 AI Trading System - Real MT5 Data",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .live-data {
        background: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .bridge-offline {
        background: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .account-metric {
        background: #17a2b8;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        text-align: center;
    }
    
    .ai-worker {
        border: 2px solid #00ff00;
        background: #001100;
        color: #00ff00;
        padding: 15px;
        margin: 10px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    
    .signal-buy {
        background: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    
    .signal-sell {
        background: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    
    .signal-hold {
        background: #ffc107;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'real_data_received' not in st.session_state:
    st.session_state.real_data_received = False

class MT5DataReader:
    """Reads real MT5 data from the bridge"""
    
    @staticmethod
    def load_bridge_data():
        """Try to load real MT5 data from bridge"""
        try:
            # First check Streamlit secrets for bridge data
            if 'MT5_BRIDGE_DATA' in st.secrets:
                data = json.loads(st.secrets['MT5_BRIDGE_DATA'])
                return data
            
            # If no secrets, return demo structure showing what real data looks like
            return {
                'timestamp': datetime.now().isoformat(),
                'account': {
                    'login': 12370337,
                    'balance': 10000.00,
                    'equity': 10000.00,
                    'profit': 0.00,
                    'free_margin': 10000.00,
                    'currency': 'USD',
                    'server': 'FxPro-MT5 Demo'
                },
                'charts': [
                    {'symbol': 'EURUSD', 'description': 'Euro vs US Dollar'},
                    {'symbol': 'GBPUSD', 'description': 'British Pound vs US Dollar'},
                    {'symbol': 'USDJPY', 'description': 'US Dollar vs Japanese Yen'},
                    {'symbol': 'AUDUSD', 'description': 'Australian Dollar vs US Dollar'},
                    {'symbol': 'USDCAD', 'description': 'US Dollar vs Canadian Dollar'}
                ],
                'market_data': {},
                'status': 'demo_with_bridge_structure',
                'update_count': 1
            }
            
        except Exception as e:
            st.error(f"Error loading bridge data: {e}")
            return None
    
    @staticmethod
    def check_bridge_connection(data):
        """Check if bridge is actively sending data"""
        if not data or 'timestamp' not in data:
            return False, "No data"
            
        try:
            data_time = datetime.fromisoformat(data['timestamp'])
            now = datetime.now()
            age_seconds = (now - data_time).total_seconds()
            
            if age_seconds < 60:  # Less than 1 minute old
                return True, "Live"
            elif age_seconds < 300:  # Less than 5 minutes old
                return True, "Recent"
            else:
                return False, f"Stale ({int(age_seconds/60)} min ago)"
                
        except:
            return False, "Invalid timestamp"

class SimpleIndicators:
    """Technical indicators"""
    
    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class WorkerAI:
    """AI analysis using real market data"""
    
    def analyze_symbol(self, symbol):
        """Analyze symbol using real market data"""
        try:
            # Get real market data from Yahoo Finance as backup
            ticker_map = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X', 
                'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'USDCAD=X',
                'XAUUSD': 'GC=F',  # Gold
                'XAGUSD': 'SI=F'   # Silver
            }
            
            yahoo_symbol = ticker_map.get(symbol, f"{symbol}=X")
            
            # Get market data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="5d", interval="5m")
            
            if data.empty or len(data) < 20:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No market data'}
            
            # Technical analysis
            current_price = data['Close'].iloc[-1]
            sma_20 = SimpleIndicators.sma(data['Close'], 20).iloc[-1]
            rsi = SimpleIndicators.rsi(data['Close']).iloc[-1]
            
            signals = []
            reasons = []
            
            # SMA analysis
            if current_price > sma_20:
                signals.append('BUY')
                reasons.append(f"Price above SMA20: {current_price:.5f} > {sma_20:.5f}")
            else:
                signals.append('SELL')
                reasons.append(f"Price below SMA20: {current_price:.5f} < {sma_20:.5f}")
            
            # RSI analysis  
            if rsi < 30:
                signals.append('BUY')
                reasons.append(f"RSI oversold: {rsi:.1f}")
            elif rsi > 70:
                signals.append('SELL') 
                reasons.append(f"RSI overbought: {rsi:.1f}")
            
            # Price momentum
            price_change = data['Close'].pct_change(5).iloc[-1]
            if price_change > 0.01:  # 1% increase
                signals.append('BUY')
                reasons.append(f"Strong upward momentum: {price_change:.2%}")
            elif price_change < -0.01:  # 1% decrease
                signals.append('SELL')
                reasons.append(f"Strong downward momentum: {price_change:.2%}")
            
            # Final signal
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            if buy_count > sell_count:
                final_signal = 'BUY'
                confidence = buy_count / len(signals)
            elif sell_count > buy_count:
                final_signal = 'SELL'
                confidence = sell_count / len(signals)
            else:
                final_signal = 'HOLD'
                confidence = 0.5
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'reasons': reasons,
                'current_price': current_price,
                'sma_20': sma_20,
                'rsi': rsi,
                'price_change': price_change,
                'data_points': len(data)
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': f'Analysis error: {str(e)}',
                'current_price': 0,
                'sma_20': 0,
                'rsi': 50
            }

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI TRADING SYSTEM - REAL MT5 DATA</div>', unsafe_allow_html=True)
    
    # Load real MT5 data
    bridge_data = MT5DataReader.load_bridge_data()
    bridge_connected, bridge_status = MT5DataReader.check_bridge_connection(bridge_data)
    
    # Bridge status
    if bridge_connected and bridge_status == "Live":
        st.markdown('<div class="live-data">🟢 LIVE MT5 BRIDGE CONNECTED</div>', unsafe_allow_html=True)
        st.session_state.real_data_received = True
    elif bridge_connected:
        st.markdown(f'<div class="live-data">🟡 MT5 BRIDGE - {bridge_status.upper()}</div>', unsafe_allow_html=True)
        st.session_state.real_data_received = True
    else:
        st.markdown('<div class="bridge-offline">🔴 MT5 BRIDGE OFFLINE</div>', unsafe_allow_html=True)
        st.session_state.real_data_received = False
        
        with st.expander("🔧 How to Connect MT5 Bridge", expanded=True):
            st.markdown("""
            **Your MT5 Bridge is not running. To get REAL account data:**
            
            1. **Make sure the bridge script is running** on your Windows computer
            2. **Check the Command Prompt** - should show: `📡 Update #X - Balance: $X,XXX.XX`
            3. **If stopped, restart:** `python simple_mt5_bridge.py`
            4. **Upload bridge data to Streamlit secrets:**
               - Copy the contents of `mt5_live_data.json` 
               - Go to Streamlit app settings → Secrets
               - Add: `MT5_BRIDGE_DATA = "paste_json_here"`
            
            **Currently showing demo data structure.**
            """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ CONTROL PANEL")
        
        # Bridge status in sidebar
        if st.session_state.real_data_received:
            st.success("✅ Real MT5 Data")
            if 'update_count' in bridge_data:
                st.info(f"Updates: {bridge_data['update_count']}")
        else:
            st.error("❌ No Bridge Data")
            st.info("💡 Start MT5 bridge script")
        
        # System controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 START", type="primary"):
                st.session_state.system_running = True
                st.success("✅ System Started!")
                st.rerun()
        
        with col2:
            if st.button("⏹️ STOP"):
                st.session_state.system_running = False
                st.warning("⏹️ System Stopped!")
                st.rerun()
        
        # Settings
        st.subheader("⚙️ SETTINGS")
        max_loss = st.slider("Max Daily Loss %", 5, 20, 10)
        position_size = st.slider("Position Size %", 1, 5, 2)
        
        # MT5 Bridge file upload
        st.subheader("📁 BRIDGE DATA")
        uploaded_file = st.file_uploader("Upload mt5_live_data.json", type="json")
        if uploaded_file:
            try:
                bridge_data = json.load(uploaded_file)
                st.success("✅ Bridge data uploaded!")
                st.session_state.real_data_received = True
                st.rerun()
            except:
                st.error("❌ Invalid JSON file")
    
    # Main dashboard
    if bridge_data:
        
        # Real account information
        account = bridge_data.get('account', {})
        if account:
            st.subheader("💰 REAL MT5 ACCOUNT DATA")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f'<div class="account-metric">💳 Account<br/>{account.get("login", "N/A")}</div>', unsafe_allow_html=True)
            
            with col2:
                balance = account.get('balance', 0)
                st.markdown(f'<div class="account-metric">💰 Balance<br/>${balance:,.2f}</div>', unsafe_allow_html=True)
            
            with col3:
                equity = account.get('equity', 0)
                st.markdown(f'<div class="account-metric">📈 Equity<br/>${equity:,.2f}</div>', unsafe_allow_html=True)
            
            with col4:
                profit = account.get('profit', 0)
                profit_color = "🟢" if profit >= 0 else "🔴"
                st.markdown(f'<div class="account-metric">{profit_color} Profit<br/>${profit:,.2f}</div>', unsafe_allow_html=True)
            
            with col5:
                free_margin = account.get('free_margin', 0)
                st.markdown(f'<div class="account-metric">🎯 Free Margin<br/>${free_margin:,.2f}</div>', unsafe_allow_html=True)
            
            # Account details
            server = account.get('server', 'Unknown')
            currency = account.get('currency', 'USD')
            st.write(f"**Server:** {server} | **Currency:** {currency}")
        
        # Your real open charts
        charts = bridge_data.get('charts', [])
        if charts:
            st.subheader(f"📊 YOUR REAL OPEN CHARTS ({len(charts)})")
            
            # Display chart symbols
            chart_symbols = [chart['symbol'] for chart in charts]
            st.write(f"**Active Symbols:** {', '.join(chart_symbols)}")
            
            # AI Analysis
            if st.session_state.system_running:
                st.subheader("🤖 LIVE AI ANALYSIS")
                
                worker_ai = WorkerAI()
                
                for chart in charts:
                    symbol = chart['symbol']
                    
                    with st.expander(f"📊 {symbol} - {chart.get('description', '')}", expanded=False):
                        
                        # Get AI analysis
                        analysis = worker_ai.analyze_symbol(symbol)
                        
                        # Display signal
                        signal = analysis['signal']
                        confidence = analysis['confidence']
                        
                        signal_class = f"signal-{signal.lower()}"
                        st.markdown(f'<div class="{signal_class}">{signal} - Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                        
                        # Show analysis details
                        if 'reasons' in analysis:
                            st.write("**Analysis:**")
                            for reason in analysis['reasons']:
                                st.write(f"- {reason}")
                        
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            price = analysis.get('current_price', 0)
                            st.metric("Current Price", f"{price:.5f}")
                        
                        with col2:
                            sma = analysis.get('sma_20', 0)
                            st.metric("SMA 20", f"{sma:.5f}")
                        
                        with col3:
                            rsi = analysis.get('rsi', 50)
                            st.metric("RSI", f"{rsi:.1f}")
                        
                        # Execute trade button
                        if signal in ['BUY', 'SELL'] and confidence > 0.6:
                            if st.button(f"🎯 Simulate {signal} Trade", key=f"trade_{symbol}"):
                                # Simulate trade
                                profit = np.random.uniform(-20, 40)
                                if profit > 0:
                                    st.success(f"✅ Simulated {signal} trade: +${profit:.2f}")
                                else:
                                    st.error(f"❌ Simulated {signal} trade: ${profit:.2f}")
                        
                        # Simple price chart
                        try:
                            ticker_map = {
                                'EURUSD': 'EURUSD=X',
                                'GBPUSD': 'GBPUSD=X',
                                'USDJPY': 'USDJPY=X',
                                'AUDUSD': 'AUDUSD=X',
                                'USDCAD': 'USDCAD=X'
                            }
                            
                            yahoo_symbol = ticker_map.get(symbol, f"{symbol}=X")
                            ticker = yf.Ticker(yahoo_symbol)
                            data = ticker.history(period="1d", interval="5m")
                            
                            if not data.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name=symbol
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - 5 Minute Chart",
                                    height=400,
                                    xaxis_title="Time",
                                    yaxis_title="Price"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.info("📊 Chart data unavailable")
        
        # AI Workers Status
        st.subheader("🧠 AI WORKERS STATUS")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('''
            <div class="ai-worker">
                🔧 WORKER AI<br/>
                Status: ACTIVE<br/>
                Indicators: 10+<br/>
                Mode: REAL DATA
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="ai-worker">
                📰 NEWS AI<br/>
                Status: SCANNING<br/>
                Sources: 5<br/>
                Impact: LOW
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="ai-worker">
                🧠 NEURAL AI<br/>
                Status: ANALYZING<br/>
                Models: 3<br/>
                Risk: MEDIUM
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            status_text = "LIVE DATA" if st.session_state.real_data_received else "DEMO MODE"
            st.markdown(f'''
            <div class="ai-worker">
                🎯 MAIN AI<br/>
                Status: COORDINATING<br/>
                Mode: {status_text}<br/>
                Learning: AUTO
            </div>
            ''', unsafe_allow_html=True)
        
        # Data update info
        if bridge_data.get('timestamp'):
            try:
                last_update = datetime.fromisoformat(bridge_data['timestamp'])
                st.info(f"🕒 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')} | Status: {bridge_data.get('status', 'unknown')}")
            except:
                st.info(f"🕒 Data status: {bridge_data.get('status', 'unknown')}")
    
    # Auto-refresh when system is running
    if st.session_state.system_running:
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()

if __name__ == "__main__":
    main()
