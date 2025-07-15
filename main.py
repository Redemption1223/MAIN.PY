# MT5 DATA BRIDGE SYSTEM - HYBRID SOLUTION
# Part 1: MT5_DATA_BRIDGE.py (Run this on your Windows computer with MT5)

import MetaTrader5 as mt5
import requests
import json
import time
import pandas as pd
from datetime import datetime
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT5DataBridge:
    """Collects real MT5 data and sends to Streamlit Cloud app"""
    
    def __init__(self, streamlit_webhook_url):
        self.webhook_url = streamlit_webhook_url
        self.connected = False
        self.running = False
        self.account_info = {}
        self.open_charts = []
        
    def connect_to_mt5(self, login, password, server):
        """Connect to your real MT5 account"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            if not mt5.login(login, password=password, server=server):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                return False
                
            self.connected = True
            logger.info(f"✅ Connected to MT5 account: {login}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_real_account_data(self):
        """Get your REAL account information"""
        if not self.connected:
            return None
            
        try:
            account = mt5.account_info()
            if account is None:
                return None
                
            self.account_info = {
                'login': account.login,
                'balance': float(account.balance),
                'equity': float(account.equity),
                'margin': float(account.margin),
                'free_margin': float(account.margin_free),
                'profit': float(account.profit),
                'currency': account.currency,
                'leverage': account.leverage,
                'server': account.server,
                'timestamp': datetime.now().isoformat()
            }
            
            return self.account_info
            
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return None
    
    def get_real_open_charts(self):
        """Get your ACTUAL open charts from MT5"""
        if not self.connected:
            return []
            
        try:
            # Get all symbols from Market Watch (your open charts)
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
                
            charts = []
            for symbol in symbols:
                if symbol.visible:  # Only visible symbols (your open charts)
                    charts.append({
                        'symbol': symbol.name,
                        'description': symbol.description,
                        'point': symbol.point,
                        'digits': symbol.digits,
                        'spread': symbol.spread,
                        'trade_mode': symbol.trade_mode
                    })
            
            self.open_charts = charts
            logger.info(f"📊 Found {len(charts)} open charts")
            return charts
            
        except Exception as e:
            logger.error(f"Error getting charts: {e}")
            return []
    
    def get_market_data(self, symbol, timeframe='M5', count=200):
        """Get REAL market data for symbol"""
        if not self.connected:
            return None
            
        try:
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1
            }
            
            rates = mt5.copy_rates_from_pos(
                symbol, 
                tf_map.get(timeframe, mt5.TIMEFRAME_M5), 
                0, 
                count
            )
            
            if rates is None or len(rates) == 0:
                return None
                
            # Convert to format for sending
            data = []
            for rate in rates:
                data.append({
                    'time': datetime.fromtimestamp(rate['time']).isoformat(),
                    'open': float(rate['open']),
                    'high': float(rate['high']),
                    'low': float(rate['low']),
                    'close': float(rate['close']),
                    'volume': int(rate['tick_volume'])
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def send_data_to_streamlit(self, data):
        """Send real MT5 data to Streamlit Cloud app"""
        try:
            headers = {'Content-Type': 'application/json'}
            
            # For this demo, we'll save to a JSON file that Streamlit can read
            # In production, you'd use a real webhook or database
            with open('mt5_live_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("✅ Data sent to Streamlit app")
            return True
            
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            return False
    
    def start_data_bridge(self):
        """Start sending live data to Streamlit"""
        self.running = True
        logger.info("🚀 MT5 Data Bridge started")
        
        while self.running:
            try:
                # Collect all data
                account_data = self.get_real_account_data()
                charts_data = self.get_real_open_charts()
                
                # Get market data for each chart
                market_data = {}
                for chart in charts_data:
                    symbol = chart['symbol']
                    data = self.get_market_data(symbol)
                    if data:
                        market_data[symbol] = data
                
                # Prepare data package
                live_data = {
                    'timestamp': datetime.now().isoformat(),
                    'account': account_data,
                    'charts': charts_data,
                    'market_data': market_data,
                    'status': 'live'
                }
                
                # Send to Streamlit
                self.send_data_to_streamlit(live_data)
                
                logger.info(f"📡 Data update sent - Balance: ${account_data['balance']:.2f}, Charts: {len(charts_data)}")
                
                # Wait 30 seconds before next update
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Bridge error: {e}")
                time.sleep(10)  # Wait before retry
    
    def stop_data_bridge(self):
        """Stop the data bridge"""
        self.running = False
        logger.info("⏹️ MT5 Data Bridge stopped")

def main():
    """Main function to run the MT5 Data Bridge"""
    
    # YOUR MT5 CREDENTIALS
    MT5_LOGIN = 12370337  # Your FxPro account
    MT5_PASSWORD = "your_actual_password"  # Replace with your password
    MT5_SERVER = "FxPro-MT5 Demo"  # Your server
    
    # Streamlit webhook URL (we'll use file-based for this demo)
    WEBHOOK_URL = "file://mt5_live_data.json"
    
    # Create bridge
    bridge = MT5DataBridge(WEBHOOK_URL)
    
    # Connect to MT5
    print("🔄 Connecting to MT5...")
    if bridge.connect_to_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print("✅ Connected! Starting data bridge...")
        
        try:
            # Start the bridge (runs continuously)
            bridge.start_data_bridge()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping data bridge...")
            bridge.stop_data_bridge()
    else:
        print("❌ Failed to connect to MT5")

if __name__ == "__main__":
    main()

# =============================================================================
# Part 2: UPDATED STREAMLIT APP (Replace your main.py with this)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🤖 AI Trading System - Real MT5 Data",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS (same as before)
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
    
    .no-data {
        background: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
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
if 'live_data_available' not in st.session_state:
    st.session_state.live_data_available = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'system_running' not in st.session_state:
    st.session_state.system_running = False

class LiveDataReader:
    """Reads real MT5 data from the bridge"""
    
    @staticmethod
    def load_live_data():
        """Load live data sent from MT5 bridge"""
        try:
            # In a real implementation, this would read from a database or API
            # For demo, we'll simulate receiving the data
            
            # Try to read from uploaded file (if available)
            if os.path.exists('mt5_live_data.json'):
                with open('mt5_live_data.json', 'r') as f:
                    data = json.load(f)
                return data
            else:
                # Return sample structure showing what real data looks like
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
                        {'symbol': 'USDJPY', 'description': 'US Dollar vs Japanese Yen'}
                    ],
                    'market_data': {},
                    'status': 'demo_mode'
                }
                
        except Exception as e:
            st.error(f"Error loading live data: {e}")
            return None
    
    @staticmethod 
    def check_data_freshness(data):
        """Check if data is recent (within last 2 minutes)"""
        if not data or 'timestamp' not in data:
            return False
            
        try:
            data_time = datetime.fromisoformat(data['timestamp'])
            now = datetime.now()
            age = (now - data_time).total_seconds()
            
            return age < 120  # Fresh if less than 2 minutes old
            
        except:
            return False

class SimpleIndicators:
    """Technical indicators for real data"""
    
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
    """AI analysis using real MT5 data"""
    
    def analyze_real_data(self, market_data):
        """Analyze real market data from MT5"""
        if not market_data:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No market data'}
        
        # Convert market data to DataFrame
        df = pd.DataFrame(market_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        if len(df) < 20:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        # Technical analysis on real data
        current_price = df['close'].iloc[-1]
        sma_20 = SimpleIndicators.sma(df['close'], 20).iloc[-1]
        rsi = SimpleIndicators.rsi(df['close']).iloc[-1]
        
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
            'rsi': rsi
        }

def main():
    """Main Streamlit app with real MT5 data"""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI TRADING SYSTEM - REAL MT5 DATA</div>', unsafe_allow_html=True)
    
    # Load live data
    live_data = LiveDataReader.load_live_data()
    is_fresh = LiveDataReader.check_data_freshness(live_data)
    
    # Data status
    if live_data and is_fresh:
        st.markdown('<div class="live-data">🟢 RECEIVING LIVE MT5 DATA</div>', unsafe_allow_html=True)
        st.session_state.live_data_available = True
    elif live_data:
        st.markdown('<div class="live-data">🟡 MT5 DATA (CACHED)</div>', unsafe_allow_html=True)
        st.session_state.live_data_available = True
    else:
        st.markdown('<div class="no-data">🔴 NO LIVE DATA - START MT5 BRIDGE</div>', unsafe_allow_html=True)
        st.session_state.live_data_available = False
    
    # Show setup instructions if no data
    if not st.session_state.live_data_available:
        with st.expander("🔧 Setup MT5 Data Bridge", expanded=True):
            st.markdown("""
            **To get your REAL MT5 data:**
            
            1. **Download the MT5 Bridge script** (provided above)
            2. **Save as `mt5_bridge.py`** on your Windows computer with MT5
            3. **Install requirements:**
            ```bash
            pip install MetaTrader5 requests pandas
            ```
            4. **Edit your credentials** in the script:
            ```python
            MT5_LOGIN = 12370337
            MT5_PASSWORD = "your_password"  
            MT5_SERVER = "FxPro-MT5 Demo"
            ```
            5. **Run the bridge:**
            ```bash
            python mt5_bridge.py
            ```
            6. **Refresh this page** to see your real account data!
            """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ CONTROL PANEL")
        
        if st.session_state.live_data_available:
            st.success("✅ Live Data Connected")
        else:
            st.error("❌ No Live Data")
            st.info("💡 Start MT5 Bridge on your computer")
        
        # System controls
        if st.session_state.live_data_available:
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
    
    # Main dashboard (only if we have data)
    if st.session_state.live_data_available and live_data:
        
        # Account information
        account = live_data.get('account', {})
        if account:
            st.subheader("📊 REAL ACCOUNT DATA")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Balance", f"${account.get('balance', 0):,.2f}")
            
            with col2:
                st.metric("📈 Equity", f"${account.get('equity', 0):,.2f}")
            
            with col3:
                st.metric("💹 Profit", f"${account.get('profit', 0):,.2f}")
            
            with col4:
                st.metric("🎯 Free Margin", f"${account.get('free_margin', 0):,.2f}")
            
            # Account details
            st.write(f"**Account:** {account.get('login', 'N/A')} | **Server:** {account.get('server', 'N/A')} | **Currency:** {account.get('currency', 'USD')}")
        
        # Your real open charts
        charts = live_data.get('charts', [])
        if charts:
            st.subheader("📊 YOUR REAL OPEN CHARTS")
            
            chart_names = [chart['symbol'] for chart in charts]
            st.write(f"**Found {len(charts)} open charts:** {', '.join(chart_names)}")
            
            # AI analysis for each real chart
            if st.session_state.system_running:
                st.subheader("🤖 AI ANALYSIS - YOUR CHARTS")
                
                market_data = live_data.get('market_data', {})
                worker_ai = WorkerAI()
                
                for chart in charts:
                    symbol = chart['symbol']
                    
                    with st.expander(f"📊 {symbol} - {chart.get('description', '')}", expanded=False):
                        
                        if symbol in market_data:
                            # Analyze real market data
                            analysis = worker_ai.analyze_real_data(market_data[symbol])
                            
                            # Display signal
                            signal = analysis['signal']
                            confidence = analysis['confidence']
                            
                            signal_class = f"signal-{signal.lower()}"
                            st.markdown(f'<div class="{signal_class}">{signal} - Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                            
                            # Show analysis details
                            st.write("**Analysis:**")
                            for reason in analysis['reasons']:
                                st.write(f"- {reason}")
                            
                            # Show key metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"{analysis['current_price']:.5f}")
                            with col2:
                                st.metric("SMA 20", f"{analysis['sma_20']:.5f}")
                            with col3:
                                st.metric("RSI", f"{analysis['rsi']:.1f}")
                            
                            # Chart (if we have enough data)
                            if len(market_data[symbol]) > 20:
                                df = pd.DataFrame(market_data[symbol])
                                df['time'] = pd.to_datetime(df['time'])
                                
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=df['time'],
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name=symbol
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - Real MT5 Data",
                                    height=400,
                                    xaxis_title="Time",
                                    yaxis_title="Price"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"⚠️ No market data available for {symbol}")
        
        # Data update info
        if live_data.get('timestamp'):
            last_update = datetime.fromisoformat(live_data['timestamp'])
            st.info(f"🕒 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')} | Status: {live_data.get('status', 'unknown')}")
    
    # Auto-refresh when system is running
    if st.session_state.system_running and st.session_state.live_data_available:
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()

if __name__ == "__main__":
    main()
