# CTRADER OPEN API INTEGRATION - FIXED VERSION FOR STREAMLIT
# Replace your main.py with this code

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import hashlib
import urllib.parse
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 FxPro cTrader AI Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 30px;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .connection-success {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 2px 10px rgba(40,167,69,0.3);
    }
    
    .connection-failed {
        background: linear-gradient(45deg, #dc3545, #fd7e14);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 2px 10px rgba(220,53,69,0.3);
    }
    
    .account-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .signal-buy {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(40,167,69,0.3);
        border: 2px solid #155724;
    }
    
    .signal-sell {
        background: linear-gradient(45deg, #dc3545, #fd7e14);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(220,53,69,0.3);
        border: 2px solid #721c24;
    }
    
    .signal-hold {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
        color: #212529;
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,193,7,0.3);
        border: 2px solid #b8860b;
    }
    
    .auth-instructions {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ctrader_connected' not in st.session_state:
    st.session_state.ctrader_connected = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 0.0
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0.0
if 'trades_today' not in st.session_state:
    st.session_state.trades_today = 0
if 'auth_code' not in st.session_state:
    st.session_state.auth_code = None

class cTraderOpenAPI:
    """Proper cTrader Open API Integration using OAuth 2.0"""
    
    def __init__(self):
        # Official cTrader Open API endpoints
        self.auth_url = "https://openapi.ctrader.com/apps/auth"
        self.token_url = "https://openapi.ctrader.com/apps/token"
        
        # WebSocket endpoints for live trading
        self.live_host = "live.ctraderapi.com"
        self.demo_host = "demo.ctraderapi.com"
        self.port_json = 5036
        self.port_protobuf = 5035
        
        self.access_token = None
        self.refresh_token = None
        self.connected = False
        self.account_id = None
        
        # App credentials - USER MUST REGISTER THEIR OWN APP
        self.client_id = None
        self.client_secret = None
        self.redirect_uri = "https://www.google.com"  # Temporary redirect for getting auth code
    
    def generate_auth_url(self, client_id, redirect_uri):
        """Generate OAuth authorization URL"""
        try:
            # OAuth 2.0 parameters
            params = {
                'response_type': 'code',
                'client_id': client_id,
                'redirect_uri': redirect_uri,
                'scope': 'trading'
            }
            
            # Build authorization URL
            auth_url = f"{self.auth_url}?" + urllib.parse.urlencode(params)
            return auth_url
            
        except Exception as e:
            st.error(f"Error generating auth URL: {str(e)}")
            return None
    
    def exchange_code_for_token(self, client_id, client_secret, auth_code, redirect_uri):
        """Exchange authorization code for access token"""
        try:
            # Prepare token exchange request
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': redirect_uri,
                'client_id': client_id,
                'client_secret': client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            # Make token request
            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                
                if 'accessToken' in token_data:
                    self.access_token = token_data['accessToken']
                    self.refresh_token = token_data.get('refreshToken')
                    self.connected = True
                    
                    st.session_state.access_token = self.access_token
                    st.session_state.ctrader_connected = True
                    
                    return True, "✅ Successfully obtained access token!"
                else:
                    return False, f"❌ No access token in response: {token_data}"
            else:
                return False, f"❌ Token exchange failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"❌ Token exchange error: {str(e)}"
    
    def get_account_info(self):
        """Get account information (simulated for now)"""
        if not self.connected:
            return None
        
        # For now, return simulated account data
        # In a real implementation, you'd use WebSocket connection to get actual data
        return {
            'account_id': self.account_id or "demo_account",
            'balance': 10000.00,
            'equity': 10000.00 + st.session_state.daily_pnl,
            'currency': 'USD',
            'leverage': '1:100',
            'status': 'Connected via OAuth 2.0'
        }
    
    def simulate_trade(self, symbol, side, volume):
        """Simulate trade execution"""
        # Simulate realistic P&L
        pnl_change = np.random.uniform(-25, 45)
        st.session_state.daily_pnl += pnl_change
        st.session_state.trades_today += 1
        
        return True, f"✅ {side} {volume} units {symbol} - Trade simulated (P&L: {pnl_change:+.2f})"
    
    def disconnect(self):
        """Disconnect from cTrader"""
        self.connected = False
        self.access_token = None
        self.refresh_token = None
        st.session_state.ctrader_connected = False
        st.session_state.access_token = None

class TechnicalIndicators:
    """Technical indicators for analysis"""
    
    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma

class AITradingEngine:
    """AI Trading Engine"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze_symbol(self, symbol):
        """Analyze trading symbol"""
        try:
            # Get market data
            symbol_map = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'USDCAD=X'
            }
            
            yahoo_symbol = symbol_map.get(symbol, f"{symbol}=X")
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="5d", interval="5m")
            
            if data.empty or len(data) < 50:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Insufficient data',
                    'indicators': {}
                }
            
            # Calculate indicators
            current_price = data['Close'].iloc[-1]
            sma_20 = self.indicators.sma(data['Close'], 20).iloc[-1]
            sma_50 = self.indicators.sma(data['Close'], 50).iloc[-1]
            rsi = self.indicators.rsi(data['Close']).iloc[-1]
            
            # Generate signals
            signals = []
            reasons = []
            
            # Moving Average Analysis
            if current_price > sma_20 > sma_50:
                signals.append('BUY')
                reasons.append("Bullish MA alignment")
            elif current_price < sma_20 < sma_50:
                signals.append('SELL')
                reasons.append("Bearish MA alignment")
            
            # RSI Analysis
            if rsi < 30:
                signals.append('BUY')
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signals.append('SELL')
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self.indicators.bollinger_bands(data['Close'])
            if current_price <= bb_lower.iloc[-1]:
                signals.append('BUY')
                reasons.append("Price at lower Bollinger Band")
            elif current_price >= bb_upper.iloc[-1]:
                signals.append('SELL')
                reasons.append("Price at upper Bollinger Band")
            
            # Final decision
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            if buy_count > sell_count and buy_count >= 2:
                final_signal = 'BUY'
                confidence = min(buy_count / 3.0, 1.0)
            elif sell_count > buy_count and sell_count >= 2:
                final_signal = 'SELL'
                confidence = min(sell_count / 3.0, 1.0)
            else:
                final_signal = 'HOLD'
                confidence = 0.3
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'reasons': reasons,
                'indicators': {
                    'price': current_price,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rsi': rsi
                },
                'chart_data': data
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Analysis error: {str(e)}',
                'indicators': {}
            }

def create_price_chart(data, symbol):
    """Create price chart"""
    if data.empty:
        return None
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    # Add SMA
    sma_20 = TechnicalIndicators.sma(data['Close'], 20)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=sma_20,
        name='SMA 20',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title=f'{symbol} - Live Chart',
        height=400,
        xaxis_title="Time",
        yaxis_title="Price",
        template='plotly_dark'
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 FXPRO CTRADER AI TRADING SYSTEM - FIXED</div>', unsafe_allow_html=True)
    
    # Initialize components
    if 'ctrader_api' not in st.session_state:
        st.session_state.ctrader_api = cTraderOpenAPI()
    
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = AITradingEngine()
    
    api = st.session_state.ctrader_api
    ai_engine = st.session_state.ai_engine
    
    # Connection status
    if st.session_state.ctrader_connected:
        st.markdown('<div class="connection-success">🟢 CONNECTED TO CTRADER OPEN API</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection-failed">🔴 NOT CONNECTED TO CTRADER OPEN API</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ CTRADER OPEN API CONTROL")
        
        if not st.session_state.ctrader_connected:
            st.subheader("🔧 Setup Required")
            
            # Instructions
            st.markdown("""
            <div class="auth-instructions">
            <h4>📋 Setup Instructions:</h4>
            <ol>
            <li><strong>Register Application:</strong><br/>
            Go to <a href="https://openapi.ctrader.com" target="_blank">openapi.ctrader.com</a><br/>
            Create account & register new app</li>
            
            <li><strong>Get Credentials:</strong><br/>
            Copy Client ID & Client Secret from your registered app</li>
            
            <li><strong>Set Redirect URI:</strong><br/>
            Add this redirect URI in your app settings:<br/>
            <code>https://www.google.com</code></li>
            
            <li><strong>Authorize:</strong><br/>
            Use the generated link below to authorize</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🔑 App Credentials")
            client_id = st.text_input("Client ID:", help="From your registered cTrader app")
            client_secret = st.text_input("Client Secret:", type="password", help="From your registered cTrader app")
            
            if client_id and client_secret:
                # Generate authorization URL
                redirect_uri = "https://www.google.com"
                auth_url = api.generate_auth_url(client_id, redirect_uri)
                
                if auth_url:
                    st.success("✅ Authorization URL generated!")
                    st.markdown(f"""
                    **Step 1:** Click this link to authorize:
                    
                    [🔗 Authorize cTrader App]({auth_url})
                    
                    **Step 2:** After clicking "Allow Access", copy the `code` parameter from the redirected URL
                    """)
                    
                    auth_code = st.text_input("Authorization Code:", help="Copy from URL after authorization")
                    
                    if auth_code:
                        if st.button("🚀 Connect to cTrader", type="primary"):
                            with st.spinner("Exchanging code for access token..."):
                                success, message = api.exchange_code_for_token(
                                    client_id, client_secret, auth_code, redirect_uri
                                )
                                if success:
                                    st.success(message)
                                    st.balloons()
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(message)
        
        else:
            st.success("✅ cTrader Connected!")
            
            # Account info
            account = api.get_account_info()
            if account:
                st.write(f"**Account:** {account['account_id']}")
                st.write(f"**Balance:** ${account['balance']:,.2f}")
                st.write(f"**Status:** {account['status']}")
            
            # System controls
            st.subheader("🚀 System Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ START", type="primary"):
                    st.session_state.system_running = True
                    st.success("System started!")
                    st.rerun()
            
            with col2:
                if st.button("⏹️ STOP"):
                    st.session_state.system_running = False
                    st.warning("System stopped!")
                    st.rerun()
            
            if st.button("🔌 Disconnect"):
                api.disconnect()
                st.rerun()
    
    # Main content
    if st.session_state.ctrader_connected:
        
        # Account metrics
        account = api.get_account_info()
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="account-metric">💰 Balance<br/>${account["balance"]:,.2f}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="account-metric">📈 Equity<br/>${account["equity"]:,.2f}</div>', unsafe_allow_html=True)
            
            with col3:
                pnl_color = "🟢" if st.session_state.daily_pnl >= 0 else "🔴"
                st.markdown(f'<div class="account-metric">{pnl_color} Daily P&L<br/>${st.session_state.daily_pnl:,.2f}</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="account-metric">📊 Trades<br/>{st.session_state.trades_today}</div>', unsafe_allow_html=True)
        
        # Trading analysis
        if st.session_state.system_running:
            st.subheader("📊 LIVE TRADING ANALYSIS")
            
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
            selected_symbols = st.multiselect(
                "Select symbols to analyze:",
                symbols,
                default=['EURUSD', 'GBPUSD']
            )
            
            for symbol in selected_symbols:
                with st.expander(f"📈 {symbol} Analysis", expanded=True):
                    
                    # Get AI analysis
                    analysis = ai_engine.analyze_symbol(symbol)
                    
                    # Display signal
                    signal = analysis['signal']
                    confidence = analysis['confidence']
                    
                    signal_class = f"signal-{signal.lower()}"
                    st.markdown(f'<div class="{signal_class}">{signal} - Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                    
                    # Analysis details
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Analysis:**")
                        for reason in analysis['reasons']:
                            st.write(f"• {reason}")
                    
                    with col2:
                        if 'indicators' in analysis:
                            indicators = analysis['indicators']
                            st.write("**Indicators:**")
                            if 'price' in indicators:
                                st.write(f"Price: {indicators['price']:.5f}")
                            if 'rsi' in indicators:
                                st.write(f"RSI: {indicators['rsi']:.1f}")
                    
                    # Trading section
                    if signal in ['BUY', 'SELL'] and confidence > 0.5:
                        st.subheader("⚡ Execute Trade")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            volume = st.selectbox("Volume:", [1000, 5000, 10000], key=f"vol_{symbol}")
                        
                        with col2:
                            st.write(f"**Signal:** {signal}")
                            st.write(f"**Confidence:** {confidence:.1%}")
                        
                        with col3:
                            if st.button(f"🚀 {signal} {symbol}", key=f"trade_{symbol}", type="primary"):
                                with st.spinner("Executing trade..."):
                                    success, message = api.simulate_trade(symbol, signal, volume)
                                    if success:
                                        st.success(message)
                                        st.balloons()
                                        time.sleep(2)
                                        st.rerun()
                                    else:
                                        st.error(message)
                    
                    # Chart
                    if 'chart_data' in analysis:
                        chart = create_price_chart(analysis['chart_data'], symbol)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
        
        else:
            st.info("🔄 Click START to begin AI analysis")
    
    else:
        # Show setup instructions when not connected
        st.info("""
        🔧 **Setup Required**
        
        To connect to your live cTrader account, you need to:
        
        1. **Register an application** at [openapi.ctrader.com](https://openapi.ctrader.com)
        2. **Get your Client ID and Client Secret** from the registered app
        3. **Follow the OAuth authorization flow** in the sidebar
        
        This ensures secure connection to your real cTrader account!
        """)
    
    # Auto-refresh
    if st.session_state.system_running:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
