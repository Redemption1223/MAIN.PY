# CTRADER REAL LIVE TRADING SYSTEM
# ACTUAL WebSocket connection to your real cTrader account

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import websocket
import ssl
import threading
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔥 LIVE cTrader Trading System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #FF0000, #FF6B6B);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 30px;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
    }
    
    .live-trading {
        background: linear-gradient(45deg, #FF0000, #FF4444);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        box-shadow: 0 2px 10px rgba(255,0,0,0.3);
        animation: pulse 3s infinite;
    }
    
    .connection-status {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .account-metric {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF0000 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .signal-buy {
        background: linear-gradient(45deg, #00FF00, #00AA00);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,255,0,0.3);
        border: 2px solid #004400;
    }
    
    .signal-sell {
        background: linear-gradient(45deg, #FF0000, #AA0000);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
        border: 2px solid #440000;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ctrader_ws_connected' not in st.session_state:
    st.session_state.ctrader_ws_connected = False
if 'real_account_balance' not in st.session_state:
    st.session_state.real_account_balance = 0.0
if 'real_account_equity' not in st.session_state:
    st.session_state.real_account_equity = 0.0
if 'live_trades' not in st.session_state:
    st.session_state.live_trades = []
if 'connection_log' not in st.session_state:
    st.session_state.connection_log = []
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
if 'account_authorized' not in st.session_state:
    st.session_state.account_authorized = False

class cTraderWebSocketClient:
    """REAL cTrader WebSocket API Client"""
    
    def __init__(self):
        # REAL cTrader Open API endpoints
        self.demo_host = "wss://demo.ctraderapi.com:5036"
        self.live_host = "wss://live.ctraderapi.com:5036"
        
        self.ws = None
        self.connected = False
        self.authorized = False
        
        # Your credentials
        self.client_id = "16128_1N2FGw1faESealOA"
        self.client_secret = None
        self.access_token = "FZVyeFsxKkElJrvinCQxoTPSRu7ryZXd8Qn66szleKk"
        self.account_id = "10618580"
        
        # Message ID counter
        self.msg_id = 1
        
    def log_message(self, message):
        """Log connection messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        st.session_state.connection_log.append(log_entry)
        # Keep only last 20 logs
        if len(st.session_state.connection_log) > 20:
            st.session_state.connection_log = st.session_state.connection_log[-20:]
    
    def on_open(self, ws):
        """WebSocket connection opened"""
        self.connected = True
        st.session_state.ctrader_ws_connected = True
        self.log_message("🔥 CONNECTED to cTrader WebSocket!")
        
        # Send application authentication
        self.authenticate_application()
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            # cTrader sends JSON messages
            data = json.loads(message)
            
            self.log_message(f"📨 Received: {data.get('payloadType', 'Unknown')}")
            
            # Handle different message types
            if data.get('payloadType') == 'PROTO_OA_APPLICATION_AUTH_RES':
                self.handle_app_auth_response(data)
            elif data.get('payloadType') == 'PROTO_OA_ACCOUNT_AUTH_RES':
                self.handle_account_auth_response(data)
            elif data.get('payloadType') == 'PROTO_OA_TRADER_RES':
                self.handle_trader_response(data)
            elif data.get('payloadType') == 'PROTO_OA_EXECUTION_EVENT':
                self.handle_execution_event(data)
            elif data.get('payloadType') == 'ERROR_RES':
                self.handle_error_response(data)
            else:
                self.log_message(f"📋 Unknown message type: {data.get('payloadType')}")
                
        except json.JSONDecodeError:
            self.log_message("❌ Failed to parse message as JSON")
        except Exception as e:
            self.log_message(f"❌ Message handling error: {str(e)}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.log_message(f"❌ WebSocket Error: {str(error)}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        self.connected = False
        self.authorized = False
        st.session_state.ctrader_ws_connected = False
        st.session_state.account_authorized = False
        self.log_message(f"🔌 Connection closed: {close_status_code} - {close_msg}")
    
    def authenticate_application(self):
        """Send application authentication message"""
        auth_msg = {
            "clientMsgId": str(self.msg_id),
            "payloadType": "PROTO_OA_APPLICATION_AUTH_REQ",
            "payload": {
                "clientId": self.client_id,
                "clientSecret": self.client_secret or "dummy_secret"
            }
        }
        
        self.send_message(auth_msg)
        self.msg_id += 1
        self.log_message("🔐 Sent application authentication...")
    
    def authenticate_account(self):
        """Send account authentication message"""
        if not self.access_token:
            self.log_message("❌ No access token provided")
            return
        
        auth_msg = {
            "clientMsgId": str(self.msg_id),
            "payloadType": "PROTO_OA_ACCOUNT_AUTH_REQ",
            "payload": {
                "ctidTraderAccountId": int(self.account_id),
                "accessToken": self.access_token
            }
        }
        
        self.send_message(auth_msg)
        self.msg_id += 1
        self.log_message("🔑 Sent account authentication...")
    
    def send_message(self, message):
        """Send message to WebSocket"""
        try:
            if self.ws and self.connected:
                json_msg = json.dumps(message)
                self.ws.send(json_msg)
                return True
            else:
                self.log_message("❌ Cannot send - not connected")
                return False
        except Exception as e:
            self.log_message(f"❌ Send error: {str(e)}")
            return False
    
    def handle_app_auth_response(self, data):
        """Handle application authentication response"""
        if data.get('payload', {}).get('errorCode'):
            self.log_message(f"❌ App auth failed: {data['payload']['errorCode']}")
        else:
            self.log_message("✅ Application authenticated!")
            # Now authenticate account
            self.authenticate_account()
    
    def handle_account_auth_response(self, data):
        """Handle account authentication response"""
        if data.get('payload', {}).get('errorCode'):
            error_code = data['payload']['errorCode']
            self.log_message(f"❌ Account auth failed: {error_code}")
        else:
            self.authorized = True
            st.session_state.account_authorized = True
            self.log_message("🔥 ACCOUNT AUTHENTICATED! LIVE TRADING ACTIVE!")
            
            # Request account info
            self.request_trader_info()
    
    def handle_trader_response(self, data):
        """Handle trader information response"""
        payload = data.get('payload', {})
        trader = payload.get('trader', {})
        
        if trader:
            balance = trader.get('balance', 0) / 100  # cTrader sends in cents
            st.session_state.real_account_balance = balance
            st.session_state.real_account_equity = balance  # Simplified
            
            self.log_message(f"💰 Real Account Balance: ${balance:,.2f}")
    
    def handle_execution_event(self, data):
        """Handle trade execution events"""
        execution = data.get('payload', {})
        order = execution.get('order', {})
        
        if order:
            trade_info = {
                'orderId': order.get('orderId'),
                'symbol': order.get('symbolId'),
                'side': order.get('orderType'),
                'volume': order.get('requestedVolume', 0) / 100,
                'status': order.get('orderStatus'),
                'time': datetime.now().strftime("%H:%M:%S")
            }
            
            st.session_state.live_trades.append(trade_info)
            self.log_message(f"📈 Trade executed: {trade_info['symbol']} {trade_info['side']}")
    
    def handle_error_response(self, data):
        """Handle error responses"""
        error_code = data.get('errorCode', 'Unknown')
        description = data.get('description', 'No description')
        self.log_message(f"❌ API Error {error_code}: {description}")
    
    def request_trader_info(self):
        """Request trader account information"""
        trader_req = {
            "clientMsgId": str(self.msg_id),
            "payloadType": "PROTO_OA_TRADER_REQ",
            "payload": {
                "ctidTraderAccountId": int(self.account_id)
            }
        }
        
        self.send_message(trader_req)
        self.msg_id += 1
    
    def place_market_order(self, symbol, side, volume):
        """Place a real market order"""
        if not self.authorized:
            return False, "Account not authorized"
        
        order_msg = {
            "clientMsgId": str(self.msg_id),
            "payloadType": "PROTO_OA_NEW_ORDER_REQ",
            "payload": {
                "ctidTraderAccountId": int(self.account_id),
                "symbolId": symbol,
                "orderType": "MARKET",
                "tradeSide": side.upper(),
                "volume": int(volume * 100),  # cTrader expects volume in cents
                "timeInForce": "IMMEDIATE_OR_CANCEL",
                "comment": "AI Trading System"
            }
        }
        
        success = self.send_message(order_msg)
        self.msg_id += 1
        
        if success:
            self.log_message(f"🚀 LIVE ORDER SENT: {side} {volume} {symbol}")
            return True, f"Live order placed: {side} {volume} {symbol}"
        else:
            return False, "Failed to send order"
    
    def connect(self, use_demo=True):
        """Connect to cTrader WebSocket"""
        try:
            host = self.demo_host if use_demo else self.live_host
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Enable WebSocket debugging
            websocket.enableTrace(False)  # Set to True for debugging
            
            self.log_message(f"🔌 Connecting to {host}...")
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                host,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run in a separate thread
            def run_websocket():
                self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            ws_thread = threading.Thread(target=run_websocket, daemon=True)
            ws_thread.start()
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
        self.connected = False
        self.authorized = False

class TechnicalIndicators:
    """Technical analysis indicators"""
    
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

class AITradingEngine:
    """AI Trading Engine for live trading"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def analyze_symbol(self, symbol):
        """Real-time market analysis"""
        try:
            symbol_map = {
                'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X'
            }
            
            yahoo_symbol = symbol_map.get(symbol, f"{symbol}=X")
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="1d", interval="5m")
            
            if data.empty or len(data) < 20:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Technical analysis
            current_price = data['Close'].iloc[-1]
            sma_20 = self.indicators.sma(data['Close'], 20).iloc[-1]
            rsi = self.indicators.rsi(data['Close']).iloc[-1]
            
            # Generate signals
            signals = []
            reasons = []
            
            if current_price > sma_20 and rsi < 30:
                signals.append('BUY')
                reasons.append(f"Price above SMA20 + RSI oversold ({rsi:.1f})")
            elif current_price < sma_20 and rsi > 70:
                signals.append('SELL')
                reasons.append(f"Price below SMA20 + RSI overbought ({rsi:.1f})")
            
            # Decision
            if 'BUY' in signals:
                return {'signal': 'BUY', 'confidence': 0.8, 'reasons': reasons, 'price': current_price}
            elif 'SELL' in signals:
                return {'signal': 'SELL', 'confidence': 0.8, 'reasons': reasons, 'price': current_price}
            else:
                return {'signal': 'HOLD', 'confidence': 0.3, 'reasons': ['Market conditions unclear'], 'price': current_price}
                
        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Analysis error: {str(e)}'}

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔥 REAL LIVE CTRADER TRADING SYSTEM 🔥</div>', unsafe_allow_html=True)
    
    # Initialize components
    if 'ws_client' not in st.session_state or st.session_state.ws_client is None:
        st.session_state.ws_client = cTraderWebSocketClient()
    
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = AITradingEngine()
    
    ws_client = st.session_state.ws_client
    ai_engine = st.session_state.ai_engine
    
    # Connection status
    if st.session_state.account_authorized:
        st.markdown('<div class="live-trading">🔥 LIVE TRADING ACTIVE - REAL MONEY ACCOUNT CONNECTED 🔥</div>', unsafe_allow_html=True)
    elif st.session_state.ctrader_ws_connected:
        st.markdown('<div class="live-trading">🔌 CONNECTED TO CTRADER - AUTHENTICATING ACCOUNT...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="live-trading">🚀 READY TO CONNECT TO YOUR LIVE CTRADER ACCOUNT</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔥 LIVE TRADING CONTROL")
        
        # Connection controls
        st.subheader("🔌 WebSocket Connection")
        
        # Your live trading tokens
        access_token = st.text_input(
            "Access Token:", 
            value="FZVyeFsxKkElJrvinCQxoTPSRu7ryZXd8Qn66szleKk",
            type="password"
        )
        
        client_secret = st.text_input(
            "Client Secret:",
            type="password",
            help="Required for some brokers"
        )
        
        account_id = st.text_input("Account ID:", value="10618580")
        
        # Update credentials
        if access_token:
            ws_client.access_token = access_token
            ws_client.client_secret = client_secret
            ws_client.account_id = account_id
        
        # Connection buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 CONNECT LIVE", type="primary"):
                ws_client.log_message("🚀 Attempting LIVE connection...")
                success = ws_client.connect(use_demo=False)
                if not success:
                    st.error("Failed to initiate connection")
                else:
                    st.success("Connection initiated!")
                    time.sleep(2)
                    st.rerun()
        
        with col2:
            if st.button("🧪 DEMO MODE"):
                ws_client.log_message("🧪 Attempting DEMO connection...")
                success = ws_client.connect(use_demo=True)
                if not success:
                    st.error("Failed to initiate connection")
                else:
                    st.success("Demo connection initiated!")
                    time.sleep(2)
                    st.rerun()
        
        if st.button("🔌 Disconnect"):
            ws_client.disconnect()
            st.rerun()
        
        # Connection log
        st.subheader("📋 Connection Log")
        if st.session_state.connection_log:
            for log in st.session_state.connection_log[-5:]:  # Show last 5
                st.text(log)
        else:
            st.text("No logs yet...")
        
        # Clear logs
        if st.button("🗑️ Clear Logs"):
            st.session_state.connection_log = []
            st.rerun()
    
    # Account information
    if st.session_state.account_authorized:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            balance = st.session_state.real_account_balance
            st.markdown(f'<div class="account-metric">💰 REAL Balance<br/>${balance:,.2f}</div>', unsafe_allow_html=True)
        
        with col2:
            equity = st.session_state.real_account_equity
            st.markdown(f'<div class="account-metric">📈 REAL Equity<br/>${equity:,.2f}</div>', unsafe_allow_html=True)
        
        with col3:
            trades_count = len(st.session_state.live_trades)
            st.markdown(f'<div class="account-metric">🔥 Live Trades<br/>{trades_count}</div>', unsafe_allow_html=True)
        
        with col4:
            status = "LIVE TRADING" if st.session_state.account_authorized else "CONNECTING"
            st.markdown(f'<div class="account-metric">⚡ Status<br/>{status}</div>', unsafe_allow_html=True)
    
    # Trading section
    if st.session_state.account_authorized:
        st.subheader("🤖 AI LIVE TRADING")
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        selected_symbols = st.multiselect(
            "Select currency pairs for LIVE trading:",
            symbols,
            default=['EURUSD'],
            help="⚠️ WARNING: This will place REAL trades with REAL money!"
        )
        
        for symbol in selected_symbols:
            with st.expander(f"🔥 {symbol} - LIVE TRADING", expanded=True):
                
                # AI analysis
                analysis = ai_engine.analyze_symbol(symbol)
                signal = analysis['signal']
                confidence = analysis['confidence']
                current_price = analysis.get('price', 0)
                
                # Display signal
                signal_class = f"signal-{signal.lower()}"
                st.markdown(f'<div class="{signal_class}">🎯 AI SIGNAL: {signal} | Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                
                st.write(f"**Current Price:** {current_price:.5f}")
                
                # Analysis details
                if 'reasons' in analysis:
                    st.write("**AI Analysis:**")
                    for reason in analysis['reasons']:
                        st.write(f"• {reason}")
                
                # LIVE TRADING SECTION
                if signal in ['BUY', 'SELL'] and confidence > 0.6:
                    st.subheader("⚠️ LIVE TRADE EXECUTION")
                    
                    st.warning("🚨 WARNING: This will place a REAL trade with REAL money! 🚨")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        volume = st.selectbox("Volume (units):", [1000, 5000, 10000], key=f"vol_{symbol}")
                    
                    with col2:
                        st.write(f"**Signal:** {signal}")
                        st.write(f"**Confidence:** {confidence:.1%}")
                    
                    with col3:
                        if st.button(f"🔥 LIVE {signal} {symbol}", key=f"trade_{symbol}", type="primary"):
                            with st.spinner("🚀 PLACING LIVE TRADE..."):
                                success, message = ws_client.place_market_order(symbol, signal, volume/100)  # Convert to lots
                                
                                if success:
                                    st.success(f"🔥 LIVE TRADE PLACED!\n\n{message}")
                                    st.balloons()
                                else:
                                    st.error(f"❌ Trade failed: {message}")
                                
                                time.sleep(3)
                                st.rerun()
    
    elif st.session_state.ctrader_ws_connected:
        st.info("🔑 Connected to cTrader WebSocket. Waiting for account authentication...")
        st.write("Check the connection log in the sidebar for details.")
    
    else:
        st.warning("""
        🔥 **LIVE TRADING SETUP**
        
        1. **Enter your live trading tokens** in the sidebar
        2. **Click "CONNECT LIVE"** for real money trading
        3. **Click "DEMO MODE"** for testing
        
        ⚠️ **WARNING**: LIVE mode uses REAL MONEY from your cTrader account!
        """)
    
    # Live trades history
    if st.session_state.live_trades:
        st.subheader("📊 LIVE TRADES HISTORY")
        
        df = pd.DataFrame(st.session_state.live_trades)
        st.dataframe(df, use_container_width=True)
    
    # Connection status details
    st.subheader("🔌 Connection Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown(f"""
        <div class="connection-status">
        <strong>WebSocket:</strong> {'🟢 Connected' if st.session_state.ctrader_ws_connected else '🔴 Disconnected'}<br/>
        <strong>Account:</strong> {'🟢 Authorized' if st.session_state.account_authorized else '🔴 Not Authorized'}<br/>
        <strong>Trading:</strong> {'🔥 LIVE ACTIVE' if st.session_state.account_authorized else '❌ Inactive'}
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        if st.session_state.connection_log:
            st.write("**Recent Activity:**")
            for log in st.session_state.connection_log[-3:]:
                st.text(log)
    
    # Footer warning
    st.markdown("---")
    st.error("""
    🚨 **LIVE TRADING WARNING** 🚨
    
    This system connects to your REAL cTrader account and can place REAL trades with REAL money.
    
    • ✅ **DEMO MODE** - Safe testing with virtual money
    • ⚠️ **LIVE MODE** - Real trades with your actual account balance
    
    **Use DEMO MODE first to test the system!**
    """)
    
    # Auto-refresh connection status
    if st.session_state.ctrader_ws_connected:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
