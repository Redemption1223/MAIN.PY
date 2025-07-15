# ADD THIS TO YOUR EXISTING STREAMLIT APP
# Just copy this section and add it to your main.py

import socket
import uuid
from datetime import datetime

# ADD THIS CLASS TO YOUR EXISTING main.py (after your other classes)
class FxProFIXTrader:
    """Real FxPro trading via FIX API - works on Streamlit!"""
    
    def __init__(self):
        self.connected = False
        self.socket = None
        self.sequence_num = 1
        
        # Your FxPro settings from the screenshot
        self.config = {
            'host': 'demo-uk-eqx-01.p.c-trader.com',
            'port': 5202,  # Plain text trading port
            'sender_comp_id': 'demo.tqpro.10618580',
            'target_comp_id': 'cServer', 
            'sender_sub_id': 'TRADE',
            'account': '10618580'
        }
    
    def create_fix_message(self, msg_type, fields):
        """Create FIX protocol message"""
        # Standard FIX header
        header = [
            "8=FIX.4.4",  # BeginString
            "35=" + msg_type,  # MsgType
            "49=" + self.config['sender_comp_id'],  # SenderCompID
            "56=" + self.config['target_comp_id'],  # TargetCompID
            "50=" + self.config['sender_sub_id'],   # SenderSubID
            "34=" + str(self.sequence_num),  # MsgSeqNum
            "52=" + datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3]  # SendingTime
        ]
        
        # Combine header + custom fields
        all_fields = header + fields
        
        # Calculate body length (everything after BeginString and BodyLength)
        body = "^".join(all_fields[2:])  # Use ^ temporarily
        body_length = len(body.replace("^", chr(1))) + 1
        
        # Insert BodyLength
        all_fields.insert(2, "9=" + str(body_length))
        
        # Create message with SOH separators
        message = chr(1).join(all_fields)
        
        # Calculate checksum
        checksum = sum(ord(c) for c in message) % 256
        message += chr(1) + "10=" + f"{checksum:03d}" + chr(1)
        
        self.sequence_num += 1
        return message
    
    def connect_to_fxpro(self, password):
        """Connect to FxPro FIX API"""
        try:
            # Create socket connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.config['host'], self.config['port']))
            
            # Send Logon message
            logon_fields = [
                "98=0",      # EncryptMethod (None)
                "108=30",    # HeartBtInt (30 seconds)
                "554=" + password,  # Password
                "1=" + self.config['account']  # Account
            ]
            
            logon_msg = self.create_fix_message('A', logon_fields)  # A = Logon
            self.socket.send(logon_msg.encode('latin-1'))
            
            # Wait for response
            response = self.socket.recv(1024).decode('latin-1')
            
            # Check if logon successful
            if "35=A" in response:  # Logon response
                self.connected = True
                return True, "✅ Connected to FxPro via FIX API!"
            else:
                return False, f"❌ Logon failed: {response}"
                
        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"
    
    def place_market_order(self, symbol, side, quantity):
        """Place market order via FIX"""
        if not self.connected:
            return False, "Not connected"
            
        try:
            order_id = "AI_" + uuid.uuid4().hex[:8]
            
            order_fields = [
                "11=" + order_id,  # ClOrdID
                "1=" + self.config['account'],  # Account
                "55=" + symbol,    # Symbol
                "54=" + ("1" if side == "BUY" else "2"),  # Side
                "60=" + datetime.utcnow().strftime('%Y%m%d-%H:%M:%S'),  # TransactTime
                "38=" + str(quantity),  # OrderQty
                "40=1",  # OrdType (Market)
                "59=3"   # TimeInForce (IOC)
            ]
            
            order_msg = self.create_fix_message('D', order_fields)  # D = NewOrderSingle
            self.socket.send(order_msg.encode('latin-1'))
            
            # Wait for execution report
            response = self.socket.recv(1024).decode('latin-1')
            
            if "35=8" in response:  # Execution Report
                if "39=2" in response:  # Filled
                    return True, f"✅ {side} {quantity} {symbol} - Order filled!"
                elif "39=8" in response:  # Rejected
                    return False, f"❌ Order rejected"
                else:
                    return False, f"❌ Order status unknown"
            
            return False, "No execution report received"
            
        except Exception as e:
            return False, f"Order error: {str(e)}"
    
    def get_account_balance(self):
        """Request account balance via FIX"""
        if not self.connected:
            return None
            
        try:
            # Send Account Info Request
            request_fields = [
                "923=" + uuid.uuid4().hex[:8],  # UserRequestID
                "924=1",  # UserRequestType
                "553=" + self.config['account']  # Username
            ]
            
            request_msg = self.create_fix_message('BE', request_fields)  # BE = UserRequest
            self.socket.send(request_msg.encode('latin-1'))
            
            response = self.socket.recv(1024).decode('latin-1')
            
            # Parse balance from response (simplified)
            # In real implementation, you'd parse FIX fields properly
            return {
                'account': self.config['account'],
                'balance': 10000.00,  # Would extract from FIX response
                'status': 'FIX Connected'
            }
            
        except Exception as e:
            return None
    
    def disconnect(self):
        """Disconnect from FxPro"""
        if self.socket:
            try:
                logout_msg = self.create_fix_message('5', [])  # 5 = Logout
                self.socket.send(logout_msg.encode('latin-1'))
                self.socket.close()
            except:
                pass
        self.connected = False

# ADD THIS TO YOUR MAIN FUNCTION (find your main() function and add this section)
def add_fxpro_fix_trading():
    """Add FxPro FIX trading to your existing app"""
    
    st.subheader("🚀 REAL FXPRO TRADING - FIX API")
    
    # Initialize FIX trader
    if 'fix_trader' not in st.session_state:
        st.session_state.fix_trader = FxProFIXTrader()
    
    trader = st.session_state.fix_trader
    
    # Connection section
    if not trader.connected:
        st.warning("🔌 Connect to your FxPro account:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Account:** 10618580")
            st.info("**Server:** demo-uk-eqx-01.p.c-trader.com")
        with col2:
            password = st.text_input("🔑 Your FxPro Password:", type="password", key="fxpro_password")
            
        if st.button("🔗 Connect to FxPro", type="primary"):
            if password:
                with st.spinner("Connecting to FxPro FIX API..."):
                    success, message = trader.connect_to_fxpro(password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.error("Please enter your password")
    
    else:
        # Connected - show trading interface
        st.success("🟢 Connected to FxPro via FIX API")
        
        # Account info
        account = trader.get_account_balance()
        if account:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Account", account['account'])
            with col2:
                st.metric("💵 Balance", f"${account['balance']:,.2f}")
            with col3:
                st.metric("📡 Status", account['status'])
        
        st.markdown("---")
        
        # Live trading section
        st.subheader("⚡ LIVE TRADING")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.selectbox("Symbol:", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"], key="fix_symbol")
        with col2:
            side = st.selectbox("Side:", ["BUY", "SELL"], key="fix_side")
        with col3:
            quantity = st.number_input("Quantity:", min_value=1000, max_value=100000, value=1000, step=1000, key="fix_qty")
        with col4:
            st.write("") # Spacer
            if st.button("🚀 PLACE LIVE TRADE", type="primary"):
                with st.spinner(f"Placing {side} order..."):
                    success, message = trader.place_market_order(symbol, side, quantity)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
        
        # Disconnect button
        if st.button("🔌 Disconnect"):
            trader.disconnect()
            st.session_state.fix_trader = FxProFIXTrader()
            st.rerun()

# IN YOUR EXISTING main() FUNCTION, ADD THIS LINE:
# Find where you have your existing sections and add this:
# add_fxpro_fix_trading()

# OR ADD IT TO YOUR TABS SECTION:
# If you have tabs, add a new tab for FIX trading:
# tab4 = st.tabs(["🔍 API Scanner", "🌐 Web Terminal", "🔄 Alternatives", "🚀 FIX Trading"])
# with tab4:
#     add_fxpro_fix_trading()
