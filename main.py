# ADD THIS TO YOUR EXISTING main.py FILE ON GITHUB

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests  # ← Make sure this import exists
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="🤖 AI Trading System - Universal",
    page_icon="🤖", 
    layout="wide"
)

# ADD THIS NEW FUNCTION - PUT IT RIGHT AFTER THE IMPORTS
def test_fxpro_api_discovery():
    """🔍 DISCOVER FXPRO'S HIDDEN APIs"""
    
    st.subheader("🔍 FxPro API Discovery Scanner")
    st.info("💡 Scanning for FxPro's trading APIs...")
    
    # Potential FxPro API endpoints
    endpoints_to_test = [
        "https://api.fxpro.com",
        "https://rest.fxpro.com", 
        "https://trade.fxpro.com/api",
        "https://www.fxpro.com/api",
        "https://webtrader.fxpro.com/api",
        "https://ctrader.fxpro.com/api",
        "https://fxpro.com/webapi",
        "https://mt5.fxpro.com/api",
        "https://api-live.fxpro.com",
        "https://api-demo.fxpro.com",
        "https://trading.fxpro.com/api",
        "https://platform.fxpro.com/api"
    ]
    
    found_apis = []
    
    for endpoint in endpoints_to_test:
        try:
            with st.spinner(f"Testing {endpoint}..."):
                response = requests.get(endpoint, timeout=5)
                
                if response.status_code == 200:
                    st.success(f"✅ **FOUND API!** {endpoint}")
                    st.json({
                        "url": endpoint,
                        "status": response.status_code,
                        "headers": dict(response.headers),
                        "content_preview": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    })
                    found_apis.append(endpoint)
                    
                elif response.status_code in [401, 403]:
                    st.warning(f"🔐 **PROTECTED API** {endpoint} - Status: {response.status_code} (Needs authentication)")
                    found_apis.append(f"{endpoint} (Protected)")
                    
                elif response.status_code != 404:
                    st.info(f"📡 **Potential API** {endpoint} - Status: {response.status_code}")
                    
                else:
                    st.write(f"❌ Not found: {endpoint}")
                    
        except requests.ConnectionError:
            st.write(f"🔌 Connection failed: {endpoint}")
        except requests.Timeout:
            st.write(f"⏰ Timeout: {endpoint}")
        except Exception as e:
            st.write(f"❓ Error testing {endpoint}: {str(e)}")
    
    # Summary
    if found_apis:
        st.success(f"🎉 **DISCOVERY COMPLETE!** Found {len(found_apis)} potential APIs:")
        for api in found_apis:
            st.write(f"• {api}")
            
        st.markdown("""
        **🎯 NEXT STEPS:**
        1. **Test these APIs** with your FxPro credentials
        2. **Check for API documentation** at these URLs
        3. **Contact FxPro support** and ask about API access
        4. **Try cTrader platform** if available
        """)
    else:
        st.warning("🔍 **No APIs found** - But this doesn't mean they don't exist!")
        st.markdown("""
        **💡 FxPro might have:**
        - **Private APIs** requiring special access
        - **cTrader platform** with separate API
        - **FIX API** for institutional clients
        - **Webhook integration** in web terminal
        
        **🎯 Try these:**
        1. **Login to FxPro client area** → Search "API"
        2. **Check cTrader platform** if available
        3. **Contact FxPro support** directly
        4. **Test web terminal** automation
        """)

def test_web_terminal_automation():
    """🌐 TEST WEB TERMINAL AUTOMATION"""
    
    st.subheader("🌐 Web Terminal Automation Test")
    
    st.markdown("""
    **📋 MANUAL TEST STEPS:**
    
    1. **Open new tab:** [FxPro Web Terminal](https://webtrader.fxpro.com)
    2. **Login** with your account: `12370337`
    3. **Open browser Dev Tools** (Press F12)
    4. **Go to Network tab** in Dev Tools
    5. **Place a small test trade** (0.01 lot)
    6. **Watch the Network requests** - look for API calls!
    7. **Copy any REST API endpoints** you see
    
    **🔍 Look for URLs containing:**
    - `/api/`
    - `/trade`
    - `/order`
    - `/account`
    - JSON responses
    """)
    
    if st.button("📋 I've checked the web terminal"):
        api_endpoint = st.text_input("🔗 Paste any API endpoint you found:")
        if api_endpoint:
            st.success(f"✅ Testing: {api_endpoint}")
            try:
                response = requests.get(api_endpoint, timeout=5)
                st.json({
                    "status": response.status_code,
                    "response": response.text[:500]
                })
            except Exception as e:
                st.error(f"Error: {e}")

def test_alternative_brokers():
    """🔄 TEST ALTERNATIVE BROKERS WITH GOOD APIS"""
    
    st.subheader("🔄 Alternative Broker APIs")
    st.info("💡 These brokers have excellent APIs that work on any platform:")
    
    brokers = {
        "OANDA": {
            "api_url": "https://api-fxpractice.oanda.com",
            "demo_signup": "https://www.oanda.com/demo-account/",
            "features": "Excellent REST API, same forex pairs, easy setup"
        },
        "Interactive Brokers": {
            "api_url": "https://api.ibkr.com",
            "demo_signup": "https://www.interactivebrokers.com/en/trading/free-trial.php",
            "features": "Professional API, institutional grade"
        },
        "Alpaca": {
            "api_url": "https://broker-api.alpaca.markets",
            "demo_signup": "https://alpaca.markets/",
            "features": "Modern API, crypto + forex"
        }
    }
    
    for broker, info in brokers.items():
        with st.expander(f"🏦 {broker} API Test"):
            st.write(f"**Features:** {info['features']}")
            st.write(f"**Demo Signup:** {info['demo_signup']}")
            
            if st.button(f"Test {broker} API", key=f"test_{broker}"):
                try:
                    response = requests.get(info['api_url'], timeout=5)
                    if response.status_code == 401:
                        st.success(f"✅ {broker} API is working! (Needs authentication)")
                    else:
                        st.info(f"📡 {broker} responded with status: {response.status_code}")
                except Exception as e:
                    st.warning(f"⚠️ {broker} API test: {e}")

# ADD THIS TO YOUR SIDEBAR IN THE MAIN FUNCTION
# FIND THE SIDEBAR SECTION AND ADD THIS:

def main():
    """Main Streamlit app - UPDATED WITH API DISCOVERY"""
    
    # Header (keep your existing header)
    st.markdown('<div class="main-header">🤖 AI TRADING SYSTEM - UNIVERSAL PLATFORM</div>', unsafe_allow_html=True)
    
    # Initialize APIs (keep your existing code)
    if 'forex_api' not in st.session_state:
        st.session_state.forex_api = UniversalForexAPI()
    
    forex_api = st.session_state.forex_api
    
    # Connection status (keep existing)
    st.markdown('<div class="account-connected">🟢 UNIVERSAL API CONNECTED - WORKS ANYWHERE!</div>', unsafe_allow_html=True)
    
    # ADD THIS NEW SECTION - PUT IT RIGHT AFTER THE CONNECTION STATUS
    # =================================================================
    st.subheader("🔍 REAL MT5 INTEGRATION DISCOVERY")
    
    # Add tabs for different discovery methods
    tab1, tab2, tab3 = st.tabs(["🔍 API Scanner", "🌐 Web Terminal", "🔄 Alternatives"])
    
    with tab1:
        test_fxpro_api_discovery()
    
    with tab2:
        test_web_terminal_automation()
    
    with tab3:
        test_alternative_brokers()
    
    st.markdown("---")  # Separator line
    # =================================================================
    
    # Sidebar (keep your existing sidebar code)
    with st.sidebar:
        st.header("🎛️ CONTROL PANEL")
        
        st.success("✅ Universal API Active")
        st.info("💡 Works on Railway, Vercel, Heroku, etc.")
        
        # ADD THIS TO YOUR SIDEBAR TOO:
        st.markdown("---")
        st.subheader("🔍 MT5 DISCOVERY")
        if st.button("🚀 Run FxPro Scanner"):
            st.rerun()
        
        # Keep the rest of your existing sidebar code...
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 START", type="primary"):
                st.session_state.system_running = True
                st.success("✅ Started!")
                st.rerun()
        
        with col2:
            if st.button("⏹️ STOP"):
                st.session_state.system_running = False
                st.warning("⏹️ Stopped!")
                st.rerun()
    
    # KEEP ALL YOUR EXISTING CODE BELOW THIS POINT
    # (Account dashboard, AI workers, trading analysis, etc.)
    
    # ... rest of your existing main() function code ...

# MAKE SURE YOU KEEP ALL YOUR EXISTING CLASSES:
# - UniversalForexAPI
# - SimpleIndicators  
# - WorkerAI
# - All the rest of your existing code

if __name__ == "__main__":
    main()
