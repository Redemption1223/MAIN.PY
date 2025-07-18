# Railway.app One-Click Deploy cTrader Bot
# Just fork this repo and deploy to Railway - NO CMD NEEDED!

import os
import asyncio
import websockets
import ssl
import json
import logging
import time
from datetime import datetime
import requests
import yfinance as yf
import smtplib
from email.mime.text import MimeText

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RailwaycTraderBot:
    """24/7 cTrader Bot for Railway.app deployment"""
    
    def __init__(self):
        # Get credentials from Railway environment variables
        self.access_token = os.getenv('CTRADER_ACCESS_TOKEN', 'FZVyeFsxKkElJrvinCQxoTPSRu7ryZXd8Qn66szleKk')
        self.refresh_token = os.getenv('CTRADER_REFRESH_TOKEN', 'I4M1fXeHOkFfLUDeozkHiA-uEwlHm_k8ZjWij02BQX0')
        self.client_id = os.getenv('CTRADER_CLIENT_ID', '16128_1N2FGw1faESealOA')
        self.account_id = os.getenv('CTRADER_ACCOUNT_ID', '10618580')
        self.email = os.getenv('NOTIFICATION_EMAIL', '')  # For trade alerts
        
        # Trading settings
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '5'))
        self.symbols = os.getenv('TRADING_SYMBOLS', 'EURUSD,GBPUSD,USDJPY').split(',')
        
        # WebSocket connection
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.message_id = 1
        self.daily_trades = 0
        self.account_balance = 0.0
        
        # Endpoints
        self.demo_ws = "wss://demo.ctraderapi.com:5036"
        self.live_ws = "wss://live.ctraderapi.com:5036"
    
    async def start(self):
        """Start the bot - Railway will keep this running 24/7"""
        logger.info("🚀 Railway cTrader Bot Starting...")
        
        while True:
            try:
                await self.connect_and_run()
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def connect_and_run(self):
        """Connect to cTrader and start trading"""
        endpoint = self.demo_ws if self.demo_mode else self.live_ws
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            async with websockets.connect(endpoint, ssl=ssl_context) as websocket:
                self.ws = websocket
                self.connected = True
                logger.info(f"✅ Connected to cTrader ({'DEMO' if self.demo_mode else 'LIVE'})")
                
                # Authenticate
                await self.authenticate()
                
                # Start trading
                await asyncio.gather(
                    self.message_handler(),
                    self.trading_loop(),
                    self.health_monitor()
                )
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    async def authenticate(self):
        """Authenticate with cTrader"""
        # App authentication
        app_auth = {
            "clientMsgId": str(self.message_id),
            "payloadType": "PROTO_OA_APPLICATION_AUTH_REQ",
            "payload": {
                "clientId": self.client_id,
                "clientSecret": ""
            }
        }
        
        await self.send_message(app_auth)
        await asyncio.sleep(2)
        
        # Account authentication
        account_auth = {
            "clientMsgId": str(self.message_id),
            "payloadType": "PROTO_OA_ACCOUNT_AUTH_REQ",
            "payload": {
                "ctidTraderAccountId": int(self.account_id),
                "accessToken": self.access_token
            }
        }
        
        await self.send_message(account_auth)
        logger.info("🔐 Authentication sent")
    
    async def send_message(self, message):
        """Send message to cTrader"""
        if self.ws and self.connected:
            await self.ws.send(json.dumps(message))
            self.message_id += 1
    
    async def message_handler(self):
        """Handle incoming messages"""
        async for message in self.ws:
            try:
                data = json.loads(message)
                msg_type = data.get('payloadType', '')
                
                if msg_type == 'PROTO_OA_ACCOUNT_AUTH_RES':
                    if not data.get('payload', {}).get('errorCode'):
                        self.authenticated = True
                        logger.info("🔥 AUTHENTICATED! Bot is LIVE!")
                        await self.notify("🔥 cTrader Bot is LIVE and authenticated!")
                
                elif msg_type == 'PROTO_OA_EXECUTION_EVENT':
                    await self.handle_execution(data)
                
            except Exception as e:
                logger.error(f"❌ Message error: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        while True:
            try:
                if not self.authenticated:
                    await asyncio.sleep(10)
                    continue
                
                if self.daily_trades >= self.max_daily_trades:
                    logger.info("📊 Daily trade limit reached")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    continue
                
                # Analyze markets
                for symbol in self.symbols:
                    signal = await self.analyze_symbol(symbol)
                    
                    if signal['action'] in ['BUY', 'SELL'] and signal['confidence'] > 0.75:
                        await self.place_order(symbol, signal)
                
                await asyncio.sleep(300)  # 5-minute cycle
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def analyze_symbol(self, symbol):
        """AI analysis"""
        try:
            # Get Yahoo Finance data
            symbol_map = {'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X'}
            yf_symbol = symbol_map.get(symbol, f"{symbol}=X")
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d", interval="5m")
            
            if len(data) < 20:
                return {'action': 'HOLD', 'confidence': 0}
            
            # Simple RSI strategy
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Generate signals
            if current_rsi < 30 and current_price > sma_20:
                return {'action': 'BUY', 'confidence': 0.8, 'price': current_price, 'rsi': current_rsi}
            elif current_rsi > 70 and current_price < sma_20:
                return {'action': 'SELL', 'confidence': 0.8, 'price': current_price, 'rsi': current_rsi}
            else:
                return {'action': 'HOLD', 'confidence': 0.3, 'price': current_price, 'rsi': current_rsi}
                
        except Exception as e:
            logger.error(f"❌ Analysis error for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    async def place_order(self, symbol, signal):
        """Place trading order"""
        try:
            volume = 1000  # 0.01 lots
            
            order = {
                "clientMsgId": str(self.message_id),
                "payloadType": "PROTO_OA_NEW_ORDER_REQ",
                "payload": {
                    "ctidTraderAccountId": int(self.account_id),
                    "symbolId": symbol,
                    "orderType": "MARKET",
                    "tradeSide": signal['action'],
                    "volume": volume,
                    "timeInForce": "IMMEDIATE_OR_CANCEL"
                }
            }
            
            await self.send_message(order)
            self.daily_trades += 1
            
            message = f"🚀 Order Placed: {signal['action']} {symbol} (Confidence: {signal['confidence']:.1%})"
            logger.info(message)
            await self.notify(message)
            
        except Exception as e:
            logger.error(f"❌ Order error: {e}")
    
    async def handle_execution(self, data):
        """Handle trade execution"""
        try:
            execution = data.get('payload', {})
            order = execution.get('order', {})
            
            if order:
                symbol = order.get('symbolId')
                side = order.get('tradeSide')
                status = order.get('orderStatus')
                
                message = f"✅ Trade Executed: {side} {symbol} - {status}"
                logger.info(message)
                await self.notify(message)
                
        except Exception as e:
            logger.error(f"❌ Execution handler error: {e}")
    
    async def notify(self, message):
        """Send notification (email if configured)"""
        if self.email:
            try:
                # Simple email notification (you'd need to configure SMTP)
                logger.info(f"📧 Notification: {message}")
            except:
                pass
        
        # Log to Railway console
        logger.info(f"🔔 {message}")
    
    async def health_monitor(self):
        """Monitor bot health"""
        while True:
            try:
                # Send heartbeat every 30 seconds
                if self.authenticated:
                    heartbeat = {
                        "clientMsgId": str(self.message_id),
                        "payloadType": "HEARTBEAT_EVENT",
                        "payload": {}
                    }
                    await self.send_message(heartbeat)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(60)

# Railway entry point
if __name__ == "__main__":
    bot = RailwaycTraderBot()
    asyncio.run(bot.start())

# requirements.txt file content:
"""
websockets==11.0.3
yfinance==0.2.18
pandas==2.0.3
numpy==1.24.3
requests==2.31.0
"""

# railway.json file content:
"""
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python main.py",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
"""

# Procfile for Railway:
"""
web: python main.py
"""
