import logging
from binance.client import Client
from binance.enums import *
import time
from datetime import datetime
import pandas as pd
from bots.config import config_erkan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Trading configuration
API_KEY = config_erkan["key"]
API_SECRET = config_erkan["secret"]
# Trading configuration
BASE_ASSET = "NOT"
QUOTE_ASSET = "USDT"
TRADING_SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
LEVERAGE = 5  # Trading leverage (e.g., 5x)
POSITION_SIZE = 0.1  # 10% of available balance
INTERVAL = '1h'  # Timeframe for analysis

class FuturesBot:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        self.setup_futures_account()

    def setup_futures_account(self):
        """Initialize futures trading settings"""
        try:
            # Hedge mode kontrolü
            try:
                self.client.futures_change_position_mode(dualSidePosition=False)
            except Exception as e:
                logger.info("Position mode already set or not needed to be changed")
            
            # Set leverage
            try:
                self.client.futures_change_leverage(symbol=TRADING_SYMBOL, leverage=LEVERAGE)
                logger.info(f"Leverage set to {LEVERAGE}x")
            except Exception as e:
                logger.info(f"Leverage setting error: {e}")
            
            # Set margin type to ISOLATED
            try:
                self.client.futures_change_margin_type(symbol=TRADING_SYMBOL, marginType='ISOLATED')
                logger.info("Margin type set to ISOLATED")
            except Exception as e:
                logger.info("Margin type already set or not needed to be changed")
            
            logger.info("Futures account setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up futures account: {e}")
            raise

    def get_account_balance(self):
        """Get USDT futures balance"""
        try:
            futures_balance = self.client.futures_account_balance()
            usdt_balance = next(balance for balance in futures_balance if balance['asset'] == 'USDT')
            return float(usdt_balance['balance'])
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0

    def calculate_position_size(self):
        """Calculate the position size based on current balance"""
        try:
            # Get symbol info
            symbol_info = self.client.futures_exchange_info()
            symbol_details = next(filter(lambda x: x['symbol'] == TRADING_SYMBOL, symbol_info['symbols']))
            quantity_precision = symbol_details['quantityPrecision']
            
            # Calculate position size
            balance = self.get_account_balance()
            position_value = balance * POSITION_SIZE
            current_price = float(self.client.futures_symbol_ticker(symbol=TRADING_SYMBOL)['price'])
            quantity = (position_value * LEVERAGE) / current_price
            
            # Round to the correct precision
            quantity = round(quantity, quantity_precision)
            
            logger.info(f"Calculated position size: {quantity}")
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def get_market_data(self):
        """Get historical market data and calculate indicators"""
        try:
            # Get klines (candlestick data)
            klines = self.client.futures_klines(symbol=TRADING_SYMBOL, interval=INTERVAL, limit=100)
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                             'taker_buy_quote', 'ignored'])
            
            # Convert price columns to float
            df['close'] = df['close'].astype(float)
            
            # Calculate 20 EMA
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def get_trading_signal(self, df):
        """Generate trading signal based on EMA strategy - only BUY or SELL"""
        if df is None or df.empty:
            return 'SELL'  # Default to SELL if no data
        
        current_price = float(df['close'].iloc[-1])
        ema20 = float(df['ema20'].iloc[-1])
        
        if current_price > ema20:
            return 'BUY'
        else:
            return 'SELL'

    def execute_trade(self, side):
        """Execute futures trade"""
        try:
            quantity = self.calculate_position_size()
            
            if quantity > 0:
                order = self.client.futures_create_order(
                    symbol=TRADING_SYMBOL,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Order executed: {order}")
                return True
            else:
                logger.warning("Insufficient balance for trade")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def run(self):
        """Main bot loop"""
        logger.info("Starting Futures Trading Bot...")
        
        while True:
            try:
                # Get market data and generate signal
                df = self.get_market_data()
                signal = self.get_trading_signal(df)
                
                logger.info(f"Current signal: {signal}")
                
                # Execute trades based on signal
                if signal == 'BUY':
                    self.execute_trade(SIDE_BUY)
                elif signal == 'SELL':
                    self.execute_trade(SIDE_SELL)
                
                # Wait before next iteration
                logger.info("Waiting for next cycle...")
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retry

if __name__ == "__main__":
    try:
        bot = FuturesBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
