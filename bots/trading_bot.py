import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import time
from datetime import datetime
from config import config_halit, config_erkan

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading configuration
BASE_ASSET = "NOT"
QUOTE_ASSET = "USDT"
TRADING_SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
MIN_TRADE_AMOUNT = 0.01

def get_bitcoin_data():
    """Get latest price data from Binance"""
    logger.info(f"Getting {BASE_ASSET} data...")
    try:
        klines = client.get_historical_klines(
            TRADING_SYMBOL, 
            Client.KLINE_INTERVAL_1HOUR,
            "1 day ago UTC"
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_asset_volume', 
            'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['close'] = pd.to_numeric(df['close'])
        logger.info(f"{BASE_ASSET} data retrieved successfully!")
        logger.debug(f"Data head:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Error getting {BASE_ASSET} data: {e}")
        raise

def calculate_signals(df):
    """Calculate trading signals using moving averages"""
    logger.info("Calculating signals...")
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    logger.info("Signals calculated successfully!")
    return df

def trading_decision(df):
    """Make trading decision based on MA crossover - always returns either BUY or SELL"""
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
        logger.info("BUY decision made!")
        return 'BUY'
    else:
        logger.info("SELL decision made!")
        return 'SELL'

def execute_trade(decision):
    """Execute trade based on the decision"""
    logger.info("Executing trade...")
    try:
        if decision == 'BUY':
            logger.info(f"Getting {QUOTE_ASSET} balance...")
            quote_balance = float(client.get_asset_balance(asset=QUOTE_ASSET)['free'])
            logger.info(f"{QUOTE_ASSET} balance: {quote_balance}")
            
            if quote_balance > 10:
                order = client.create_order(
                    symbol=TRADING_SYMBOL,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quoteOrderQty=quote_balance * 0.5
                )
                logger.info(f"Buy order executed: {order}")
            else:
                logger.warning(f"Insufficient {QUOTE_ASSET} balance for trade")
                
        elif decision == 'SELL':
            logger.info(f"Getting {BASE_ASSET} balance...")
            base_balance = float(client.get_asset_balance(asset=BASE_ASSET)['free'])
            logger.info(f"{BASE_ASSET} balance: {base_balance}")
            
            if base_balance > MIN_TRADE_AMOUNT:
                quantity = f"{(base_balance * 0.5):.8f}".rstrip('0').rstrip('.')
                logger.debug(f"Formatted quantity: {quantity}")
                order = client.create_order(
                    symbol=TRADING_SYMBOL,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Sell order executed: {order}")
            else:
                logger.warning(f"Insufficient {BASE_ASSET} balance for trade")
                
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during trade execution: {e}")

def main():
    logger.info(f"Starting {BASE_ASSET} Trading Bot...")
    while True:
        try:
            df = get_bitcoin_data()
            df = calculate_signals(df)
            decision = trading_decision(df)
            
            if decision != 'HOLD':
                logger.info(f"Signal: {decision} at {datetime.now()}")
                execute_trade(decision)
            
            logger.info("Waiting for next cycle...")
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.info("Waiting 60 seconds before retry...")
            time.sleep(60)

if __name__ == "__main__":
    try:
        client = Client(config_erkan["key"], config_erkan["secret"])
        logger.info("Binance client initialized successfully")
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")