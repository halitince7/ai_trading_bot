from binance.client import Client
from datetime import datetime
from typing import Dict
import logging
from config import config_halit, config_erkan  # import API keys from config.py

BASE_ASSET = "NOT"     # The crypto you're trading (ETH)
QUOTE_ASSET = "USDT"   # The currency you're trading against (USDT)
TRADING_SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/account_info.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_client() -> Client:
    """Initialize Binance client"""
    try:
        client = Client(config_erkan["key"], config_erkan["secret"])
        client.get_account_status()
        logger.info("Binance connection successful!")
        return client
    except Exception as e:
        logger.error(f"Binance connection failed: {e}")
        raise e

def get_account_balance(client: Client) -> Dict:
    """Get account balance for all assets"""
    try:
        account = client.get_account()
        balances = account['balances']
        
        logger.info("=== Account Balances ===")
        for balance in balances:
            free = float(balance['free'])
            locked = float(balance['locked'])
            
            if (free > 0 or locked > 0) and (balance['asset'] == BASE_ASSET or balance['asset'] == QUOTE_ASSET):
                logger.info(f"\n{balance['asset']}:")
                logger.info(f"Available: {free}")
                logger.info(f"Locked: {locked}")
                logger.info(f"Total: {free + locked}")
        
        return balances
    except Exception as e:
        logger.error(f"Failed to get balance info: {e}")
        return {}

def get_current_prices(client: Client, symbols=[TRADING_SYMBOL]):
    """Get current prices for specified symbols"""
    try:
        prices = client.get_all_tickers()
        
        logger.info("=== Current Prices ===")
        for price in prices:
            if price['symbol'] in symbols:
                logger.info(f"{price['symbol']}: {price['price']}")
                
        return prices
    except Exception as e:
        logger.error(f"Failed to get price info: {e}")
        return {}

def get_account_status(client: Client):
    """Get account status"""
    try:
        status = client.get_account_status()
        logger.info("\n=== Account Status ===")
        logger.info(f"Status: {status['data']}")
        
        return status
    except Exception as e:
        logger.error(f"Failed to get account status: {e}")
        return {}

def get_trade_history(client: Client, symbol=TRADING_SYMBOL, limit=5):
    """Get recent trade history"""
    try:
        trades = client.get_my_trades(symbol=symbol, limit=limit)
        
        logger.info(f"\n=== Recent {symbol} Trades ===")
        for trade in trades:
            logger.info(f"\nTrade ID: {trade['id']}")
            logger.info(f"Price: {trade['price']}")
            logger.info(f"Quantity: {trade['qty']}")
            logger.info(f"Time: {datetime.fromtimestamp(trade['time']/1000)}")
            logger.info(f"Side: {trade['isBuyer'] and 'BUY' or 'SELL'}")
            
        return trades
    except Exception as e:
        logger.error(f"Failed to get trade history: {e}")
        return []

def get_open_orders(client: Client):
    """Get all open orders"""
    try:
        orders = client.get_open_orders()
        
        logger.info("\n=== Open Orders ===")
        if not orders:
            logger.info("No open orders")
            
        for order in orders:
            logger.info(f"\nOrder ID: {order['orderId']}")
            logger.info(f"Symbol: {order['symbol']}")
            logger.info(f"Type: {order['type']}")
            logger.info(f"Side: {order['side']}")
            logger.info(f"Price: {order['price']}")
            logger.info(f"Amount: {order['origQty']}")
            
        return orders
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []

def main():
    try:
        logger.info("Initializing Binance client...")
        client = setup_client()
        
        # Get account information
        get_account_balance(client)
        get_current_prices(client)
        get_account_status(client)
        get_trade_history(client)
        get_open_orders(client)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()