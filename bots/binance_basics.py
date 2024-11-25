from binance.client import Client
from datetime import datetime
from typing import Dict
import logging
from config import config_halit, config_erkan  # import API keys from config.py

BASE_ASSET = "PEPE"     # The crypto you're trading (ETH)
QUOTE_ASSET = "USDT"   # The currency you're trading against (USDT)
TRADING_SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('account_info.log'),
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

def get_open_positions(client: Client):
    """Get all open positions with detailed information including entry time"""
    try:
        positions = client.futures_position_information()
        
        logger.info("\n=== Open Positions ===")
        has_positions = False
        
        for position in positions:
            amount = float(position['positionAmt'])
            if amount != 0:  # Only show positions with non-zero amount
                has_positions = True
                symbol = position['symbol']
                
                # Get recent trades to find entry time
                trades = client.futures_account_trades(symbol=symbol, limit=50)
                entry_time = None
                
                # Find the earliest trade that matches our position direction
                for trade in reversed(trades):
                    trade_amt = float(trade['qty'])
                    is_buyer = trade['buyer']
                    if (amount > 0 and is_buyer) or (amount < 0 and not is_buyer):
                        entry_time = datetime.fromtimestamp(trade['time']/1000)
                        break
                
                last_update_time = datetime.fromtimestamp(float(position['updateTime'])/1000)
                
                # Basic position information
                logger.info(f"\nSymbol: {position['symbol']}")
                logger.info(f"Position Amount: {position['positionAmt']}")
                logger.info(f"Entry Price: {position['entryPrice']}")
                logger.info(f"Mark Price: {position['markPrice']}")
                logger.info(f"Unrealized PNL: {position['unRealizedProfit']}")
                
                # Optional information - check if exists
                if 'liquidationPrice' in position:
                    logger.info(f"Liquidation Price: {position['liquidationPrice']}")
                
                # Get leverage from futures account
                try:
                    leverage_info = client.futures_leverage_bracket(symbol=symbol)
                    if leverage_info:
                        logger.info(f"Max Leverage: {leverage_info[0]['brackets'][0]['initialLeverage']}x")
                except:
                    pass
                
                logger.info(f"Entry Time: {entry_time if entry_time else 'Unknown'}")
                logger.info(f"Last Updated: {last_update_time}")
                
                # Optional position details
                if 'positionSide' in position:
                    logger.info(f"Position Side: {position['positionSide']}")
                
                # Calculate ROI
                entry_price = float(position['entryPrice'])
                mark_price = float(position['markPrice'])
                if entry_price > 0:
                    roi = ((mark_price - entry_price) / entry_price) * 100
                    roi = roi if amount > 0 else -roi  # Adjust ROI sign for short positions
                    logger.info(f"ROI: {roi:.2f}%")
        
        if not has_positions:
            logger.info("No open positions")
            
        return positions
    except Exception as e:
        logger.error(f"Failed to get open positions: {e}")
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
        get_open_positions(client)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()