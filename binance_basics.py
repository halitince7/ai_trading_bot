from binance.client import Client
from datetime import datetime
import config
import pandas as pd

class BinanceActions:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)

    def get_account_balance(self, asset=None):
        """
        Get balance for all assets or a specific asset
        Example: get_account_balance('BTC') or get_account_balance()
        """
        try:
            if asset:
                balance = self.client.get_asset_balance(asset=asset)
                print(f"\n{asset} Balance:")
                print(f"Free: {balance['free']}")
                print(f"Locked: {balance['locked']}")
                return balance
            else:
                balances = self.client.get_account()['balances']
                # Filter out zero balances
                non_zero = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
                print("\nAll Non-Zero Balances:")
                for balance in non_zero:
                    print(f"{balance['asset']}:")
                    print(f"  Free: {balance['free']}")
                    print(f"  Locked: {balance['locked']}")
                return non_zero
        except Exception as e:
            print(f"Error getting balance: {e}")

    def get_current_price(self, symbol='BTCUSDT'):
        """Get current price of a trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            print(f"\nCurrent {symbol} price: {ticker['price']}")
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting price: {e}")

    def get_recent_trades(self, symbol='BTCUSDT', limit=10):
        """Get your recent trades"""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            print(f"\nRecent {symbol} trades:")
            for trade in trades:
                side = "BUY" if trade['isBuyer'] else "SELL"
                print(f"Time: {datetime.fromtimestamp(trade['time']/1000)}")
                print(f"Side: {side}")
                print(f"Price: {trade['price']}")
                print(f"Quantity: {trade['qty']}")
                print("------------------------")
            return trades
        except Exception as e:
            print(f"Error getting trades: {e}")

    def get_open_orders(self, symbol='BTCUSDT'):
        """Get all open orders"""
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            print(f"\nOpen orders for {symbol}:")
            for order in orders:
                print(f"Order ID: {order['orderId']}")
                print(f"Type: {order['type']}")
                print(f"Side: {order['side']}")
                print(f"Price: {order['price']}")
                print(f"Original Quantity: {order['origQty']}")
                print("------------------------")
            return orders
        except Exception as e:
            print(f"Error getting open orders: {e}")

    def get_deposit_history(self, days=30):
        """Get deposit history"""
        try:
            deposits = self.client.get_deposit_history()
            print("\nDeposit History:")
            for deposit in deposits:
                print(f"Asset: {deposit['coin']}")
                print(f"Amount: {deposit['amount']}")
                print(f"Status: {deposit['status']}")
                print(f"Time: {datetime.fromtimestamp(deposit['insertTime']/1000)}")
                print("------------------------")
            return deposits
        except Exception as e:
            print(f"Error getting deposit history: {e}")

    def get_market_depth(self, symbol='BTCUSDT', limit=5):
        """Get order book / market depth"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            print(f"\nMarket Depth for {symbol}:")
            print("\nBids (Buy Orders):")
            for bid in depth['bids'][:limit]:
                print(f"Price: {bid[0]}, Quantity: {bid[1]}")
            print("\nAsks (Sell Orders):")
            for ask in depth['asks'][:limit]:
                print(f"Price: {ask[0]}, Quantity: {ask[1]}")
            return depth
        except Exception as e:
            print(f"Error getting market depth: {e}")

    def get_24h_stats(self, symbol='BTCUSDT'):
        """Get 24-hour statistics"""
        try:
            stats = self.client.get_ticker(symbol=symbol)
            print(f"\n24h Statistics for {symbol}:")
            print(f"High: {stats['highPrice']}")
            print(f"Low: {stats['lowPrice']}")
            print(f"Volume: {stats['volume']}")
            print(f"Price Change: {stats['priceChange']}")
            print(f"Price Change Percent: {stats['priceChangePercent']}%")
            return stats
        except Exception as e:
            print(f"Error getting 24h stats: {e}")

# Usage example
if __name__ == "__main__":
    binance = BinanceActions()
    
    # Example usage of different functions
    print("\n=== Basic Account Information ===")
    binance.get_account_balance('BTC')  # Get BTC balance
    binance.get_account_balance('USDT')  # Get USDT balance
    
    print("\n=== Market Information ===")
    binance.get_current_price('BTCUSDT')  # Get current BTC price
    binance.get_market_depth('BTCUSDT', 3)  # Get order book
    binance.get_24h_stats('BTCUSDT')  # Get 24h statistics
    
    print("\n=== Trading Information ===")
    binance.get_recent_trades('BTCUSDT', 3)  # Get recent trades
    binance.get_open_orders('BTCUSDT')  # Get open orders