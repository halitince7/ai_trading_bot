from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import time
from datetime import datetime
import config  # Create this file to store your API keys

# Initialize Binance client
client = Client(config.API_KEY, config.API_SECRET)

def get_bitcoin_data():
    """Get latest Bitcoin price data from Binance"""
    klines = client.get_historical_klines(
        "BTCUSDT", 
        Client.KLINE_INTERVAL_1HOUR,
        "1 day ago UTC"
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 
        'volume', 'close_time', 'quote_asset_volume', 
        'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['close'] = pd.to_numeric(df['close'])
    return df

def calculate_signals(df):
    """Calculate trading signals using moving averages"""
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    return df

def trading_decision(df):
    """Make trading decision based on MA crossover"""
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] <= df['MA50'].iloc[-2]:
        return 'BUY'
    elif df['MA20'].iloc[-1] < df['MA50'].iloc[-1] and df['MA20'].iloc[-2] >= df['MA50'].iloc[-2]:
        return 'SELL'
    return 'HOLD'

def execute_trade(decision):
    """Execute trade based on the decision"""
    try:
        if decision == 'BUY':
            # Get current BTC balance in USDT
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
            if usdt_balance > 10:  # Minimum trade amount
                order = client.create_order(
                    symbol='BTCUSDT',
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quoteOrderQty=usdt_balance * 0.95  # Use 95% of available USDT
                )
                print(f"Buy order executed: {order}")
                
        elif decision == 'SELL':
            # Get current BTC balance
            btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
            if btc_balance > 0.0001:  # Minimum trade amount
                order = client.create_order(
                    symbol='BTCUSDT',
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=btc_balance * 0.95  # Sell 95% of available BTC
                )
                print(f"Sell order executed: {order}")
                
    except BinanceAPIException as e:
        print(f"An error occurred: {e}")

def main():
    print("Starting Bitcoin Trading Bot...")
    while True:
        try:
            # Get current data
            df = get_bitcoin_data()
            
            # Calculate signals
            df = calculate_signals(df)
            
            # Make trading decision
            decision = trading_decision(df)
            
            # Execute trade if necessary
            if decision != 'HOLD':
                print(f"Signal: {decision} at {datetime.now()}")
                execute_trade(decision)
            
            # Wait for 1 hour before next check
            time.sleep(3600)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()
