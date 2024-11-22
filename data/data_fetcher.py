from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import bots.config as config
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, symbol='BTCUSDT', timeframe='1h'):
        """
        Initialize DataFetcher
        :param symbol: Trading pair symbol (default: 'BTCUSDT')
        :param timeframe: Candle timeframe (default: '1h')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.timeframe_minutes = self._get_timeframe_minutes()
        
    def _get_timeframe_minutes(self):
        """Convert timeframe string to minutes"""
        time_units = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080
        }
        unit = self.timeframe[-1]
        number = int(self.timeframe[:-1])
        return number * time_units[unit]

    def _get_binance_interval(self):
        """Convert timeframe to Binance interval format"""
        return self.timeframe.upper()

    def get_historical_data(self, lookback_days, include_current_candle=True):
        """
        Fetch historical data from Binance
        :param lookback_days: Number of days of historical data to fetch
        :param include_current_candle: Whether to include the current forming candle
        :return: DataFrame with OHLCV data and technical indicators
        """
        try:
            # Calculate start time
            start_time = datetime.now() - timedelta(days=lookback_days)
            
            # Fetch klines data
            klines = self.client.get_historical_klines(
                self.symbol,
                self._get_binance_interval(),
                start_str=start_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Set index
            df.set_index('timestamp', inplace=True)
            
            # Remove the current candle if specified
            if not include_current_candle:
                df = df[:-1]
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame
        :param df: DataFrame with OHLCV data
        :return: DataFrame with added technical indicators
        """
        try:
            # Moving averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Volatility
            df['ATR'] = self._calculate_atr(df)
            df['volatility'] = df['close'].rolling(window=24).std()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            
            # Price momentum
            df['ROC'] = self._calculate_roc(df['close'], 12)
            df['RSI'] = self._calculate_rsi(df['close'], 14)
            
            # Trend indicators
            df['trend'] = np.where(df['EMA_20'] > df['SMA_50'], 1, -1)
            
            logger.info(f"Added technical indicators to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def _calculate_roc(self, series, period):
        """Calculate Rate of Change"""
        return (series - series.shift(period)) / series.shift(period) * 100

    def _calculate_rsi(self, series, period):
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_latest_price(self):
        """Get the latest price for the symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None