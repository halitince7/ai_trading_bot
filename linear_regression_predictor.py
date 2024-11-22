import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from binance.client import Client
import bots.config as config
from sklearn.metrics import mean_squared_error, r_squared_score
import matplotlib.pyplot as plt

class LinearRegressionPredictor:
    def __init__(self, symbol='BTCUSDT', lookback_days=30, prediction_window=24):
        """
        Initialize the predictor
        :param symbol: Trading pair symbol
        :param lookback_days: Number of days of historical data to use
        :param prediction_window: Hours to look ahead for prediction
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.prediction_window = prediction_window
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()
        self.last_trend_slope = None
        self.confidence_score = None

    def _get_historical_data(self):
        """Fetch historical data from Binance"""
        try:
            # Get historical klines/candlestick data
            klines = self.client.get_historical_klines(
                self.symbol,
                Client.KLINE_INTERVAL_1HOUR,
                f"{self.lookback_days + 2} days ago UTC"
            )
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # Add technical indicators
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['volatility'] = df['close'].rolling(window=24).std()
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def _prepare_features(self, df):
        """Prepare features for the model"""
        # Create features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['price_momentum'] = df['close'].pct_change(12)  # 12-hour momentum
        df['volume_momentum'] = df['volume'].pct_change(12)
        df['price_volatility'] = df['close'].rolling(24).std()
        
        # Create target variable (future price change)
        df['target'] = df['close'].shift(-self.prediction_window)
        df['price_change'] = df['target'] / df['close'] - 1
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Select features for the model
        feature_columns = [
            'hour', 'day_of_week', 'price_momentum', 'volume_momentum',
            'price_volatility', 'SMA_20', 'SMA_50'
        ]
        
        return df[feature_columns], df['price_change']

    def train_model(self):
        """Train the linear regression model"""
        try:
            # Get and prepare data
            df = self._get_historical_data()
            if df is None:
                return False
            
            X, y = self._prepare_features(df)
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate confidence score
            self.confidence_score = self.model.score(X, y)
            
            # Store the trend slope
            self.last_trend_slope = self.model.coef_[2]  # price_momentum coefficient
            
            print(f"Model trained successfully. R² score: {self.confidence_score:.4f}")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def predict_next_movement(self):
        """Predict the next price movement"""
        try:
            df = self._get_historical_data()
            if df is None:
                return None, None, None
            
            X, _ = self._prepare_features(df)
            
            # Get the last row of features
            latest_features = X.iloc[-1:]
            
            # Make prediction
            predicted_change = self.model.predict(latest_features)[0]
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_change)
            
            return current_price, predicted_price, predicted_change
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None

    def generate_signal(self, threshold_percent=0.5):
        """Generate trading signal based on prediction"""
        try:
            current_price, predicted_price, predicted_change = self.predict_next_movement()
            
            if current_price is None:
                return 'HOLD'
            
            # Convert predicted change to percentage
            predicted_change_percent = predicted_change * 100
            
            # Generate signal based on prediction and confidence
            if self.confidence_score < 0.3:  # Low confidence threshold
                return 'HOLD'
            
            if predicted_change_percent > threshold_percent:
                return 'BUY'
            elif predicted_change_percent < -threshold_percent:
                return 'SELL'
            return 'HOLD'
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return 'HOLD'

    def get_prediction_metrics(self):
        """Get detailed prediction metrics"""
        current_price, predicted_price, predicted_change = self.predict_next_movement()
        
        if current_price is None:
            return None
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_percent': predicted_change * 100,
            'confidence_score': self.confidence_score,
            'trend_slope': self.last_trend_slope,
            'prediction_window': self.prediction_window
        }

    def plot_predictions(self):
        """Plot recent prices and prediction"""
        try:
            df = self._get_historical_data()
            if df is None:
                return
            
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[-48:], df['close'][-48:], label='Actual Price')
            
            # Add predicted price point
            _, predicted_price, _ = self.predict_next_movement()
            last_date = df.index[-1]
            future_date = last_date + timedelta(hours=self.prediction_window)
            plt.scatter(future_date, predicted_price, color='red', label='Predicted Price')
            
            plt.title(f'{self.symbol} Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting predictions: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = LinearRegressionPredictor(
        symbol='BTCUSDT',
        lookback_days=30,
        prediction_window=24
    )
    
    # Train the model
    print("Training model...")
    if predictor.train_model():
        # Get prediction metrics
        metrics = predictor.get_prediction_metrics()
        if metrics:
            print("\nPrediction Metrics:")
            print(f"Current Price: ${metrics['current_price']:.2f}")
            print(f"Predicted Price: ${metrics['predicted_price']:.2f}")
            print(f"Predicted Change: {metrics['predicted_change_percent']:.2f}%")
            print(f"Confidence Score: {metrics['confidence_score']:.4f}")
            
            # Get trading signal
            signal = predictor.generate_signal(threshold_percent=0.5)
            print(f"\nTrading Signal: {signal}")
            
            # Plot predictions
            predictor.plot_predictions()