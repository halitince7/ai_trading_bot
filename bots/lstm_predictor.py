import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam

class LSTMPredictor:
    def __init__(self, lookback=60, epochs=10, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        # Use only 'Close' prices
        data = df['Close'].values.reshape(-1, 1)
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train(self, df):
        """Train the model"""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        self.model = self.build_model((self.lookback, 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        # test accuracy
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss}")
        
        return history
    
    def lstm_signal(self, df):
        """Predict next price and generate signal"""
        self.train(df)
        # Get the lookback period of data EXCLUDING the current price
        last_data = df['Close'].values[-(self.lookback + 1):-1].reshape(-1, 1)  # son fiyat hariç
        scaled_data = self.scaler.transform(last_data)
        
        # Prepare data for prediction
        X = scaled_data.reshape(1, self.lookback, 1)
        
        # Make prediction for the NEXT period
        pred_scaled = self.model.predict(X)
        next_prediction = self.scaler.inverse_transform(pred_scaled)[0, 0]
        
        # Current price (son kapanış fiyatı)
        current_price = df['Close'].iloc[-1]
        
        # Generate signal
        threshold = 0.001  # 0.1% change threshold
        print(f"Next prediction: {next_prediction:.2f}, Current: {current_price:.2f}, Last data: {last_data[-1][0]:.2f}")
        
        if next_prediction > current_price * (1 + threshold):
            signal = 'up'
        elif next_prediction < current_price * (1 - threshold):
            signal = 'down'
        else:
            signal = 'none'
            
        return signal
    