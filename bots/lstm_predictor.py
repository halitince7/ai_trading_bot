import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam

class LSTMPredictor:
    def __init__(self, lookback=60, epochs=50, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        # Use Close price and trading volume for better predictions
        data = df[['Close']].values
        
        # Calculate returns instead of using raw prices
        returns = np.log(data[1:] / data[:-1])
        data_scaled = self.scaler.fit_transform(returns)
        
        X, y = [], []
        for i in range(len(data_scaled) - self.lookback):
            X.append(data_scaled[i:(i + self.lookback)])
            y.append(data_scaled[i + self.lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        # Use 80% of data for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.1),
            LSTM(units=64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.1),
            Dense(units=32, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
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
        #test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        #print(f"Test loss: {test_loss}")
        
        return history
    
    def lstm_signal(self, df):
        """Predict next price and generate signal"""
        self.train(df)
        
        # Get the lookback period + 1 of data to calculate returns
        last_data = df['Close'].values[-(self.lookback + 2):].reshape(-1, 1)  # Get one extra point
        returns = np.log(last_data[1:] / last_data[:-1])  # Calculate returns
        scaled_data = self.scaler.transform(returns)[-self.lookback:]  # Take exactly lookback points
        
        # Prepare data for prediction
        X = scaled_data.reshape(1, self.lookback, 1)
        
        # Make prediction for the NEXT period
        predicted_scaled = self.model.predict(X, verbose=0)
        predicted_return = self.scaler.inverse_transform(predicted_scaled)[0][0]
        
        # Convert predicted return to actual price
        current_price = df['Close'].iloc[-1]
        next_prediction = current_price * np.exp(predicted_return)
        
        # Generate signal
        threshold = 0.001  # 0.1% change threshold
        print(f"Next prediction: {next_prediction:.2f}, Current: {current_price:.2f}")
        
        if next_prediction > current_price * (1 + threshold):
            signal = 'up'
        elif next_prediction < current_price * (1 - threshold):
            signal = 'down'
        else:
            signal = 'none'
            
        return signal