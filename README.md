# AI Cryptocurrency Trading Bot with LSTM Prediction

An automated trading bot for cryptocurrency futures trading on Binance, leveraging LSTM (Long Short-Term Memory) neural networks for price prediction and automated trading execution.

## Overview

This bot combines deep learning with automated trading to create a sophisticated cryptocurrency trading system. It uses LSTM neural networks to predict price movements and automatically executes trades on Binance Futures based on these predictions.

## Key Features

- **LSTM Price Prediction**
  - Uses historical price data to predict future movements
  - Implements a sliding window approach for time series analysis
  - Normalizes data for better prediction accuracy
  - Provides directional signals (up/down) based on prediction confidence

- **Automated Futures Trading**
  - Real-time order execution on Binance Futures
  - Supports both long and short positions
  - Automatic position management with stop-loss and take-profit orders
  - Handles multiple trading pairs simultaneously
  - Implements isolated margin trading with configurable leverage

- **Risk Management**
  - Configurable position sizes and leverage
  - Automatic stop-loss and take-profit placement
  - Continuous monitoring of open positions
  - Error handling and graceful failure recovery
  - Minimum notional value checks for order placement

## How It Works

### 1. LSTM Prediction System
The `LSTMPredictor` class implements the price prediction logic:
- Takes historical price data with a configurable lookback period
- Normalizes the data using MinMaxScaler
- Trains an LSTM model with the following architecture:
  - LSTM layer (50 units) with return sequences
  - Dropout layer (20%)
  - LSTM layer (50 units)
  - Dropout layer (20%)
  - Dense layers for final prediction
- Generates trading signals based on predicted price movements

### 2. Trading Execution
The `BinanceFuturesTrader` class handles all trading operations:
- Monitors multiple trading pairs simultaneously
- Places market orders based on LSTM predictions
- Sets leverage and margin type for each trade
- Implements automatic stop-loss and take-profit orders
- Manages open positions and orders

### 3. Main Trading Loop
The main loop orchestrates the entire trading process:
1. Monitors current positions and their PnL
2. Processes existing positions:
   - Gets new predictions for open positions
   - Closes positions if prediction changes
3. Looks for new trading opportunities:
   - Analyzes all available trading pairs
   - Places new orders based on LSTM predictions
4. Implements error handling and retry mechanisms

## Configuration

The bot can be configured through the `TradingConfig` class:
- `volume`: Trading volume in USDT
- `tp_percentage`: Take profit percentage
- `sl_percentage`: Stop loss percentage
- `leverage`: Trading leverage (default: 1x)
- `margin_type`: Margin type (ISOLATED/CROSSED)

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Binance Futures account with API access
- Required Python packages:
  - numpy
  - pandas
  - keras
  - python-binance
  - scikit-learn

## Installation

### Prerequisites
- Python 3.8-3.11 (TensorFlow is not compatible with Python 3.13)
- pip (Python package installer)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/halitince7/ai_trading_bot.git
cd ai_trading_bot
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the trading bot:
```bash
cd bots
python future_trading.py
```

## Risk Warning

This trading bot is for educational and experimental purposes. Please note:
- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- Test thoroughly with small amounts before deploying significant capital

## Future Improvements

- [ ] Add more sophisticated LSTM architectures
- [ ] Implement portfolio management
- [ ] Add backtesting capabilities
- [ ] Include more technical indicators
- [ ] Add web interface for monitoring
- [ ] Implement dynamic position sizing
- [ ] Add more risk management features

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.