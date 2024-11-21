# AI Bitcoin Trading Bot

A sophisticated automated trading bot for Bitcoin on Binance, implementing multiple technical analysis strategies including Moving Average Crossover and Linear Regression analysis.

## Features

- **Multiple Trading Strategies**
  - Moving Average Crossover (20/50 periods)
  - Linear Regression with Dynamic Bands
  - Combined signal confirmation for reduced false positives

- **Real-time Trading**
  - Automatic order execution on Binance
  - Market order support
  - Balance management
  - Error handling and continuous operation

- **Technical Analysis**
  - Dynamic trend identification
  - Support/Resistance levels through regression bands
  - Oversold/Overbought detection
  - Automated decision making

## Prerequisites

- Python 3.8+
- Binance account with API access
- Basic understanding of cryptocurrency trading



## Trading Strategy Details

### Moving Average Crossover
- Uses 20 and 50 period moving averages
- Generates signals on MA crossovers
- Helps identify trend direction and potential reversal points

### Linear Regression
- Calculates regression line over 20 periods
- Creates dynamic support/resistance bands
- Identifies overbought/oversold conditions
- Uses 2 standard deviation bands for trade signals

### Combined Strategy
- Requires confirmation from both strategies
- Reduces false signals
- More conservative approach for higher probability trades

## Risk Management

The bot implements several risk management features:
- Uses only 95% of available balance for trades
- Implements minimum trade amount checks
- Continuous error monitoring and handling
- Hourly market analysis and decision making

## Disclaimer

This trading bot is for educational and research purposes only. Cryptocurrency trading carries significant risks:

- The bot may not be profitable in all market conditions
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- Thoroughly test the bot with small amounts before deploying with significant capital

## Future Improvements

- [ ] Implement backtesting functionality
- [ ] Add more technical indicators
- [ ] Include stop-loss and take-profit features
- [ ] Add support for multiple trading pairs
- [ ] Implement position sizing based on volatility
- [ ] Add web interface for monitoring
- [ ] Include email/telegram notifications

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Binance API Documentation
- Python-Binance library
- SciPy and NumPy communities

## Support

For support, please open an issue in the GitHub repository or contact [your-email@example.com].


