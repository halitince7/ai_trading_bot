from typing import List, Dict, Optional, Any
from decimal import Decimal
import time
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
from binance.um_futures import UMFutures
from binance.error import ClientError
import ta

from config import config_erkan
from lstm_predictor import LSTMPredictor

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('future_trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    volume: float = 10.0
    tp_percentage: float = 0.009  # Take profit percentage
    sl_percentage: float = 0.009  # Stop loss percentage
    retry_attempts: int = 3
    retry_delay: float = 1.0
    leverage: int = 1  # Default leverage
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED

class BinanceFuturesTrader:
    """Handles all futures trading operations with Binance"""
    
    def __init__(self, api_key: str, api_secret: str, config: TradingConfig = TradingConfig()):
        """Initialize the trader with API credentials and configuration"""
        self.client = UMFutures(key=api_key, secret=api_secret)
        self.config = config
        
    def get_balance_usdt(self) -> Optional[float]:
        """Get USDT balance from futures wallet"""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.balance(recvWindow=6000)
                return next(
                    (float(bal['balance']) for bal in response if bal['asset'] == 'USDT'),
                    None
                )
            except ClientError as error:
                logger.error(f"Attempt {attempt + 1} failed: {error}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(self.config.retry_delay)

    def get_tickers_usdt(self) -> List[str]:
        """Get all available USDT trading pairs"""
        try:
            resp = self.client.ticker_price()
            return [elem['symbol'] for elem in resp if 'USDT' in elem['symbol']]
        except ClientError as error:
            logger.error(f"Failed to get tickers: {error}")
            raise

    def get_historical_klines(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Get historical kline data for a symbol"""
        try:
            resp = pd.DataFrame(self.client.klines(symbol, timeframe))
            if resp.empty:
                return pd.DataFrame()
            
            resp = resp.iloc[:, :6]
            resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            resp = resp.set_index('Time')
            resp.index = pd.to_datetime(resp.index, unit='ms')
            return resp.astype(float)
        except ClientError as error:
            logger.error(f"Failed to get klines for {symbol}: {error}")
            raise

    def get_symbol_info(self, symbol: str) -> Dict[str, int]:
        """Get price and quantity precision for a symbol"""
        try:
            symbol_info = next(
                (item for item in self.client.exchange_info()['symbols'] 
                 if item['symbol'] == symbol),
                None
            )
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found")
                
            return {
                'price_precision': symbol_info['pricePrecision'],
                'qty_precision': symbol_info['quantityPrecision']
            }
        except ClientError as error:
            logger.error(f"Failed to get symbol info for {symbol}: {error}")
            raise

    def _set_leverage_and_margin(self, symbol: str) -> None:
        """Set leverage and margin type for a symbol"""
        try:
            # Set margin type
            try:
                self.client.change_margin_type(
                    symbol=symbol,
                    marginType=self.config.margin_type,
                    recvWindow=6000
                )
            except ClientError as e:
                # Ignore error if margin type is already set
                if e.error_code != -4046:  # -4046 is "No need to change margin type"
                    raise

            # Set leverage
            self.client.change_leverage(
                symbol=symbol,
                leverage=self.config.leverage,
                recvWindow=6000
            )
            logger.info(f"Set leverage to {self.config.leverage}x and margin type to {self.config.margin_type} for {symbol}")
        except ClientError as error:
            logger.error(f"Failed to set leverage/margin for {symbol}: {error}")
            raise

    def place_order(self, symbol: str, side: str) -> None:
        """Place a new order with stop loss and take profit"""
        try:
            # Log position PnL before placing new order
            current_pnl = self.get_position_pnl(symbol)
            if current_pnl:
                logger.info(f"Current position PnL for {symbol}: {current_pnl['unrealized_pnl']:.2f} USDT ({current_pnl['pnl_percentage']:.2f}%)")
            
            # Set leverage and margin type before placing order
            self._set_leverage_and_margin(symbol)
            
            current_price = float(self.client.ticker_price(symbol)['price'])
            symbol_info = self.get_symbol_info(symbol)
            
            qty = round(
                self.config.volume / current_price,
                symbol_info['qty_precision']
            )
            
            # Place main order
            main_order = self._place_market_order(symbol, side, qty)
            logger.info(f"{symbol} {side} order placed")
            
            # Place stop loss and take profit orders
            self._place_sl_tp_orders(
                symbol, side, qty, current_price,
                symbol_info['price_precision']
            )
        except ClientError as error:
            logger.error(f"Failed to place orders for {symbol}")
            raise

    def _place_market_order(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        """Place a market order"""
        return self.client.new_order(
            symbol=symbol,
            side=side.upper(),
            type='MARKET',
            quantity=qty
        )

    def _place_sl_tp_orders(self, symbol: str, side: str, qty: float,
                           current_price: float, price_precision: int) -> None:
        """Place stop loss and take profit orders"""
        is_buy = side.lower() == 'buy'
        sl_price = round(
            current_price * (1 - self.config.sl_percentage if is_buy else 1 + self.config.sl_percentage),
            price_precision
        )
        tp_price = round(
            current_price * (1 + self.config.tp_percentage if is_buy else 1 - self.config.tp_percentage),
            price_precision
        )

        # Place stop loss
        self.client.new_order(
            symbol=symbol,
            side='SELL' if is_buy else 'BUY',
            type='STOP_MARKET',
            quantity=qty,
            timeInForce='GTC',
            stopPrice=sl_price
        )
        
        # Place take profit
        self.client.new_order(
            symbol=symbol,
            side='SELL' if is_buy else 'BUY',
            type='TAKE_PROFIT_MARKET',
            quantity=qty,
            timeInForce='GTC',
            stopPrice=tp_price
        )

    def get_positions(self) -> List[str]:
        """Get current open positions"""
        try:
            positions = self.client.get_position_risk()
            return [
                pos['symbol'] for pos in positions
                if float(pos['positionAmt']) != 0
            ]
        except ClientError as error:
            logger.error(f"Failed to get positions: {error}")
            raise

    def get_open_orders(self) -> List[str]:
        """Get symbols with open orders"""
        try:
            orders = self.client.get_orders(recvWindow=6000)
            return list({order['symbol'] for order in orders})
        except ClientError as error:
            logger.error(f"Failed to get open orders: {error}")
            raise

    def close_open_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol"""
        try:
            response = self.client.cancel_open_orders(symbol=symbol, recvWindow=6000)
            logger.info(f"Cancelled open orders for {symbol}: {response}")
        except ClientError as error:
            logger.error(f"Failed to close orders for {symbol}: {error}")
            raise

    def get_position_pnl(self, symbol: str) -> Dict[str, float]:
        """Get PnL information for a specific position"""
        try:
            positions = self.client.get_position_risk(symbol=symbol, recvWindow=6000)
            for position in positions:
                if position['symbol'] == symbol:
                    entry_price = float(position['entryPrice'])
                    mark_price = float(position['markPrice'])
                    position_amt = float(position['positionAmt'])
                    unrealized_pnl = float(position['unRealizedProfit'])
                    
                    return {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'current_price': mark_price,
                        'position_size': position_amt,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percentage': (unrealized_pnl / (abs(position_amt) * entry_price)) * 100 if position_amt != 0 else 0
                    }
            return None
        except ClientError as error:
            logger.error(f"Failed to get PnL for {symbol}: {error}")
            raise

    def get_all_positions_pnl(self) -> List[Dict[str, float]]:
        """Get PnL information for all open positions"""
        try:
            positions = self.client.get_position_risk(recvWindow=6000)
            pnl_info = []
            total_unrealized_pnl = 0
            
            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:  # Only include active positions
                    entry_price = float(position['entryPrice'])
                    mark_price = float(position['markPrice'])
                    unrealized_pnl = float(position['unRealizedProfit'])
                    total_unrealized_pnl += unrealized_pnl
                    
                    pnl_info.append({
                        'symbol': position['symbol'],
                        'entry_price': entry_price,
                        'current_price': mark_price,
                        'position_size': position_amt,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percentage': (unrealized_pnl / (abs(position_amt) * entry_price)) * 100
                    })
            
            logger.info(f"Total unrealized PnL: {total_unrealized_pnl:.2f} USDT")
            return pnl_info
        except ClientError as error:
            logger.error(f"Failed to get all positions PnL: {error}")
            raise

    def get_income_history(self, income_type: str = "REALIZED_PNL", limit: int = 100) -> List[Dict]:
        """Get trading income history
        
        Args:
            income_type: Type of income to query (REALIZED_PNL, COMMISSION, FUNDING_FEE)
            limit: Number of records to return
        """
        try:
            income = self.client.get_income_history(
                incomeType=income_type,
                limit=limit,
                recvWindow=6000
            )
            
            total_pnl = sum(float(record['income']) for record in income)
            logger.info(f"Total {income_type}: {total_pnl:.2f} USDT (last {limit} trades)")
            
            return income
        except ClientError as error:
            logger.error(f"Failed to get income history: {error}")
            raise

def main():
    """Main trading loop"""
    trader = BinanceFuturesTrader(
        api_key=config_erkan['key'],
        api_secret=config_erkan['secret']
    )
    lstm_predictor = LSTMPredictor()
    
    logger.info('Trading bot started')
    
    while True:
        try:
            # Log current PnL for all positions
            pnl_info = trader.get_all_positions_pnl()
            for position in pnl_info:
                logger.info(
                    f"{position['symbol']}: PnL {position['unrealized_pnl']:.2f} USDT "
                    f"({position['pnl_percentage']:.2f}%)"
                )
            
            # Log realized PnL
            trader.get_income_history(income_type="REALIZED_PNL", limit=100)
            
            current_positions = trader.get_positions()
            current_orders = trader.get_open_orders()
            
            # Process existing positions
            for symbol in current_positions:
                try:
                    if symbol not in current_orders:
                        klines = trader.get_historical_klines(symbol)
                        prediction = lstm_predictor.lstm_signal(klines)
                        if prediction != "none":
                            trader.close_open_orders(symbol)
                except Exception as e:
                    logger.error(f"Error processing position for {symbol}: {e}")
                    continue
            
            # Look for new trading opportunities
            tickers = trader.get_tickers_usdt()
            for symbol in tickers:
                try:
                    if symbol not in current_positions and symbol not in current_orders:
                        klines = trader.get_historical_klines(symbol)
                        prediction = lstm_predictor.lstm_signal(klines)
                        
                        if prediction == 'up':
                            trader.place_order(symbol, 'buy')
                        elif prediction == 'down':
                            trader.place_order(symbol, 'sell')
                except Exception as e:
                    logger.error(f"Error placing order for {symbol}: {e}")
                    continue
            
            time.sleep(60)  # Wait for 1 minute before next iteration
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
