import logging
import sys
import traceback
from typing import Any, Dict, Optional
from pandas import DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
import talib.abstract as ta
from freqtrade.strategy import IntParameter, DecimalParameter, IStrategy
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Verbesserte Logging-Konfiguration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def safe_run(func):
    """
    Decorator für sicheres Ausführen von Funktionen
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Critical error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if func.__name__ == 'populate_indicators':
                return args[1]
            elif func.__name__ in ['populate_entry_trend', 'populate_exit_trend']:
                df = args[1]
                df['enter_long'] = 0
                df['enter_short'] = 0
                df['exit_long'] = 0
                df['exit_short'] = 0
                return df
            return None
    return wrapper

class FreqAIFuturesStrategy(IStrategy):
    minimal_roi = {
        "0": 0.005,
        "10": 0.003,
        "20": 0.001
    }

    stoploss = -0.01
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.002
    trailing_only_offset_is_reached = True
    
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    
    timeframe = '5m'

    def freqai_config(self) -> Dict:
        return {
            "enabled": True,
            "purge_old_models": True,
            "train_period_days": 1,
            "backtest_period_days": 1,
            "live_retrain_hours": 1,
            "expiration_hours": 2,
            "identifier": "futures_ml_v4",
            "fit_live_predictions_candles": 30,
            
            "data_split_parameters": {
                "test_size": 0.2,
                "shuffle": True,
                "stratify": True,
                "random_state": 42
            },

            "feature_parameters": {
                "include_timeframes": ["5m"],
                "include_corr_pairlist": [],
                "label_period_candles": 2,
                "include_shifted_candles": 1,
                "DI_threshold": 0.0,
                "weight_factor": 0.9,
                
                "indicator_periods": {
                    "rsi": [14],
                    "mfi": [14]
                },
                "normalize_features": "standard"
            },

            "model_training_parameters": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "num_leaves": 16,
                "min_data_in_leaf": 2,
                "feature_fraction": 0.8,
                "early_stopping_rounds": 25
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.info(f"Populating indicators for pair: {metadata.get('pair')}")
        
        try:
            # Basis-Indikatoren
            dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
            dataframe['mfi'] = ta.MFI(dataframe['high'], dataframe['low'], 
                                    dataframe['close'], dataframe['volume'], timeperiod=14)
            
            # Trend Indikatoren
            dataframe['ema_9'] = ta.EMA(dataframe['close'], timeperiod=9)
            dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
            dataframe['trend'] = np.where(dataframe['ema_9'] > dataframe['ema_21'], 1, -1)
            
            # Initialisiere do_predict
            dataframe['do_predict'] = 1
            
            logger.info(f"Indicators populated successfully for {metadata.get('pair')}")
        except Exception as e:
            logger.error(f"Error populating indicators for {metadata.get('pair')}: {str(e)}")
            dataframe['do_predict'] = 0
        
        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, 
                                     metadata: dict) -> DataFrame:
        logger.info(f"Feature engineering for {metadata.get('pair')} with period {period}")
        
        try:
            # Preis Features
            dataframe['price_mean'] = dataframe['close'].rolling(period).mean()
            dataframe['price_std'] = dataframe['close'].rolling(period).std()
            
            # Volumen Features
            dataframe['volume_mean'] = dataframe['volume'].rolling(window=period).mean()
            dataframe['volume_std'] = dataframe['volume'].rolling(window=period).std()
            
            logger.info(f"Feature engineering completed for {metadata.get('pair')}")
        except Exception as e:
            logger.error(f"Error in feature engineering for {metadata.get('pair')}: {str(e)}")
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        try:
            logger.info(f"Starting FreqAI target setting for {metadata.get('pair')}")
            
            dataframe['&s-up_or_down'] = (
                (dataframe['close'].shift(-2) > dataframe['close'])
            ).astype(int)
            
            logger.info(f"FreqAI targets set successfully for {metadata.get('pair')} with {len(dataframe)} rows")
                
        except Exception as e:
            logger.error(f"Error setting targets for {metadata.get('pair')}: {str(e)}")
            dataframe['&s-up_or_down'] = 0
                
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get('pair')
        
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        try:
            if '&s-up_or_down_prediction' not in dataframe.columns:
                logger.info(f"Waiting for model training to complete for {pair}")
                return dataframe
            
            # Super einfache Entry-Bedingungen
            long_conditions = (
                (dataframe['do_predict'] == 1) &
                (dataframe['&s-up_or_down_prediction'] > 0.5)
            )

            short_conditions = (
                (dataframe['do_predict'] == 1) &
                (dataframe['&s-up_or_down_prediction'] < 0.5)
            )

            dataframe.loc[long_conditions, 'enter_long'] = 1
            dataframe.loc[short_conditions, 'enter_short'] = 1
            
            entries_count = dataframe['enter_long'].sum() + dataframe['enter_short'].sum()
            if entries_count > 0:
                logger.info(f"{pair} - Generated {dataframe['enter_long'].sum()} long and {dataframe['enter_short'].sum()} short signals")
            
        except Exception as e:
            logger.error(f"Error in entry signal generation for {pair}: {str(e)}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get('pair')
        
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        try:
            if '&s-up_or_down_prediction' not in dataframe.columns:
                return dataframe
            
            # Super einfache Exit-Bedingungen
            long_exit_conditions = (
                (dataframe['do_predict'] == 1) &
                (dataframe['&s-up_or_down_prediction'] < 0.48)
            )
            
            short_exit_conditions = (
                (dataframe['do_predict'] == 1) &
                (dataframe['&s-up_or_down_prediction'] > 0.52)
            )
            
            dataframe.loc[long_exit_conditions, 'exit_long'] = 1
            dataframe.loc[short_exit_conditions, 'exit_short'] = 1
            
            exits_count = dataframe['exit_long'].sum() + dataframe['exit_short'].sum()
            if exits_count > 0:
                logger.info(f"{pair} - Generated {dataframe['exit_long'].sum()} long exits and {dataframe['exit_short'].sum()} short exits")
            
        except Exception as e:
            logger.error(f"Error in exit signal generation for {pair}: {str(e)}")
        
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, 
                       current_rate: float, proposed_stake: float,
                       min_stake: float, max_stake: float, leverage: float,
                       entry_tag: Optional[str], side: str, **kwargs) -> float:
        
        return min_stake  # Minimales Risiko für den Test

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
            proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 1.0  # Minimaler Hebel für den Test