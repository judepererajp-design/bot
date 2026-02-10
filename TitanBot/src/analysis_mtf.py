"""
TITAN-X MULTI-TIMEFRAME ANALYZER
------------------------------------------------------------------------------
Trend Validation System.
- Checks Higher Timeframe (HTF) Alignment.
- Calculates EMA Slope to detect Chop vs Trend.
- Filters signals trading into Overbought/Oversold zones.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("MTFAnalyzer")

    def confirm_trend(self, signal: Dict[str, Any], mtf_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Returns True if the signal aligns with the macro trend.
        Returns False if it fights the trend or enters bad chop.
        """
        symbol = signal['symbol']
        tf = signal['timeframe']
        direction = signal['direction']
        
        # 1. Select Comparison Timeframe
        # 15m -> check 4h | 1h -> check 4h | 4h -> check 1d
        compare_tf = self._get_higher_tf(tf)
        
        if compare_tf not in mtf_data:
            # If data missing, Fail Open (Allow) but warn
            return True
            
        df = mtf_data[compare_tf]
        if len(df) < 200: return True 
        
        try:
            # 2. Calculate Indicators (Vectorized)
            closes = df['close']
            
            # EMA 200 (The Trend Line)
            ema200 = closes.ewm(span=200, adjust=False).mean()
            current_ema = ema200.iloc[-1]
            prev_ema = ema200.iloc[-5] # 5 candles ago
            
            # RSI 14 (Momentum)
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            current_price = closes.iloc[-1]
            
            # 3. Validation Logic
            
            # A. Slope Check (Avoid Chop)
            # Calculate % change of EMA over last 5 bars
            ema_slope = (current_ema - prev_ema) / prev_ema
            is_flat = abs(ema_slope) < 0.0005 # Threshold for "Flat/Chop"
            
            # If market is flat, we ONLY allow Reversal patterns, not Trend patterns
            pattern_name = signal.get('pattern_name', '').lower()
            is_trend_pattern = 'flag' in pattern_name or 'pennant' in pattern_name
            
            if is_flat and is_trend_pattern:
                # self.logger.debug(f"🚫 {symbol} Skipped: Market is chopping (Flat EMA) on {compare_tf}")
                return False

            # B. Directional Alignment
            is_uptrend = current_price > current_ema
            
            if direction == 'LONG':
                # Don't buy if HTF is downtrend AND slope is negative
                if not is_uptrend and ema_slope < -0.001:
                    # self.logger.debug(f"🚫 {symbol} LONG Rejected: Downtrend on {compare_tf}")
                    return False
                
                # Don't buy if RSI is Overbought (>75)
                if rsi > 75:
                    return False

            elif direction == 'SHORT':
                # Don't short if HTF is uptrend AND slope is positive
                if is_uptrend and ema_slope > 0.001:
                    # self.logger.debug(f"🚫 {symbol} SHORT Rejected: Uptrend on {compare_tf}")
                    return False
                    
                # Don't short if RSI is Oversold (<25)
                if rsi < 25:
                    return False
                    
            return True

        except Exception as e:
            self.logger.error(f"MTF Error {symbol}: {e}")
            return True # Fail safe

    def _get_higher_tf(self, tf: str) -> str:
        mapping = {
            '1m': '15m',
            '5m': '1h',
            '15m': '4h',
            '1h': '4h',
            '4h': '1d',
            '1d': '1w'
        }
        return mapping.get(tf, '4h')