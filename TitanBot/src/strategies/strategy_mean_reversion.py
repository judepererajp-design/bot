import pandas as pd
import numpy as np
from typing import Dict, Optional

class MeanReversionStrategy:
    """
    TITAN-X MEAN REVERSION MODULE (MANUAL ASSISTANT)
    ------------------------------------------------
    Detects market extremes and coiling energy:
    1. BB Squeeze (Volatility Contraction) - Watchlist setup
    2. RSI Divergence (Momentum Reversal) - Top/Bottom catching
    3. Keltner Channel Fade (Statistical Extremes)
    """
    def __init__(self, context: Dict):
        self.config = context.get('config', {})
        self.bb_length = 20
        self.bb_std = 2.0
        self.rsi_length = 14
        
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        # These setups work best on 1H and 4H
        for tf in ['4h', '1h']:
            df = data.get(tf)
            if df is None or len(df) < 50: continue
            
            # 1. CHECK FOR BB SQUEEZE (The "Coil")
            # ------------------------------------
            squeeze = self._check_bb_squeeze(df)
            if squeeze:
                return self._build_signal(symbol, squeeze, df.iloc[-1], tf)

            # 2. CHECK FOR RSI DIVERGENCE
            # ---------------------------
            div = self._check_divergence(df)
            if div:
                return self._build_signal(symbol, div, df.iloc[-1], tf)
                
            # 3. CHECK FOR KELTNER FADE
            # -------------------------
            fade = self._check_keltner_fade(df)
            if fade:
                return self._build_signal(symbol, fade, df.iloc[-1], tf)

        return None

    def _check_bb_squeeze(self, df):
        # Calculate BB
        close = df['close']
        sma = close.rolling(self.bb_length).mean()
        std = close.rolling(self.bb_length).std()
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        
        # Band Width
        bandwidth = (upper - lower) / sma
        
        # Check if current bandwidth is the lowest in 20 periods
        if len(bandwidth) < 20:
            return None
            
        min_bandwidth = bandwidth.rolling(20).min().iloc[-1]
        
        if bandwidth.iloc[-1] <= min_bandwidth and min_bandwidth > 0:
            # We are in a squeeze - NEUTRAL signal (watchlist)
            return {'name': 'Bollinger Squeeze (Coil)', 'dir': 'NEUTRAL', 'conf': 60}
            
        return None

    def _check_divergence(self, df):
        # Simplified Divergence Check (Peak vs Peak)
        if len(df) < 12:
            return None
            
        close = df['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        curr_price = close.iloc[-1]
        prev_price = close.iloc[-10:-2].max() if len(close) > 10 else 0
        curr_rsi = rsi.iloc[-1]
        prev_rsi_peak = rsi.iloc[-10:-2].max() if len(rsi) > 10 else 0
        
        # Bearish Divergence (Price Higher High, RSI Lower High)
        if curr_price > prev_price and curr_rsi < prev_rsi_peak:
            if curr_rsi > 60: # Must be somewhat overbought
                return {'name': 'Bearish RSI Divergence', 'dir': 'SHORT', 'conf': 75}
        
        # Bullish Divergence (Price Lower Low, RSI Higher Low)
        prev_price_low = close.iloc[-10:-2].min() if len(close) > 10 else 0
        prev_rsi_low = rsi.iloc[-10:-2].min() if len(rsi) > 10 else 0
        
        if curr_price < prev_price_low and curr_rsi > prev_rsi_low:
            if curr_rsi < 40: # Must be somewhat oversold
                return {'name': 'Bullish RSI Divergence', 'dir': 'LONG', 'conf': 75}
                
        return None

    def _check_keltner_fade(self, df):
        # Keltner Channels (EMA 20 +/- 2 ATR)
        close = df['close']
        high = df['high']
        low = df['low']
        
        if len(close) < 20:
            return None
            
        ema = close.ewm(span=20, adjust=False).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        upper = ema + (2 * atr)
        lower = ema - (2 * atr)
        
        curr = close.iloc[-1]
        
        # Price closes OUTSIDE the channel (Extreme Statistical Event)
        if curr > upper.iloc[-1]:
            # If next candle is Red, it's a fade signal (Mean Reversion)
            if close.iloc[-1] < df['open'].iloc[-1]: # Red Candle
                 return {'name': 'Keltner Channel Fade', 'dir': 'SHORT', 'conf': 70}
                 
        if curr < lower.iloc[-1]:
            if close.iloc[-1] > df['open'].iloc[-1]: # Green Candle
                 return {'name': 'Keltner Channel Fade', 'dir': 'LONG', 'conf': 70}
                 
        return None

    def _build_signal(self, symbol, pattern, curr, tf):
        return {
            'strategy': 'Mean Reversion',
            'pattern_name': pattern['name'],
            'symbol': symbol,
            'timeframe': tf,
            'direction': pattern['dir'],
            'entry_price': curr['close'],
            'entry': curr['close'],  # Added for compatibility
            'confidence': pattern['conf'],
            'timestamp': curr['timestamp'].timestamp()
        }
