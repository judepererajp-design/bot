import pandas as pd
import numpy as np
from typing import Dict, Optional

class PriceActionStrategy:
    """
    TITAN-X PRICE ACTION MODULE (MANUAL ASSISTANT)
    ------------------------------------------------
    Detects high-probability candlestick patterns:
    1. Pin Bars (Liquidity Rejections)
    2. Engulfing Bars (Momentum Shifts)
    3. Inside Bars (Coiling/Consolidation)
    """
    def __init__(self, context: Dict):
        self.config = context.get('config', {})
        self.min_vol_mult = 1.2  # Pattern must have 1.2x avg volume to be valid

    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        # Price Action is best viewed on 1H and 4H for manual trading
        for tf in ['4h', '1h']:
            df = data.get(tf)
            if df is None or len(df) < 50: continue
            
            # Extract recent candles
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Volume Check (Must have decent participation)
            avg_vol = df['volume'].iloc[-22:-2].mean()
            has_vol = curr['volume'] > avg_vol

            # 1. CHECK FOR PIN BAR (Rejection)
            # --------------------------------
            pin_signal = self._check_pin_bar(curr, df, tf)
            if pin_signal:
                return self._build_signal(symbol, pin_signal, curr, tf)

            # 2. CHECK FOR ENGULFING (Momentum)
            # ---------------------------------
            engulf_signal = self._check_engulfing(curr, prev, has_vol)
            if engulf_signal:
                return self._build_signal(symbol, engulf_signal, curr, tf)

            # 3. CHECK FOR INSIDE BAR (Coil)
            # ------------------------------
            inside_signal = self._check_inside_bar(curr, prev)
            if inside_signal:
                return self._build_signal(symbol, inside_signal, curr, tf)

        return None

    def _check_pin_bar(self, curr, df, tf):
        body_size = abs(curr['close'] - curr['open'])
        wick_top = curr['high'] - max(curr['close'], curr['open'])
        wick_bottom = min(curr['close'], curr['open']) - curr['low']
        total_range = curr['high'] - curr['low']
        
        if total_range == 0: return None

        # Bullish Pin Bar (Hammer)
        # Long bottom wick, small top wick, occurs at local low
        if (wick_bottom > 2 * body_size) and (wick_top < body_size):
            # Context Check: Is this a local low? (Last 10 bars)
            lowest_recent = df['low'].iloc[-10:].min()
            if abs(curr['low'] - lowest_recent) / lowest_recent < 0.002: # Within 0.2% of low
                return {'name': 'Bullish Pin Bar', 'dir': 'LONG', 'conf': 75}

        # Bearish Pin Bar (Shooting Star)
        # Long top wick, small bottom wick, occurs at local high
        if (wick_top > 2 * body_size) and (wick_bottom < body_size):
            # Context Check: Is this a local high?
            highest_recent = df['high'].iloc[-10:].max()
            if abs(curr['high'] - highest_recent) / highest_recent < 0.002:
                return {'name': 'Bearish Pin Bar', 'dir': 'SHORT', 'conf': 75}
        
        return None

    def _check_engulfing(self, curr, prev, has_vol):
        # Bullish Engulfing
        # Prev Red, Curr Green, Curr Body > Prev Body, Curr Close > Prev High
        if (prev['close'] < prev['open']) and (curr['close'] > curr['open']):
            if (curr['close'] > prev['open']) and (curr['open'] < prev['close']):
                if has_vol:
                    return {'name': 'Bullish Engulfing', 'dir': 'LONG', 'conf': 80}
                return {'name': 'Bullish Engulfing (Low Vol)', 'dir': 'LONG', 'conf': 65}

        # Bearish Engulfing
        if (prev['close'] > prev['open']) and (curr['close'] < curr['open']):
            if (curr['close'] < prev['open']) and (curr['open'] > prev['close']):
                if has_vol:
                    return {'name': 'Bearish Engulfing', 'dir': 'SHORT', 'conf': 80}
                return {'name': 'Bearish Engulfing (Low Vol)', 'dir': 'SHORT', 'conf': 65}
        
        return None

    def _check_inside_bar(self, curr, prev):
        # Current range is strictly inside previous range
        if (curr['high'] < prev['high']) and (curr['low'] > prev['low']):
            # Inside bars are neutral, usually continuation or breakout setups
            # We flag them for the user to watch ("Watchlist Mode")
            return {'name': 'Inside Bar (Coil)', 'dir': 'NEUTRAL', 'conf': 60}
        return None

    def _build_signal(self, symbol, pattern, curr, tf):
        return {
            'strategy': 'Price Action',
            'pattern_name': pattern['name'],
            'symbol': symbol,
            'timeframe': tf,
            'direction': pattern['dir'],
            'entry_price': curr['close'],
            'entry': curr['close'],  # Added for compatibility
            'confidence': pattern['conf'],
            'timestamp': curr['timestamp'].timestamp()
        }
