"""
TITAN-X GEOMETRIC PATTERN ENGINE
------------------------------------------------------------------------------
Advanced Chart Pattern Recognition using Scipy.
- Identifies Pivots (Swing Highs/Lows)
- Calculates Trendline Slopes via Linear Regression
- Validates Volatility Contraction (Flags/Triangles)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import argrelextrema
import logging
from typing import List, Dict, Any, Tuple

class GeometricPatternDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('geometric', {})
        self.logger = logging.getLogger("GeoDetector")
        
        # Parameters
        self.lookback = self.config.get('lookback', 50)
        self.pivot_order = self.config.get('pivot_order', 5)
        self.tolerance = self.config.get('slope_tolerance', 0.05)

    def detect(self, symbol: str, mtf_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Scans all timeframes for geometric structures.
        """
        signals = []
        
        for tf, df in mtf_data.items():
            if df is None or len(df) < self.lookback: continue
            
            try:
                # 1. Pre-calculate Pivots (Crucial for all patterns)
                highs = df['high'].values
                lows = df['low'].values
                closes = df['close'].values
                
                pivots_high_idx = argrelextrema(highs, np.greater, order=self.pivot_order)[0]
                pivots_low_idx = argrelextrema(lows, np.less, order=self.pivot_order)[0]
                
                # 2. Run Pattern Checks
                if self.config.get('detect_flags', True):
                    flag = self._check_flag(symbol, tf, closes, highs, lows)
                    if flag: signals.append(flag)
                
                if self.config.get('detect_triangles', True):
                    tri = self._check_triangle(symbol, tf, highs, lows, pivots_high_idx, pivots_low_idx)
                    if tri: signals.append(tri)
                    
                if self.config.get('detect_double_top_bottom', True):
                    dbl = self._check_double_patterns(symbol, tf, highs, lows, pivots_high_idx, pivots_low_idx)
                    if dbl: signals.append(dbl)

                if self.config.get('detect_head_shoulders', True):
                    hs = self._check_head_shoulders(symbol, tf, highs, lows, pivots_high_idx, pivots_low_idx)
                    if hs: signals.append(hs)

            except Exception as e:
                self.logger.warning(f"Math Error on {symbol} {tf}: {e}")
                continue

        return signals

    # ========================================================
    # PATTERN LOGIC: BULL/BEAR FLAGS
    # ========================================================
    def _check_flag(self, symbol, tf, closes, highs, lows):
        # Logic: Strong Impulse -> Low Volatility Consolidation -> Breakout
        
        # 1. Check Pole (Last 15 candles vs previous)
        impulse_len = 15
        if len(closes) < impulse_len + 5: return None
        
        start_price = closes[-(impulse_len + 5)]
        peak_price = closes[-5] # Consolidation start
        
        move_pct = (peak_price - start_price) / start_price
        
        # 2. Check Consolidation (Last 5 candles)
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        volatility = (max(recent_highs) - min(recent_lows)) / peak_price
        
        # Thresholds
        min_move = 0.04 # 4% move required for pole
        max_vol = 0.02  # Consolidation must be tight (<2%)
        
        current_price = closes[-1]
        
        # Bull Flag
        if move_pct > min_move and volatility < max_vol:
            # Check for Breakout of consolidation high
            if current_price > max(recent_highs[:-1]):
                return {
                    'name': 'Bull Flag',
                    'symbol': symbol,
                    'timeframe': tf,
                    'direction': 'LONG',
                    'confidence': 85,
                    'entry_price': current_price
                }
                
        # Bear Flag
        elif move_pct < -min_move and volatility < max_vol:
            if current_price < min(recent_lows[:-1]):
                return {
                    'name': 'Bear Flag',
                    'symbol': symbol,
                    'timeframe': tf,
                    'direction': 'SHORT',
                    'confidence': 85,
                    'entry_price': current_price
                }
        return None

    # ========================================================
    # PATTERN LOGIC: TRIANGLES / WEDGES
    # ========================================================
    def _check_triangle(self, symbol, tf, highs, lows, p_highs, p_lows):
        if len(p_highs) < 3 or len(p_lows) < 3: return None
        
        # Get last 3 pivots
        y_highs = highs[p_highs[-3:]]
        x_highs = p_highs[-3:]
        
        y_lows = lows[p_lows[-3:]]
        x_lows = p_lows[-3:]
        
        # Calculate Slopes
        slope_res, _, _, _, _ = linregress(x_highs, y_highs)
        slope_sup, _, _, _, _ = linregress(x_lows, y_lows)
        
        # Ascending Triangle (Flat Resistance, Rising Support)
        if abs(slope_res) < 0.0005 and slope_sup > 0.0005:
            return {
                'name': 'Ascending Triangle',
                'symbol': symbol,
                'timeframe': tf,
                'direction': 'LONG',
                'confidence': 75,
                'entry_price': highs[-1]
            }

        # Descending Triangle (Falling Resistance, Flat Support)
        if slope_res < -0.0005 and abs(slope_sup) < 0.0005:
            return {
                'name': 'Descending Triangle',
                'symbol': symbol,
                'timeframe': tf,
                'direction': 'SHORT',
                'confidence': 75,
                'entry_price': lows[-1]
            }
            
        # Symmetrical Triangle (Converging)
        if slope_res < -0.0005 and slope_sup > 0.0005:
            # Direction depends on breakout
            current = highs[-1]
            # Simple logic: assume continuation of prior trend (not implemented here fully)
            # Defaulting to breakout check could be added
            return {
                'name': 'Symmetrical Triangle',
                'symbol': symbol,
                'timeframe': tf,
                'direction': 'NEUTRAL', # Needs breakout confirmation
                'confidence': 60,
                'entry_price': current
            }
            
        return None

    # ========================================================
    # PATTERN LOGIC: DOUBLE TOP/BOTTOM
    # ========================================================
    def _check_double_patterns(self, symbol, tf, highs, lows, p_highs, p_lows):
        if len(p_highs) < 2: return None
        
        # Double Top
        top1 = highs[p_highs[-2]]
        top2 = highs[p_highs[-1]]
        
        # Peaks match within 1%
        if abs(top1 - top2) / top1 < 0.01:
            return {
                'name': 'Double Top',
                'symbol': symbol,
                'timeframe': tf,
                'direction': 'SHORT',
                'confidence': 70,
                'entry_price': lows[-1]
            }
            
        # Double Bottom
        if len(p_lows) < 2: return None
        bot1 = lows[p_lows[-2]]
        bot2 = lows[p_lows[-1]]
        
        if abs(bot1 - bot2) / bot1 < 0.01:
            return {
                'name': 'Double Bottom',
                'symbol': symbol,
                'timeframe': tf,
                'direction': 'LONG',
                'confidence': 70,
                'entry_price': highs[-1]
            }
            
        return None

    # ========================================================
    # PATTERN LOGIC: HEAD & SHOULDERS
    # ========================================================
    def _check_head_shoulders(self, symbol, tf, highs, lows, p_highs, p_lows):
        if len(p_highs) < 3: return None
        
        # Extract Peaks
        LS = highs[p_highs[-3]]
        H = highs[p_highs[-2]]
        RS = highs[p_highs[-1]]
        
        # Head higher than shoulders
        if H > LS and H > RS:
            # Shoulders roughly equal height (5% tolerance)
            if abs(LS - RS) / LS < 0.05:
                return {
                    'name': 'Head & Shoulders',
                    'symbol': symbol,
                    'timeframe': tf,
                    'direction': 'SHORT',
                    'confidence': 80,
                    'entry_price': lows[-1]
                }
        
        # Inverse H&S (Check Lows)
        if len(p_lows) < 3: return None
        LS = lows[p_lows[-3]]
        H = lows[p_lows[-2]]
        RS = lows[p_lows[-1]]
        
        if H < LS and H < RS:
            if abs(LS - RS) / abs(LS) < 0.05:
                return {
                    'name': 'Inv. Head & Shoulders',
                    'symbol': symbol,
                    'timeframe': tf,
                    'direction': 'LONG',
                    'confidence': 80,
                    'entry_price': highs[-1]
                }
                
        return None