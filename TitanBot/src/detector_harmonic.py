"""
TITAN-X HARMONIC PATTERN ENGINE (INSTITUTIONAL)
------------------------------------------------------------------------------
Advanced Fibonacci Pattern Recognition.
Detects XABCD structures based on strict ratio definitions.
Enforces vector logic for Direction (XA Leg).
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class RatioConfig:
    # Acceptable Fibonacci Ratios
    name: str
    xB_min: float; xB_max: float  # B point retracement of XA
    aC_min: float; aC_max: float  # C point retracement of AB
    bD_min: float; bD_max: float  # D point projection of BC
    xD_min: float; xD_max: float  # D point retracement of XA (The Target)

class HarmonicPatternDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('harmonics', {})
        self.logger = logging.getLogger("HarmonicDetector")
        
        # Tolerance (e.g. 0.10 = +/- 10% error allowed on ratios)
        self.tol = self.config.get('tolerance', 0.10)
        self.pivot_order = 5

        # Standard Harmonic Library
        self.patterns = [
            # GARTLEY
            RatioConfig("Gartley", 0.618, 0.618, 0.382, 0.886, 1.272, 1.618, 0.786, 0.786),
            # BAT
            RatioConfig("Bat", 0.382, 0.500, 0.382, 0.886, 1.618, 2.618, 0.886, 0.886),
            # BUTTERFLY
            RatioConfig("Butterfly", 0.786, 0.786, 0.382, 0.886, 1.618, 2.618, 1.270, 1.618),
            # CRAB
            RatioConfig("Crab", 0.382, 0.618, 0.382, 0.886, 2.240, 3.618, 1.618, 1.618)
        ]

    def detect(self, symbol: str, mtf_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        if not self.config.get('enabled', True): return []
        
        signals = []
        for tf, df in mtf_data.items():
            if df is None or len(df) < 100: continue
            
            try:
                # 1. Get ZigZag Pivots
                pivots = self._get_zigzag_pivots(df)
                if len(pivots) < 5: continue
                
                # Check last 5 points (X, A, B, C, D)
                points = pivots[-5:] 
                
                # 2. Check Pattern Ratios
                pattern = self._check_ratios(points)
                
                if pattern:
                    # 3. Determine Direction (Based on XA Leg)
                    # Bullish: X is Low, A is High (M shape) -> Price drops to D to buy
                    # Bearish: X is High, A is Low (W shape) -> Price rallies to D to sell
                    
                    X = points[0]['price']
                    A = points[1]['price']
                    D = points[4]['price']
                    
                    if X < A: 
                        # X is low, A is high. Pattern implies D will be a low.
                        # Wait... Standard Bullish Harmonic looks like "M"
                        # X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                        direction = 'LONG'
                        stop_loss = X # Invalidated if X breaks
                    else:
                        direction = 'SHORT'
                        stop_loss = X
                    
                    signals.append({
                        'name': pattern.name,
                        'symbol': symbol,
                        'timeframe': tf,
                        'direction': direction,
                        'confidence': 90, 
                        'entry_price': D,
                        'stop_loss': stop_loss,
                        'timestamp': df['timestamp'].iloc[-1].timestamp()
                    })

            except Exception:
                continue

        return signals

    def _get_zigzag_pivots(self, df: pd.DataFrame) -> List[Dict]:
        """Extracts strictly alternating High/Low pivots."""
        highs = df['high'].values
        lows = df['low'].values
        
        h_idx = argrelextrema(highs, np.greater, order=self.pivot_order)[0]
        l_idx = argrelextrema(lows, np.less, order=self.pivot_order)[0]
        
        pivots = []
        for i in h_idx: pivots.append({'index': i, 'price': highs[i], 'type': 'high'})
        for i in l_idx: pivots.append({'index': i, 'price': lows[i], 'type': 'low'})
        
        pivots.sort(key=lambda x: x['index'])
        
        # Enforce Alternation
        if not pivots: return []
        clean = [pivots[0]]
        
        for p in pivots[1:]:
            if p['type'] != clean[-1]['type']:
                clean.append(p)
            else:
                # Update extreme if same type
                if p['type'] == 'high' and p['price'] > clean[-1]['price']:
                    clean[-1] = p
                elif p['type'] == 'low' and p['price'] < clean[-1]['price']:
                    clean[-1] = p
                    
        return clean

    def _check_ratios(self, p: List[Dict]) -> Any:
        X, A, B, C, D = [pt['price'] for pt in p]
        
        # Calculate Leg Lengths
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        XD = abs(D - X)
        
        if XA == 0 or AB == 0 or BC == 0: return None
        
        # Calculate Ratios
        xB = AB / XA
        aC = BC / AB
        bD = CD / BC
        xD = XD / XA
        
        for pat in self.patterns:
            # Check all legs with tolerance
            if not self._is_near(xB, pat.xB_min, pat.xB_max): continue
            if not self._is_near(aC, pat.aC_min, pat.aC_max): continue
            if not self._is_near(bD, pat.bD_min, pat.bD_max): continue
            if not self._is_near(xD, pat.xD_min, pat.xD_max): continue
            
            return pat
            
        return None

    def _is_near(self, value, t_min, t_max) -> bool:
        # Dynamic tolerance scaling
        lower = t_min * (1 - self.tol)
        upper = t_max * (1 + self.tol)
        return lower <= value <= upper