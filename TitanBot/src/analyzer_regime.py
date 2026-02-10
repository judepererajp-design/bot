"""
TITAN-X MARKET REGIME DETECTOR
------------------------------------------------------------------------------
Detects Trending vs. Ranging markets with ADX.
Dynamically adjusts strategy eligibility and risk parameters.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from enum import Enum

class MarketRegime(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class MarketRegimeDetector:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("RegimeDetector")
        
        # Default thresholds
        self.adx_strong = self.config.get('adx_strong', 30)
        self.adx_weak = self.config.get('adx_weak', 20)
        self.volatility_high = self.config.get('volatility_high', 0.035)  # 3.5% ATR/Price
        self.volatility_low = self.config.get('volatility_low', 0.008)   # 0.8% ATR/Price
        
        # State tracking
        self.current_regime = MarketRegime.RANGING
        self.last_update = 0
        
    def analyze(self, df: pd.DataFrame, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyzes market regime and returns actionable insights.
        """
        if len(df) < 100:
            return self._default_result()
            
        try:
            # 1. Calculate ADX (Trend Strength)
            adx_value, trend_direction = self._calculate_adx(df)
            
            # 2. Calculate Volatility Regime
            atr_ratio, atr_value = self._calculate_volatility(df)
            
            # 3. Determine Primary Regime
            regime = self._classify_regime(adx_value, trend_direction, atr_ratio)
            self.current_regime = regime
            
            # 4. Generate Strategy Recommendations
            recommendations = self._generate_recommendations(regime, adx_value, atr_ratio)
            
            return {
                'regime': regime.value,
                'adx_value': float(adx_value),
                'trend_direction': trend_direction,
                'atr_ratio': float(atr_ratio),
                'atr_value': float(atr_value),
                'recommendations': recommendations,
                'timestamp': df['timestamp'].iloc[-1].timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Regime analysis error: {e}")
            return self._default_result()
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Calculates ADX and determines trend direction."""
        high, low, close = df['high'], df['low'], df['close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth with Wilder's smoothing (14 period)
        alpha = 1/14
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / atr
        
        # Calculate ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        direction = "NEUTRAL"
        if current_plus_di > current_minus_di:
            direction = "BULLISH"
        elif current_minus_di > current_plus_di:
            direction = "BEARISH"
            
        return float(current_adx), direction
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculates ATR/Price ratio for volatility regime."""
        high, low, close = df['high'], df['low'], df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        atr_ratio = current_atr / current_price if current_price > 0 else 0
        
        return float(atr_ratio), float(current_atr)
    
    def _classify_regime(self, adx: float, direction: str, atr_ratio: float) -> MarketRegime:
        """Classifies the market regime based on multiple factors."""
        
        # First, check volatility regime
        if atr_ratio > self.volatility_high:
            return MarketRegime.HIGH_VOLATILITY
        elif atr_ratio < self.volatility_low:
            return MarketRegime.LOW_VOLATILITY
            
        # Then, check trend regime
        if adx >= self.adx_strong:
            if direction == "BULLISH":
                return MarketRegime.STRONG_UPTREND
            elif direction == "BEARISH":
                return MarketRegime.STRONG_DOWNTREND
            else:
                return MarketRegime.RANGING
        elif adx >= self.adx_weak:
            if direction == "BULLISH":
                return MarketRegime.WEAK_UPTREND
            elif direction == "BEARISH":
                return MarketRegime.WEAK_DOWNTREND
            else:
                return MarketRegime.RANGING
        else:
            return MarketRegime.RANGING
    
    def _generate_recommendations(self, regime: MarketRegime, adx: float, atr_ratio: float) -> Dict[str, Any]:
        """Generates actionable trading recommendations."""
        rec = {
            'enabled_strategies': [],
            'disabled_strategies': [],
            'position_size_adjustment': 1.0,
            'stop_loss_multiplier': 1.0,
            'max_holding_period_hours': 24,
            'entry_aggressiveness': 'NORMAL'  # CONSERVATIVE, NORMAL, AGGRESSIVE
        }
        
        if regime == MarketRegime.STRONG_UPTREND:
            rec['enabled_strategies'] = ['breakout', 'trend_following']
            rec['disabled_strategies'] = ['reversal', 'mean_reversion']
            rec['position_size_adjustment'] = 1.2
            rec['entry_aggressiveness'] = 'AGGRESSIVE'
            
        elif regime == MarketRegime.STRONG_DOWNTREND:
            rec['enabled_strategies'] = ['breakout', 'trend_following']
            rec['disabled_strategies'] = ['reversal', 'mean_reversion']
            rec['position_size_adjustment'] = 1.2
            rec['entry_aggressiveness'] = 'AGGRESSIVE'
            
        elif regime == MarketRegime.RANGING:
            rec['enabled_strategies'] = ['reversal', 'mean_reversion']
            rec['disabled_strategies'] = ['breakout', 'trend_following']
            rec['position_size_adjustment'] = 0.8
            rec['stop_loss_multiplier'] = 0.7  # Tighter stops in ranges
            rec['max_holding_period_hours'] = 12
            rec['entry_aggressiveness'] = 'CONSERVATIVE'
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            rec['enabled_strategies'] = ['breakout']  # Only clear breakouts
            rec['disabled_strategies'] = ['reversal', 'trend_following', 'mean_reversion']
            rec['position_size_adjustment'] = 0.5  # Cut size in half
            rec['stop_loss_multiplier'] = 1.5  # Wider stops
            rec['entry_aggressiveness'] = 'CONSERVATIVE'
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            rec['enabled_strategies'] = ['breakout', 'reversal']
            rec['disabled_strategies'] = []
            rec['position_size_adjustment'] = 1.3  # Increase size
            rec['stop_loss_multiplier'] = 0.6  # Tighter stops
            rec['entry_aggressiveness'] = 'AGGRESSIVE'
            
        return rec
    
    def should_enable_strategy(self, strategy_name: str, regime_data: Dict[str, Any]) -> bool:
        """Checks if a strategy should be enabled in current regime."""
        enabled = regime_data['recommendations']['enabled_strategies']
        disabled = regime_data['recommendations']['disabled_strategies']
        
        # Map strategy names
        strategy_map = {
            'breakout': ['breakout', 'trend_following'],
            'reversal': ['reversal', 'mean_reversion']
        }
        
        for key, aliases in strategy_map.items():
            if key in strategy_name.lower():
                # Check if any alias is enabled
                for alias in aliases:
                    if alias in disabled:
                        return False
                    if alias in enabled:
                        return True
                        
        # Default: enable if not explicitly disabled
        for alias in disabled:
            if alias in strategy_name.lower():
                return False
        return True
    
    def _default_result(self):
        return {
            'regime': MarketRegime.RANGING.value,
            'adx_value': 15.0,
            'trend_direction': 'NEUTRAL',
            'atr_ratio': 0.015,
            'atr_value': 0.0,
            'recommendations': self._generate_recommendations(MarketRegime.RANGING, 15.0, 0.015),
            'timestamp': 0
        }
