"""
TITAN-X POSITION OPTIMIZER (SIGNAL BOT EDITION)
------------------------------------------------------------------------------
Since this is a signal-only bot, we provide SUGGESTED position sizing
based on simple rules rather than complex Kelly calculations.
"""

import logging
from typing import Dict, Any, List

class PositionOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("PositionOptimizer")
        
        # Simple risk limits for signal suggestions
        self.max_risk_per_trade = 0.02    # 2% MAX per trade
        self.min_risk_per_trade = 0.0025  # 0.25% minimum
        
        # Strategy risk multipliers (adjust based on your preferences)
        self.strategy_multipliers = {
            'Institutional Breakout': 1.2,
            'Institutional Reversal': 1.0,
            'Smart Money': 1.3,
            'Price Action': 1.1,
            'Mean Reversion': 0.8,
            'default': 1.0
        }
        
    def calculate_position_size(self,
                               signal: Dict[str, Any],
                               account_size: float,
                               regime_data: Dict[str, Any],
                               current_portfolio: List[Dict],
                               stop_loss_distance: float) -> Dict[str, Any]:
        """
        Calculates SUGGESTED position size for manual trading.
        Returns simple recommendations.
        """
        try:
            # 1. BASE RISK (1% of account)
            base_risk_pct = 0.01
            
            # 2. Adjust for STRATEGY TYPE
            strategy_name = signal.get('strategy', 'default')
            strategy_mult = self.strategy_multipliers.get(strategy_name, 1.0)
            adjusted_risk_pct = base_risk_pct * strategy_mult
            
            # 3. Adjust for CONFIDENCE
            confidence = signal.get('confidence', 50.0)
            if confidence >= 80:
                confidence_mult = 1.3
            elif confidence >= 70:
                confidence_mult = 1.15
            elif confidence >= 60:
                confidence_mult = 1.0
            elif confidence >= 50:
                confidence_mult = 0.85
            else:
                confidence_mult = 0.7
                
            adjusted_risk_pct *= confidence_mult
            
            # 4. Adjust for MARKET REGIME
            regime = regime_data.get('regime', 'UNKNOWN')
            regime_mult = self._get_regime_multiplier(regime)
            adjusted_risk_pct *= regime_mult
            
            # 5. Apply HARD LIMITS
            adjusted_risk_pct = max(self.min_risk_per_trade,
                                  min(self.max_risk_per_trade, adjusted_risk_pct))
            
            # 6. Calculate $ amounts
            risk_amount = account_size * adjusted_risk_pct
            
            # 7. Calculate POSITION SIZE (units)
            if stop_loss_distance > 0:
                position_size = risk_amount / abs(stop_loss_distance)
            else:
                position_size = 0
                self.logger.warning(f"Zero stop distance for {signal.get('symbol')}")
            
            # 8. Generate SUGGESTION TEXT
            suggestion = self._generate_suggestion_text(strategy_name, confidence, regime)
            
            return {
                'risk_amount': round(risk_amount, 2),
                'position_size': position_size,
                'risk_percent': round(adjusted_risk_pct * 100, 2),
                'suggestion': suggestion,
                'stop_distance': stop_loss_distance,
                'strategy_multiplier': strategy_mult,
                'confidence_multiplier': confidence_mult,
                'regime_multiplier': regime_mult
            }
            
        except Exception as e:
            self.logger.error(f"Position calc error: {e}")
            # FAIL SAFE: 1% risk
            return {
                'risk_amount': account_size * 0.01,
                'position_size': 0,
                'risk_percent': 1.0,
                'suggestion': 'Error in calculation - use 1% risk',
                'stop_distance': stop_loss_distance
            }
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Simple regime-based adjustments.
        """
        multipliers = {
            'STRONG_UPTREND': 1.3,
            'WEAK_UPTREND': 1.1,
            'STRONG_DOWNTREND': 1.3,
            'WEAK_DOWNTREND': 1.1,
            'RANGING': 0.7,
            'HIGH_VOLATILITY': 0.5,
            'LOW_VOLATILITY': 1.2,
            'UNKNOWN': 1.0
        }
        return multipliers.get(regime, 1.0)
    
    def _generate_suggestion_text(self, strategy: str, confidence: float, regime: str) -> str:
        """Generates human-readable suggestion for manual trading."""
        
        confidence_text = "HIGH" if confidence >= 75 else "MEDIUM" if confidence >= 60 else "LOW"
        
        # Strategy-specific advice
        advice_map = {
            'Institutional Breakout': "Enter on breakout confirmation with volume",
            'Institutional Reversal': "Enter on reversal candle close",
            'Smart Money': "Enter on retrace to identified structure",
            'Price Action': "Enter on pattern completion",
            'Mean Reversion': "Scale in gradually on extreme moves",
            'default': "Enter at market with tight stop"
        }
        
        advice = advice_map.get(strategy, advice_map['default'])
        
        return f"{confidence_text} confidence | {regime.replace('_', ' ').title()} | {advice}"
    
    def _get_asset_sector(self, symbol: str) -> str:
        """
        Simple sector detection for correlation awareness.
        """
        symbol_lower = symbol.lower()
        
        if 'btc' in symbol_lower:
            return 'bitcoin'
        elif 'eth' in symbol_lower:
            return 'ethereum'
        elif any(x in symbol_lower for x in ['sol', 'ada', 'dot', 'avax']):
            return 'layer1'
        elif any(x in symbol_lower for x in ['link', 'uni', 'aave', 'comp']):
            return 'defi'
        elif any(x in symbol_lower for x in ['matic', 'arb', 'op']):
            return 'layer2'
        else:
            return 'altcoin'
