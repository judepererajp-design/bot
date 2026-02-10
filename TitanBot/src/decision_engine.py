"""
TITAN-X DECISION ENGINE (SCORECARD)
------------------------------------------------------------------------------
The Final Judge.
Aggregates all data points into a unified Institutional Score (0-100).
Now includes:
- Technicals (Pattern Confidence)
- Volume Profile (VWAP/POC context)
- Order Flow (Imbalance)
- Correlation (Beta/BTC Check)
- Derivatives (Funding Rates)
- Sentiment (AI/LLM)
"""

from typing import Dict, Any

class DecisionEngine:
    def __init__(self):
        # Adjusted Weights to include Volume Profile (Total = 1.0)
        self.weights = {
            'technical': 0.30,   # Strategy Confidence
            'volume': 0.20,      # VWAP/POC Context (High Importance)
            'orderflow': 0.15,   # Tape Reading
            'derivatives': 0.15, # Funding/Crowd
            'correlation': 0.10, # Macro Beta
            'sentiment': 0.10    # AI/News
        }

    def generate_scorecard(self, 
                           tech_signal: Dict, 
                           volume_profile: Dict,
                           order_flow: Dict, 
                           correlation: Dict, 
                           sentiment: Dict,
                           derivatives: Dict) -> Dict[str, Any]:
        
        # 1. Technical Score (Base Confidence)
        score_tech = tech_signal.get('confidence', 50)
        
        # 2. Volume Profile Score (Institutional Context)
        score_vol = self._score_volume(volume_profile, tech_signal['direction'])
        
        # 3. Order Flow Score (Liquidity Imbalance)
        score_of = order_flow.get('imbalance_score', 50)
        
        # 4. Correlation Score (Beta/BTC alignment)
        score_corr = correlation.get('correlation_score', 50)
        
        # 5. Sentiment Score (AI/News)
        score_sent = sentiment.get('sentiment_score', 50)

        # 6. Derivatives Score (Contrarian Funding)
        score_deriv = derivatives.get('deriv_score', 50)
        
        # 7. Weighted Average Calculation
        final_score = (
            (score_tech * self.weights['technical']) +
            (score_vol * self.weights['volume']) +
            (score_of * self.weights['orderflow']) +
            (score_corr * self.weights['correlation']) +
            (score_sent * self.weights['sentiment']) +
            (score_deriv * self.weights['derivatives'])
        )
        
        # 8. Recommendation Generation
        recommendation = "WAIT"
        direction = tech_signal.get('direction', 'LONG')
        
        # Thresholds
        if final_score >= 80: 
            recommendation = "STRONG_BUY" if direction == 'LONG' else "STRONG_SELL"
        elif final_score >= 65: 
            recommendation = "BUY" if direction == 'LONG' else "SELL"
        elif final_score < 40:
            recommendation = "AVOID"
        
        return {
            'final_score': round(final_score, 1),
            'recommendation': recommendation,
            'components': {
                'technical': score_tech,
                'volume': score_vol,
                'orderflow': score_of,
                'correlation': score_corr,
                'derivatives': score_deriv,
                'sentiment': score_sent
            }
        }

    def _score_volume(self, vp: Dict, direction: str) -> float:
        """
        Scoring Logic for Volume Profile & VWAP.
        """
        context = vp.get('signal_context', 'NEUTRAL')
        score = 50.0 # Neutral baseline
        
        if direction == 'LONG':
            if context == 'OVERSOLD_VWAP': score = 85.0     # Cheap (Below bands)
            elif context == 'AT_POC_SUPPORT': score = 90.0  # High Volume Support
            elif context == 'ABOVE_VALUE_AREA': score = 40.0 # Chasing the top
            elif context == 'OVERBOUGHT_VWAP': score = 20.0  # Don't buy the top
            elif context == 'IN_VALUE_AREA': score = 50.0
            
        elif direction == 'SHORT':
            if context == 'OVERBOUGHT_VWAP': score = 85.0   # Expensive (Above bands)
            elif context == 'AT_POC_SUPPORT': score = 40.0  # Don't short support
            elif context == 'BELOW_VALUE_AREA': score = 40.0 # Chasing the bottom
            elif context == 'OVERSOLD_VWAP': score = 20.0   # Don't short the hole
            elif context == 'IN_VALUE_AREA': score = 50.0

        return score