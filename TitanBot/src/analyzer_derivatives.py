"""
TITAN-X DERIVATIVES ANALYZER
------------------------------------------------------------------------------
Analyzes Funding Rates and Open Interest.
"""

import logging
from typing import Dict, Any

class DerivativesAnalyzer:
    def __init__(self, api_interface):
        self.api = api_interface
        self.logger = logging.getLogger("Derivatives")

    async def analyze(self, symbol: str) -> Dict[str, Any]:
        try:
            # Fetch Funding Rate
            # Note: Not all exchanges support this in the same way via CCXT
            # We use a safe try-block
            funding = await self.api.exchange.fetch_funding_rate(symbol)
            
            rate = funding['fundingRate']
            predicted = funding.get('predictedFundingRate', 0)
            
            # Interpret
            sentiment = "NEUTRAL"
            score = 50
            
            # Annualized %
            apr = rate * 3 * 365 * 100 
            
            if apr > 50: # Extremely Bullish Crowd -> Contrarian Short
                sentiment = "OVERHEATED_LONG"
                score = 20 # Bearish signal
            elif apr < -20: # Extremely Bearish Crowd -> Contrarian Long
                sentiment = "OVERCROWDED_SHORT"
                score = 80 # Bullish signal
                
            return {
                'funding_rate': rate,
                'funding_apr': apr,
                'deriv_sentiment': sentiment,
                'deriv_score': score
            }

        except Exception:
            # Symbol might not have funding (Spot pair?)
            return {'deriv_score': 50, 'deriv_sentiment': 'UNAVAILABLE'}