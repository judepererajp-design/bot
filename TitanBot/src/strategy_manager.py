"""
TITAN-X STRATEGY MANAGER (FULL SUITE)
------------------------------------------------------------------------------
The Central Hub for Strategy Execution.
Includes:
1. Breakout (Auto/Manual)
2. Reversal (Auto/Manual)
3. Price Action (Manual)
4. Smart Money Concepts (Manual)
5. Mean Reversion (Manual) <--- NEW
"""

import logging
import asyncio
from typing import Dict, Any, List

# Import Strategies
from .strategies.strategy_breakout import InstitutionalBreakout
from .strategies.strategy_reversal import InstitutionalReversal
from .strategies.strategy_price_action import PriceActionStrategy
from .strategies.strategy_smc import SmartMoneyStrategy
from .strategies.strategy_mean_reversion import MeanReversionStrategy  # <--- NEW IMPORT

class StrategyManager:
    def __init__(self, context: Dict[str, Any]):
        self.logger = logging.getLogger("StrategyManager")
        self.strategies = []
        
        # Initialize All Strategies
        self.strategies.append(InstitutionalBreakout(context))
        self.strategies.append(InstitutionalReversal(context))
        self.strategies.append(PriceActionStrategy(context))
        self.strategies.append(SmartMoneyStrategy(context))
        self.strategies.append(MeanReversionStrategy(context)) # <--- NEW REGISTRATION
        
        self.logger.info(f"✅ Loaded {len(self.strategies)} Institutional Strategies")

    async def run_analysis(self, symbol: str, data: Dict[str, Any], regime_data: Dict[str, Any]) -> List[Dict]:
        valid_signals = []
        seen_patterns = set()

        # 1. Get Disabled List from Regime Detector
        if regime_data:
            disabled_keywords = regime_data['recommendations']['disabled_strategies']
        else:
            disabled_keywords = []

        # 2. Filter Strategies
        active_strategies = []
        for strat in self.strategies:
            strat_name = strat.__class__.__name__.lower()
            
            # --- MANUAL ASSISTANT LOGIC ---
            # We ALWAYS run manual assistant strategies (Price Action, SMC, Mean Reversion)
            # because the user wants to see these setups even if the auto-trader wouldn't take them.
            if any(x in strat_name for x in ["priceaction", "smartmoney", "meanreversion"]):
                active_strategies.append(strat)
                continue

            # --- AUTO-TRADER LOGIC (Strict Regime Compliance) ---
            is_disabled = False
            for kw in disabled_keywords:
                if kw in strat_name:
                    is_disabled = True
                    break
            
            if not is_disabled:
                active_strategies.append(strat)

        if not active_strategies:
            return []

        # 3. Launch Analysis
        tasks = [strategy.analyze(symbol, data) for strategy in active_strategies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Strategy Error on {symbol}: {res}")
                continue
                
            if res:
                sig_id = f"{res['pattern_name']}_{res['timeframe']}"
                if sig_id not in seen_patterns:
                    seen_patterns.add(sig_id)
                    res['regime_context'] = {
                        'regime': regime_data.get('regime', 'UNKNOWN'),
                        'adx': regime_data.get('adx_value', 0),
                    }
                    valid_signals.append(res)
                
        return valid_signals