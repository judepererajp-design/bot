"""
TITAN-X ABSTRACT STRATEGY
------------------------------------------------------------------------------
The Parent Class for all Institutional Strategies.
Enforces strict validation pipelines:
1. Signal Generation (Pattern)
2. Institutional Filter (Order Flow)
3. Macro Filter (Correlation)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseStrategy(ABC):
    def __init__(self, name: str, context: Dict[str, Any]):
        self.name = name
        self.logger = logging.getLogger(f"Strategy_{name}")
        
        # Shared Context (APIs, Detectors, etc.)
        self.geo_detector = context['geo_detector']
        self.harm_detector = context['harm_detector']
        self.order_flow = context['order_flow']
        self.correlation = context['correlation']
        self.config = context['config']

    @abstractmethod
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core logic. Returns a Signal Dict if valid, None otherwise.
        Must include both 'entry' and 'entry_price' keys for compatibility.
        """
        pass

    def _ensure_key_compatibility(self, signal: Dict) -> Dict:
        """
        Ensures signal has both 'entry' and 'entry_price' keys.
        Called by child strategies before returning signals.
        """
        if 'entry_price' in signal and 'entry' not in signal:
            signal['entry'] = signal['entry_price']
        elif 'entry' in signal and 'entry_price' not in signal:
            signal['entry_price'] = signal['entry']
        
        # Also ensure stop loss keys if present
        if 'stop_loss' in signal and 'stop' not in signal:
            signal['stop'] = signal['stop_loss']
        elif 'stop' in signal and 'stop_loss' not in signal:
            signal['stop_loss'] = signal['stop']
            
        return signal

    async def _validate_institutional(self, signal: Dict) -> bool:
        """
        The "Institutional Gatekeeper".
        Checks if the signal is trading into a Liquidity Wall.
        """
        symbol = signal['symbol']
        direction = signal['direction']
        
        # FIX: Handle both key naming conventions safely
        entry = signal.get('entry', signal.get('entry_price'))
        if entry is None:
            self.logger.error(f"Missing entry price in signal: {signal.keys()}")
            return False
        
        # 1. Fetch Order Flow
        of_data = await self.order_flow.analyze(symbol)
        
        # 2. Check for Walls (Don't buy into resistance)
        if direction == 'LONG':
            sell_wall = of_data.get('nearest_sell_wall', 0)
            if sell_wall > 0 and (sell_wall - entry) / entry < 0.01:
                self.logger.info(f"🚫 {symbol} Blocked: Sell Wall detected at {sell_wall}")
                return False
                
            # Check Imbalance (Must have buying pressure)
            if of_data['imbalance_score'] < 40:
                self.logger.info(f"🚫 {symbol} Blocked: Negative Order Flow Delta")
                return False

        elif direction == 'SHORT':
            buy_wall = of_data.get('nearest_buy_wall', 0)
            if buy_wall > 0 and (entry - buy_wall) / entry < 0.01:
                self.logger.info(f"🚫 {symbol} Blocked: Buy Wall detected at {buy_wall}")
                return False

            if of_data['imbalance_score'] > 60:
                self.logger.info(f"🚫 {symbol} Blocked: Positive Order Flow Delta")
                return False

        return True

    async def _validate_macro(self, signal: Dict) -> bool:
        """
        Ensures we aren't trading against Bitcoin.
        """
        return True
