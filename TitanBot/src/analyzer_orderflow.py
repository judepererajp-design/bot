"""
TITAN-X ORDER FLOW ENGINE (INSTITUTIONAL v2)
------------------------------------------------------------------------------
Analyzes Market Depth with Anti-Spoofing logic.
- Micro-Averaging: Takes 2 snapshots to detect flickering liquidity.
- Depth Weighting: Orders close to price matter more.
- Wall Detection: Identifies large blockades.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any

class OrderFlowAnalyzer:
    def __init__(self, api_interface):
        self.api = api_interface
        self.logger = logging.getLogger("OrderFlow")

    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches Order Book with Temporal Smoothing (Anti-Spoofing).
        """
        try:
            # SNAPSHOT 1
            ob1 = await self.api.exchange.fetch_order_book(symbol, limit=100)
            
            # Wait briefly to catch spoofing/flicker (0.5s)
            await asyncio.sleep(0.5)
            
            # SNAPSHOT 2
            ob2 = await self.api.exchange.fetch_order_book(symbol, limit=100)
            
            # Process
            metrics1 = self._process_book(ob1)
            metrics2 = self._process_book(ob2)
            
            # 1. Spoofing Check
            # If liquidity vanished between snapshots, confidence drops
            liquidity_volatility = abs(metrics1['total_liq'] - metrics2['total_liq']) / metrics1['total_liq']
            is_spoofing = liquidity_volatility > 0.30 # >30% change in 0.5s is suspicious
            
            # 2. Average the Metrics (Smoothing)
            avg_imbalance = (metrics1['imbalance_ratio'] + metrics2['imbalance_ratio']) / 2
            
            # 3. Wall Detection (Use the second, most recent snapshot)
            buy_wall = self._find_wall(np.array(ob2['bids']))
            sell_wall = self._find_wall(np.array(ob2['asks']))

            # 4. Normalize Score (0-100)
            # 50 = Neutral
            # >60 = Bullish Pressure
            # <40 = Bearish Pressure
            score = 50.0
            if avg_imbalance > 1.5: score = 75.0
            if avg_imbalance > 2.5: score = 90.0
            if avg_imbalance < 0.66: score = 25.0
            if avg_imbalance < 0.4: score = 10.0
            
            # Penalty for spoofing
            if is_spoofing:
                score = 50.0 # Revert to neutral if data is unreliable
                # self.logger.debug(f"⚠️ Spoofing detected on {symbol}")

            return {
                'imbalance_score': score,
                'buy_pressure': metrics2['bid_vol'],
                'sell_pressure': metrics2['ask_vol'],
                'nearest_buy_wall': buy_wall,
                'nearest_sell_wall': sell_wall,
                'is_spoofing': is_spoofing
            }

        except Exception as e:
            self.logger.warning(f"OrderFlow Error {symbol}: {e}")
            return self._default_result()

    def _process_book(self, ob: Dict) -> Dict:
        """Calculates weighted volume metrics."""
        bids = np.array(ob['bids'])
        asks = np.array(ob['asks'])
        
        if len(bids) == 0 or len(asks) == 0:
            return {'bid_vol': 0, 'ask_vol': 0, 'imbalance_ratio': 1.0, 'total_liq': 1.0}

        # Weighted Depth: Top 20 orders matter more than orders 21-100
        # Simple summation of top 20
        bid_vol = np.sum(bids[:20, 1])
        ask_vol = np.sum(asks[:20, 1])
        
        total = bid_vol + ask_vol
        ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
        
        return {
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'imbalance_ratio': ratio,
            'total_liq': total
        }

    def _find_wall(self, orders: np.ndarray) -> float:
        """Finds price where volume is 5x the local average."""
        if len(orders) < 20: return 0.0
        
        avg_size = np.mean(orders[:20, 1])
        threshold = avg_size * 5.0
        
        for price, vol in orders:
            if vol > threshold:
                return float(price)
        return 0.0

    def _default_result(self):
        return {
            'imbalance_score': 50.0,
            'buy_pressure': 0.0,
            'sell_pressure': 0.0,
            'nearest_buy_wall': 0.0,
            'nearest_sell_wall': 0.0,
            'is_spoofing': False
        }