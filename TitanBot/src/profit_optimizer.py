"""
TITAN-X PROFIT OPTIMIZER (DYNAMIC SCALING)
------------------------------------------------------------------------------
Calculates multiple Take Profit levels based on:
1. Volatility (ATR Multiples)
2. Structural Levels (Volume Profile, VWAP)
3. Fibonacci Extensions
"""

import logging
from typing import Dict, Any, List

class ProfitOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ProfitOptimizer")

    def calculate_targets(self, 
                          entry: float, 
                          stop_loss: float, 
                          direction: str, 
                          atr: float,
                          volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a scaling plan with 3 Take Profit levels.
        """
        try:
            risk_dist = abs(entry - stop_loss)
            if risk_dist == 0: return {}

            targets = []

            # 1. BASE TARGETS (Volatility Based)
            # ----------------------------------
            # TP1: 1.5R (Bank some profit)
            # TP2: 2.5R (Standard target)
            # TP3: 4.0R (Runner)
            multipliers = [1.5, 2.5, 4.0]
            
            for m in multipliers:
                price = entry + (risk_dist * m) if direction == 'LONG' else entry - (risk_dist * m)
                targets.append({'price': price, 'type': f'ATR {m}R', 'weight': 30})

            # 2. STRUCTURAL TARGETS (Volume Profile)
            # --------------------------------------
            # FIX: Only add if volume_profile has valid data
            vp_levels = []
            if volume_profile and isinstance(volume_profile, dict):
                # Extract VWAP bands if available
                vwap_upper = volume_profile.get('vwap_upper', 0)
                vwap_lower = volume_profile.get('vwap_lower', 0)
                vah = volume_profile.get('vah', 0)
                val = volume_profile.get('val', 0)

                if direction == 'LONG':
                    if vwap_upper > entry and vwap_upper > 0:
                        vp_levels.append(('VWAP Upper', vwap_upper))
                    if vah > entry and vah > 0:
                        vp_levels.append(('VA High', vah))
                else:
                    if vwap_lower > 0 and vwap_lower < entry:
                        vp_levels.append(('VWAP Lower', vwap_lower))
                    if val > 0 and val < entry:
                        vp_levels.append(('VA Low', val))

            # Filter VP levels: Must be > 1R distance to be worth targeting
            for name, price in vp_levels:
                dist = abs(price - entry)
                if dist > risk_dist:
                    targets.append({'price': price, 'type': name, 'weight': 50})  # Higher weight for structural levels

            # 3. SORT AND SELECT BEST 3
            # -------------------------
            if direction == 'LONG':
                targets.sort(key=lambda x: x['price'])
            else:
                targets.sort(key=lambda x: x['price'], reverse=True)

            # We want to distribute targets across the range. 
            # Pick one "Safe" (near 1.5R), one "Structural" (mid), one "Moon" (far)
            
            final_targets = []
            
            # Safe Target (TP1) - Always the first ATR target
            if targets:
                final_targets.append(targets[0]) 
            
            # Mid Target (TP2) - Try to find a structural level, else take the middle ATR
            mid_found = False
            if len(targets) > 1:
                for t in targets[1:]:
                    if 'ATR' not in t['type'] and t.get('weight', 0) >= 50:  # It's a structural level
                        final_targets.append(t)
                        mid_found = True
                        break
            if not mid_found and len(targets) > 1:
                final_targets.append(targets[1] if len(targets) > 1 else targets[0])
                
            # Moon Target (TP3) - Always the last target
            if len(targets) > 2:
                final_targets.append(targets[-1])

            # Ensure we have 3, fill if missing
            while len(final_targets) < 3:
                # Add a simple multiplier
                last_price = final_targets[-1]['price'] if final_targets else entry
                extension = risk_dist * 1.5
                new_price = last_price + extension if direction == 'LONG' else last_price - extension
                final_targets.append({'price': new_price, 'type': 'Extension', 'weight': 20})

            # 4. CONSTRUCT PLAN
            # -----------------
            if len(final_targets) >= 3:
                return {
                    'tp1': final_targets[0]['price'],
                    'tp2': final_targets[1]['price'],
                    'tp3': final_targets[2]['price'],
                    'desc_1': final_targets[0]['type'],
                    'desc_2': final_targets[1]['type'],
                    'desc_3': final_targets[2]['type'],
                    'close_1': 30, # Close 30%
                    'close_2': 30, # Close 30%
                    'close_3': 40  # Close 40% (or Trail)
                }
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Target Calc Error: {e}")
            return {}
