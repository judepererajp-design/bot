"""
TITAN-X MARKET GRAPH ENGINE (INSTITUTIONAL v2)
------------------------------------------------------------------------------
Analyzes Sector Correlations and Beta.
- Includes BTC and ETH baselines.
- Robust Math (No division by zero errors).
- Detects 'Altcoin Season' behavior (High ETH corr, Low BTC corr).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

class CorrelationAnalyzer:
    def __init__(self, api_interface):
        self.api = api_interface
        self.logger = logging.getLogger("MarketGraph")
        self.btc_cache = None
        self.eth_cache = None

    async def update_macro_data(self):
        """
        Fetches BTC and ETH data to use as baselines.
        Call this once per hour in the Maintenance Loop.
        """
        try:
            # Fetch 4H data for macro trend (Longer horizon)
            batch = await self.api.fetch_ohlcv_batch("BTC/USDT", ["4h"])
            batch_eth = await self.api.fetch_ohlcv_batch("ETH/USDT", ["4h"])
            
            if batch and "4h" in batch:
                self.btc_cache = batch["4h"]['close']
                
            if batch_eth and "4h" in batch_eth:
                self.eth_cache = batch_eth["4h"]['close']
                
            self.logger.info("✅ Macro Baselines (BTC/ETH) Updated")
        except Exception as e:
            self.logger.error(f"Macro Update Failed: {e}")

    async def analyze(self, symbol: str, target_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Beta and Correlation against BTC and ETH.
        """
        # Safety Check
        if self.btc_cache is None or target_df is None:
            return self._neutral_result()

        try:
            # 1. Align Data Series
            target_series = target_df['close']
            btc_series = self.btc_cache
            eth_series = self.eth_cache
            
            min_len = min(len(target_series), len(btc_series))
            # Need at least 20 points for valid correlation
            if min_len < 20: 
                return self._neutral_result()

            # Slice to matching length (Recent data)
            target_s = target_series.iloc[-min_len:].reset_index(drop=True)
            btc_s = btc_series.iloc[-min_len:].reset_index(drop=True)
            
            # 2. Calculate Correlation (Numpy handles math better than pandas here)
            with np.errstate(all='ignore'): # Suppress divide by zero warnings
                # Calculate Returns first
                ret_target = target_s.pct_change().fillna(0)
                ret_btc = btc_s.pct_change().fillna(0)
                
                # BTC Correlation
                corr_btc = np.corrcoef(ret_target, ret_btc)[0, 1]
                if np.isnan(corr_btc): corr_btc = 0.0
                
                # ETH Correlation (Optional Logic)
                corr_eth = 0.0
                if eth_series is not None:
                    eth_s = eth_series.iloc[-min_len:].reset_index(drop=True)
                    ret_eth = eth_s.pct_change().fillna(0)
                    corr_eth = np.corrcoef(ret_target, ret_eth)[0, 1]
                    if np.isnan(corr_eth): corr_eth = 0.0

                # Beta (Volatility relative to BTC)
                # Beta = Covariance / Variance
                cov_matrix = np.cov(ret_target, ret_btc)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0

            # 3. Institutional Classification
            status = "NEUTRAL"
            score = 50.0
            
            # SCENARIO A: Decoupled Alpha (The Holy Grail)
            # Low correlation to BTC, but high internal momentum
            if abs(corr_btc) < 0.4:
                status = "DECOUPLED"
                score = 85.0 # Premium trade condition
                
            # SCENARIO B: Altcoin Follower
            # Correlated to ETH more than BTC
            elif corr_eth > corr_btc and corr_eth > 0.7:
                status = "ETH_BETA"
                score = 65.0
                
            # SCENARIO C: High Beta Risk
            # Moves with BTC but 3x faster (Dangerous)
            elif beta > 2.5:
                status = "HIGH_BETA"
                score = 40.0 # Reduce position size advised
                
            # SCENARIO D: Market Follower
            else:
                status = "COUPLED"
                score = 60.0

            return {
                'correlation_score': score,
                'btc_corr': float(corr_btc),
                'eth_corr': float(corr_eth),
                'beta': float(beta),
                'status': status
            }

        except Exception:
            # Fail gracefully on math errors
            return self._neutral_result()

    def _neutral_result(self):
        return {
            'correlation_score': 50.0, 
            'beta': 1.0, 
            'status': 'NEUTRAL',
            'btc_corr': 0.0,
            'eth_corr': 0.0
        }