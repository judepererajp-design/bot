"""
TITAN-X VOLUME PROFILE & VWAP ANALYZER (OPTIMIZED)
------------------------------------------------------------------------------
High-Performance Liquidity Analysis using Vectorized NumPy operations.
Features:
1. Rolling VWAP with Standard Deviation Bands (Institutional entry zones).
2. Volume Profile with Value Area (VAH/VAL/POC) calculated via Histogram.
3. Dynamic Thresholds (Adapts to volatility, not fixed %).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy.stats import percentileofscore

class VolumeProfileAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VolumeProfile")
        
        # Settings
        self.vwap_window = self.config.get('vwap_window', 24)  # 24 periods (e.g., 1 day on 1h)
        self.profile_bins = self.config.get('profile_bins', 100)  # Higher precision
        self.va_pct = self.config.get('value_area_pct', 0.70)  # 70% Value Area
        
        # Advanced Settings
        self.adaptive_sd_multiplier = self.config.get('adaptive_sd_multiplier', True)
        self.min_samples_for_profile = self.config.get('min_samples_for_profile', 50)
        self.volume_weighted_vwap = self.config.get('volume_weighted_vwap', True)
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs high-speed Volume Analysis with institutional-grade metrics.
        Returns comprehensive volume profile data.
        """
        if df is None or len(df) < self.vwap_window:
            self.logger.warning(f"Insufficient data for volume analysis: {len(df) if df else 0} bars")
            return self._empty_result()

        try:
            # 1. Calculate VWAP & Adaptive Bands
            vwap_data = self._calculate_adaptive_vwap(df)
            current_price = df['close'].iloc[-1]
            
            # 2. Calculate Volume Profile (POC, VAH, VAL, VPOC)
            vp_data = self._calculate_volume_profile(df)
            
            # 3. Calculate Volume Delta (Buy/Sell Pressure)
            volume_delta = self._calculate_volume_delta(df)
            
            # 4. Calculate VWAP Slope (Trend)
            vwap_slope = self._calculate_vwap_slope(df, vwap_data['vwap'])
            
            # 5. Advanced Metrics
            vwap_deviation = self._calculate_vwap_deviation(current_price, vwap_data['vwap'])
            profile_strength = self._assess_profile_strength(vp_data, df)
            
            # 6. Determine Institutional Context
            signal_context = self._determine_institutional_context(
                current_price, vwap_data, vp_data, volume_delta, df
            )
            
            # 7. Calculate Composite Volume Score
            composite_score = self._calculate_composite_volume_score(
                vwap_deviation, volume_delta, profile_strength, vwap_slope
            )
            
            result = {
                # VWAP Metrics
                'vwap': float(vwap_data['vwap']),
                'vwap_upper': float(vwap_data['upper_band']),
                'vwap_lower': float(vwap_data['lower_band']),
                'vwap_slope': float(vwap_slope),
                'vwap_deviation_pct': float(vwap_deviation),
                
                # Volume Profile
                'poc': float(vp_data['poc']),
                'vpoc': float(vp_data['vpoc']),  # Volume Point of Control
                'vah': float(vp_data['vah']),
                'val': float(vp_data['val']),
                'value_area_width_pct': float(vp_data['va_width_pct']),
                'poc_volume_ratio': float(vp_data['poc_volume_ratio']),
                
                # Volume Analysis
                'volume_delta': float(volume_delta['delta']),
                'buy_volume': float(volume_delta['buy_volume']),
                'sell_volume': float(volume_delta['sell_volume']),
                'volume_ratio': float(volume_delta['volume_ratio']),
                'volume_trend': volume_delta['trend'],
                
                # Quality & Strength
                'profile_quality': vp_data['quality'],
                'profile_strength': profile_strength,
                'composite_volume_score': composite_score,
                
                # Signal Context
                'signal_context': signal_context,
                'vwap_zone': self._get_vwap_zone(current_price, vwap_data),
                'value_area_position': self._get_va_position(current_price, vp_data),
                
                # Timestamp
                'timestamp': df['timestamp'].iloc[-1].timestamp() if 'timestamp' in df.columns else pd.Timestamp.now().timestamp()
            }
            
            # Log significant findings
            if abs(vwap_deviation) > 0.03:  # >3% from VWAP
                self.logger.debug(f"Significant VWAP deviation: {vwap_deviation*100:.2f}%")
            
            if vp_data['quality'] == 'HIGH':
                self.logger.debug(f"Strong volume profile detected: POC ratio {vp_data['poc_volume_ratio']:.2f}")
                
            return result

        except Exception as e:
            self.logger.error(f"Volume Analysis Error: {e}", exc_info=True)
            return self._empty_result()

    def _calculate_adaptive_vwap(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates Rolling VWAP with adaptive standard deviation bands.
        Uses volume-weighted standard deviation for more accurate bands.
        """
        window = df.iloc[-self.vwap_window:].copy()
        
        if len(window) < 10:
            return {'vwap': 0.0, 'upper_band': 0.0, 'lower_band': 0.0}
        
        # Calculate Typical Price
        typical_price = (window['high'] + window['low'] + window['close']) / 3
        
        # Calculate VWAP
        if self.volume_weighted_vwap:
            cumulative_vp = (typical_price * window['volume']).cumsum()
            cumulative_vol = window['volume'].cumsum()
            vwap_series = cumulative_vp / cumulative_vol
            current_vwap = vwap_series.iloc[-1]
        else:
            # Simple VWAP (equal weight)
            current_vwap = typical_price.mean()
        
        # Calculate Adaptive Standard Deviation
        if self.adaptive_sd_multiplier:
            # Use rolling volatility to adjust bands
            returns = typical_price.pct_change().dropna()
            if len(returns) > 5:
                volatility = returns.std()
                # Scale multiplier based on volatility (higher vol = wider bands)
                base_multiplier = 2.0
                adaptive_multiplier = base_multiplier * (1 + volatility * 10)
                adaptive_multiplier = min(adaptive_multiplier, 4.0)  # Cap at 4
            else:
                adaptive_multiplier = 2.0
        else:
            adaptive_multiplier = 2.0
        
        # Calculate Volume-Weighted Standard Deviation
        price_diff = typical_price - current_vwap
        weighted_sq_diff = (price_diff ** 2) * window['volume']
        
        if window['volume'].sum() > 0:
            variance = weighted_sq_diff.sum() / window['volume'].sum()
            stdev = np.sqrt(variance)
        else:
            stdev = typical_price.std()
        
        # Calculate Bands
        upper_band = current_vwap + (stdev * adaptive_multiplier)
        lower_band = current_vwap - (stdev * adaptive_multiplier)
        
        return {
            'vwap': float(current_vwap),
            'upper_band': float(upper_band),
            'lower_band': float(lower_band),
            'stdev': float(stdev),
            'multiplier': float(adaptive_multiplier)
        }

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced Volume Profile with histogram binning and statistical analysis.
        """
        # Use recent data (last 100 bars max for relevance)
        lookback = min(100, len(df))
        data = df.iloc[-lookback:].copy()
        
        if len(data) < self.min_samples_for_profile:
            return self._empty_profile_result()
        
        # Get price and volume
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Create histogram
        hist, bin_edges = np.histogram(prices, bins=self.profile_bins, weights=volumes)
        
        # Find POC (Point of Control) - price with highest volume
        poc_idx = hist.argmax()
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
        poc_volume = hist[poc_idx]
        
        # Calculate total volume
        total_volume = hist.sum()
        
        # Find VPOC (Volume-weighted POC)
        if total_volume > 0:
            vpoc = np.average(prices, weights=volumes)
        else:
            vpoc = poc_price
        
        # Calculate Value Area (70% of volume)
        target_volume = total_volume * self.va_pct
        
        # Sort bins by volume descending
        sorted_indices = np.argsort(hist)[::-1]
        sorted_volumes = hist[sorted_indices]
        
        # Cumulative sum to find Value Area
        cum_vol = np.cumsum(sorted_volumes)
        cutoff_idx = np.searchsorted(cum_vol, target_volume)
        
        # Get indices that constitute Value Area
        va_indices = sorted_indices[:cutoff_idx + 1]
        
        # Map back to prices to find VAH and VAL
        va_prices = []
        for idx in va_indices:
            price = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            va_prices.append(price)
        
        if va_prices:
            vah = max(va_prices)
            val = min(va_prices)
        else:
            vah = poc_price
            val = poc_price
        
        # Calculate Value Area width as percentage of price
        if poc_price > 0:
            va_width_pct = (vah - val) / poc_price
        else:
            va_width_pct = 0.0
        
        # Calculate POC strength (percentage of total volume at POC)
        if total_volume > 0:
            poc_volume_ratio = poc_volume / total_volume
        else:
            poc_volume_ratio = 0.0
        
        # Assess profile quality
        if poc_volume_ratio > 0.15:
            quality = "HIGH"
        elif poc_volume_ratio > 0.08:
            quality = "MEDIUM"
        else:
            quality = "LOW"
        
        return {
            'poc': poc_price,
            'vpoc': vpoc,
            'vah': vah,
            'val': val,
            'va_width_pct': va_width_pct,
            'poc_volume_ratio': poc_volume_ratio,
            'total_volume': total_volume,
            'quality': quality,
            'histogram': {
                'bins': bin_edges,
                'volumes': hist
            }
        }

    def _calculate_volume_delta(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Buy/Sell Volume Delta using tick rule approximation.
        """
        if len(df) < 10:
            return {'delta': 0.0, 'buy_volume': 0.0, 'sell_volume': 0.0, 'volume_ratio': 1.0, 'trend': 'NEUTRAL'}
        
        # Simplified tick rule: if close > open => buy volume, else sell volume
        buy_volume = 0.0
        sell_volume = 0.0
        
        for _, row in df.iterrows():
            if row['close'] > row['open']:
                buy_volume += row['volume']
            elif row['close'] < row['open']:
                sell_volume += row['volume']
            else:
                # Equal close/open, split volume
                buy_volume += row['volume'] * 0.5
                sell_volume += row['volume'] * 0.5
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            volume_ratio = buy_volume / total_volume
            delta = (buy_volume - sell_volume) / total_volume
        else:
            volume_ratio = 0.5
            delta = 0.0
        
        # Determine trend
        if volume_ratio > 0.6:
            trend = "STRONG_BUY"
        elif volume_ratio > 0.55:
            trend = "MODERATE_BUY"
        elif volume_ratio < 0.4:
            trend = "STRONG_SELL"
        elif volume_ratio < 0.45:
            trend = "MODERATE_SELL"
        else:
            trend = "NEUTRAL"
        
        return {
            'delta': delta,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'volume_ratio': volume_ratio,
            'trend': trend
        }

    def _calculate_vwap_slope(self, df: pd.DataFrame, current_vwap: float) -> float:
        """Calculates VWAP slope over last 5 periods."""
        if len(df) < 10:
            return 0.0
        
        # Calculate VWAP for previous period
        prev_data = df.iloc[-10:-5]
        if len(prev_data) > 0:
            typical_price_prev = (prev_data['high'] + prev_data['low'] + prev_data['close']) / 3
            if self.volume_weighted_vwap:
                prev_vwap = (typical_price_prev * prev_data['volume']).sum() / prev_data['volume'].sum()
            else:
                prev_vwap = typical_price_prev.mean()
            
            if prev_vwap > 0:
                slope = (current_vwap - prev_vwap) / prev_vwap
                return slope
        return 0.0

    def _calculate_vwap_deviation(self, price: float, vwap: float) -> float:
        """Calculates percentage deviation from VWAP."""
        if vwap > 0:
            return (price - vwap) / vwap
        return 0.0

    def _assess_profile_strength(self, vp_data: Dict[str, Any], df: pd.DataFrame) -> str:
        """Assesses the strength of the volume profile."""
        poc_ratio = vp_data['poc_volume_ratio']
        va_width = vp_data['va_width_pct']
        
        # Narrow VA with high POC concentration = strong profile
        if poc_ratio > 0.15 and va_width < 0.05:
            return "VERY_STRONG"
        elif poc_ratio > 0.10 and va_width < 0.08:
            return "STRONG"
        elif poc_ratio > 0.06:
            return "MODERATE"
        else:
            return "WEAK"

    def _determine_institutional_context(self, price: float, vwap_data: Dict, 
                                        vp_data: Dict, volume_delta: Dict, 
                                        df: pd.DataFrame) -> str:
        """
        Determines institutional context based on multiple volume factors.
        """
        # 1. VWAP Analysis
        vwap = vwap_data['vwap']
        upper_band = vwap_data['upper_band']
        lower_band = vwap_data['lower_band']
        
        # 2. Volume Profile Analysis
        vpoc = vp_data['vpoc']
        vah = vp_data['vah']
        val = vp_data['val']
        poc = vp_data['poc']
        
        # 3. Volume Delta Analysis
        volume_trend = volume_delta['trend']
        
        # Determine primary context
        if price > upper_band:
            if volume_trend in ["STRONG_SELL", "MODERATE_SELL"]:
                return "OVERBOUGHT_VWAP_SELLING"  # Topping with selling pressure
            else:
                return "OVERBOUGHT_VWAP"
        
        elif price < lower_band:
            if volume_trend in ["STRONG_BUY", "MODERATE_BUY"]:
                return "OVERSOLD_VWAP_BUYING"  # Bottoming with buying pressure
            else:
                return "OVERSOLD_VWAP"
        
        # Near VWAP analysis
        elif abs(price - vwap) / vwap < 0.01:  # Within 1% of VWAP
            if abs(price - vpoc) / vpoc < 0.005:  # Very near VPOC
                return "AT_VPOC_EQUILIBRIUM"
            elif price > vpoc:
                return "ABOVE_VPOC_SUPPORT"
            else:
                return "BELOW_VPOC_RESISTANCE"
        
        # Value Area Analysis
        elif price > vah:
            return "ABOVE_VALUE_AREA"
        elif price < val:
            return "BELOW_VALUE_AREA"
        elif abs(price - poc) / poc < 0.005:
            return "AT_POC_CONFLUENCE"
        else:
            return "IN_VALUE_AREA"

    def _calculate_composite_volume_score(self, vwap_deviation: float, 
                                         volume_delta: Dict, 
                                         profile_strength: str,
                                         vwap_slope: float) -> float:
        """Calculates composite volume score (0-100)."""
        score = 50.0  # Neutral baseline
        
        # 1. VWAP Deviation component (20%)
        # Near VWAP is good (score boost), extreme deviations are risky
        deviation_score = 50.0
        abs_dev = abs(vwap_deviation)
        if abs_dev < 0.01:  # Within 1% of VWAP
            deviation_score = 70.0
        elif abs_dev < 0.02:  # Within 2% of VWAP
            deviation_score = 60.0
        elif abs_dev > 0.05:  # Far from VWAP
            deviation_score = 30.0
        
        # 2. Volume Delta component (30%)
        delta = volume_delta['delta']
        delta_score = 50.0 + (delta * 50.0)  # Map -1..1 to 0..100
        
        # 3. Profile Strength component (30%)
        strength_map = {
            "VERY_STRONG": 90.0,
            "STRONG": 75.0,
            "MODERATE": 55.0,
            "WEAK": 35.0
        }
        strength_score = strength_map.get(profile_strength, 50.0)
        
        # 4. VWAP Slope component (20%)
        slope_score = 50.0
        if vwap_slope > 0.001:  # Positive slope
            slope_score = 65.0
        elif vwap_slope < -0.001:  # Negative slope
            slope_score = 35.0
        
        # Weighted average
        composite = (
            deviation_score * 0.2 +
            delta_score * 0.3 +
            strength_score * 0.3 +
            slope_score * 0.2
        )
        
        return max(0.0, min(100.0, composite))

    def _get_vwap_zone(self, price: float, vwap_data: Dict) -> str:
        """Determines which VWAP zone the price is in."""
        vwap = vwap_data['vwap']
        upper = vwap_data['upper_band']
        lower = vwap_data['lower_band']
        
        if price > upper:
            return "UPPER_BAND"
        elif price < lower:
            return "LOWER_BAND"
        elif price > vwap * 1.01:
            return "UPPER_HALF"
        elif price < vwap * 0.99:
            return "LOWER_HALF"
        else:
            return "VWAP_CORE"

    def _get_va_position(self, price: float, vp_data: Dict) -> str:
        """Determines position relative to Value Area."""
        vah = vp_data['vah']
        val = vp_data['val']
        
        if price > vah:
            return "ABOVE_VA"
        elif price < val:
            return "BELOW_VA"
        elif abs(price - vp_data['poc']) / vp_data['poc'] < 0.005:
            return "AT_POC"
        else:
            return "WITHIN_VA"

    def _empty_result(self):
        """Returns empty result structure."""
        return {
            'vwap': 0.0,
            'vwap_upper': 0.0,
            'vwap_lower': 0.0,
            'vwap_slope': 0.0,
            'vwap_deviation_pct': 0.0,
            'poc': 0.0,
            'vpoc': 0.0,
            'vah': 0.0,
            'val': 0.0,
            'value_area_width_pct': 0.0,
            'poc_volume_ratio': 0.0,
            'volume_delta': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'volume_ratio': 0.5,
            'volume_trend': 'NEUTRAL',
            'profile_quality': 'LOW',
            'profile_strength': 'WEAK',
            'composite_volume_score': 50.0,
            'signal_context': 'NEUTRAL',
            'vwap_zone': 'UNKNOWN',
            'value_area_position': 'UNKNOWN',
            'timestamp': 0.0
        }

    def _empty_profile_result(self):
        """Returns empty volume profile result."""
        return {
            'poc': 0.0,
            'vpoc': 0.0,
            'vah': 0.0,
            'val': 0.0,
            'va_width_pct': 0.0,
            'poc_volume_ratio': 0.0,
            'total_volume': 0.0,
            'quality': 'LOW',
            'histogram': {'bins': [], 'volumes': []}
        }