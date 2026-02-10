import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

class SmartMoneyStrategy:
    """
    TITAN-X SMC MODULE V4 (INSTITUTIONAL LAUNCHPAD - PRODUCTION)
    ------------------------------------------------------------
    Features:
    1. Trend Strength Filter with EMA alignment
    2. Launchpad Scoring System (0-100)
    3. Volume Profile Analysis (Smart Money detection)
    4. Breakout Confirmation
    5. Dynamic Risk Management
    6. Multi-timeframe confluence
    """
    
    def __init__(self, context: Dict):
        self.config = context.get('config', {})
        self.min_fvg_size = self.config.get('min_fvg_size', 0.003)
        self.launch_threshold = self.config.get('launch_threshold', 50)
        self.logger = logging.getLogger("SMC_Strategy")
        
        # Risk parameters
        self.risk_config = {
            'max_position_risk': 0.02,  # 2% max risk per trade
            'launch_risk_multiplier': 0.8,
            'strong_launch_multiplier': 1.0,
            'weak_launch_multiplier': 0.6,
        }
        
    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """
        Main analysis entry point.
        Returns trading signal if conditions met.
        """
        # 1. Get macro context from highest timeframe
        df_4h = data.get('4h')
        macro_strength = 0
        market_regime = 'NEUTRAL'
        
        if df_4h is not None and len(df_4h) > 200:
            macro_strength = self._get_trend_strength(df_4h)
            market_regime = self._get_market_regime(df_4h)
        
        # 2. Scan each timeframe for signals
        for tf in ['4h', '1h', '15m']:
            df = data.get(tf)
            if df is None or len(df) < 200: 
                continue
            
            curr_price = df['close'].iloc[-1]
            local_strength = self._get_trend_strength(df)
            
            # 3. Calculate Launch Setup Score
            launch_analysis = self._analyze_launch_setup(df)
            is_launch_ready = launch_analysis['is_ready']
            launch_score = launch_analysis['score']
            
            # 4. Check for breakout confirmation
            breakout_data = self._detect_breakout(df, launch_score)
            
            # 5. Dynamic Permission Logic
            permissions = self._calculate_permissions(
                macro_strength=macro_strength,
                local_strength=local_strength,
                launch_analysis=launch_analysis,
                breakout_data=breakout_data,
                market_regime=market_regime
            )
            
            # 6. Pattern Scanning with permissions
            signal = self._scan_patterns_with_permissions(
                symbol=symbol,
                df=df,
                tf=tf,
                curr_price=curr_price,
                permissions=permissions,
                launch_analysis=launch_analysis,
                breakout_data=breakout_data
            )
            
            if signal:
                return signal
        
        return None
    
    def _get_trend_strength(self, df: pd.DataFrame) -> int:
        """
        Calculate trend strength from -100 (bearish) to +100 (bullish).
        Uses EMA alignment and slope for scoring.
        """
        try:
            close = df['close']
            price = close.iloc[-1]
            
            # Calculate EMAs
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
            
            score = 0
            
            # 1. Price position relative to EMAs (60 points max)
            if price > ema20: score += 20
            elif price < ema20: score -= 20
            
            if price > ema50: score += 20
            elif price < ema50: score -= 20
            
            if price > ema200: score += 20
            elif price < ema200: score -= 20
            
            # 2. EMA alignment bonus (40 points max)
            # Perfect bullish alignment: Price > EMA20 > EMA50 > EMA200
            if price > ema20 > ema50 > ema200:
                score += 40
            # Perfect bearish alignment: Price < EMA20 < EMA50 < EMA200
            elif price < ema20 < ema50 < ema200:
                score -= 40
            
            # 3. Slope confirmation (recent momentum)
            price_slope = self._calculate_slope(close.iloc[-20:])
            if abs(price_slope) > 0.001:  # Significant slope
                if price_slope > 0 and score > 0:
                    score += 10
                elif price_slope < 0 and score < 0:
                    score -= 10
            
            # Clamp to [-100, 100]
            return max(-100, min(100, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0
    
    def _analyze_launch_setup(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive launch setup analysis with weighted scoring.
        Returns detailed analysis including score breakdown.
        """
        try:
            components = {}
            score = 0
            max_possible = 0
            
            # 1. VOLATILITY COMPRESSION (Coiling) - 30 points
            compression_score = self._calculate_compression_score(df)
            score += compression_score
            max_possible += 30
            components['compression'] = compression_score
            
            # 2. VOLUME ACCUMULATION - 30 points
            accumulation_score = self._calculate_accumulation_score(df)
            score += accumulation_score
            max_possible += 30
            components['accumulation'] = accumulation_score
            
            # 3. SUPPORT/RESISTANCE STRUCTURE - 20 points
            structure_score = self._calculate_structure_score(df)
            score += structure_score
            max_possible += 20
            components['structure'] = structure_score
            
            # 4. MOMENTUM PREPARATION - 20 points
            momentum_score = self._calculate_momentum_score(df)
            score += momentum_score
            max_possible += 20
            components['momentum'] = momentum_score
            
            # Calculate percentage score
            percentage = (score / max_possible * 100) if max_possible > 0 else 0
            
            # Dynamic threshold based on market volatility
            atr_pct = self._get_atr_percentage(df)
            threshold = self.launch_threshold
            if atr_pct > 0.03:  # High volatility
                threshold += 10  # Require stronger signals
            elif atr_pct < 0.01:  # Low volatility
                threshold -= 5   # Lower threshold
            
            return {
                'is_ready': score >= threshold,
                'score': score,
                'percentage': percentage,
                'threshold': threshold,
                'components': components,
                'atr_pct': atr_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing launch setup: {e}")
            return {'is_ready': False, 'score': 0, 'percentage': 0, 'components': {}}
    
    def _calculate_compression_score(self, df: pd.DataFrame) -> int:
        """Calculate Bollinger Band compression score (0-30)."""
        try:
            close = df['close']
            period = 20
            
            # Calculate Bollinger Bands
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # Bandwidth as percentage of price
            bandwidth = (upper_band - lower_band) / sma
            current_bw = bandwidth.iloc[-1]
            
            # Compare to recent history
            lookback = min(100, len(bandwidth))
            recent_bw = bandwidth.iloc[-lookback:]
            
            # Score based on percentile
            if current_bw <= recent_bw.quantile(0.2):  # Bottom 20%
                return 30
            elif current_bw <= recent_bw.quantile(0.4):  # Bottom 40%
                return 20
            elif current_bw <= recent_bw.quantile(0.6):  # Bottom 60%
                return 10
            else:
                return 0
                
        except:
            return 0
    
    def _calculate_accumulation_score(self, df: pd.DataFrame) -> int:
        """Calculate volume accumulation score (0-30)."""
        try:
            recent = df.iloc[-20:]  # Last 20 candles
            
            # Separate up and down candles
            up_candles = recent[recent['close'] > recent['open']]
            down_candles = recent[recent['close'] < recent['open']]
            
            if len(up_candles) == 0 or len(down_candles) == 0:
                return 10  # Neutral score for one-sided markets
            
            # Calculate volume ratios
            up_volume = up_candles['volume'].sum()
            down_volume = down_candles['volume'].sum()
            volume_ratio = up_volume / down_volume if down_volume > 0 else 10
            
            # Score based on buying pressure
            if volume_ratio > 2.0:  # Strong accumulation
                return 30
            elif volume_ratio > 1.5:  # Moderate accumulation
                return 20
            elif volume_ratio > 1.2:  # Mild accumulation
                return 10
            elif volume_ratio < 0.8:  # Distribution
                return -10
            else:
                return 5  # Neutral
                
        except:
            return 0
    
    def _calculate_structure_score(self, df: pd.DataFrame) -> int:
        """Calculate support/resistance structure score (0-20)."""
        try:
            recent = df.iloc[-30:]
            current_price = df['close'].iloc[-1]
            
            # Find nearest support and resistance
            support_levels = self._find_support_levels(recent)
            resistance_levels = self._find_resistance_levels(recent)
            
            score = 0
            
            # Score for being near support but not breaking it
            if len(support_levels) > 0:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                support_distance = abs(current_price - nearest_support) / current_price
                
                if support_distance < 0.01:  # Within 1% of support
                    # Check if price is rejecting support (closing above lows)
                    last_candle = df.iloc[-1]
                    candle_body = (last_candle['close'] - last_candle['open']) / last_candle['open']
                    if candle_body > 0:  # Bullish candle at support
                        score += 10
            
            # Score for compression near resistance (potential breakout)
            if len(resistance_levels) > 0:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                resistance_distance = abs(current_price - nearest_resistance) / current_price
                
                if resistance_distance < 0.02:  # Within 2% of resistance
                    score += 10
            
            return min(20, score)
            
        except:
            return 0
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> int:
        """Calculate momentum preparation score (0-20)."""
        try:
            # Check RSI for oversold/overbought conditions
            rsi = self._calculate_rsi(df['close'], period=14)
            current_rsi = rsi.iloc[-1]
            
            score = 0
            
            # In launch scenarios, we want RSI to be recovering from oversold
            if 30 <= current_rsi <= 50:  # Oversold recovery zone
                rsi_slope = self._calculate_slope(rsi.iloc[-5:])
                if rsi_slope > 0:  # RSI rising
                    score += 10
            
            # Check for hidden bullish divergence
            if self._check_hidden_bullish_divergence(df):
                score += 10
            
            return score
            
        except:
            return 0
    
    def _detect_breakout(self, df: pd.DataFrame, launch_score: int) -> Dict:
        """Detect breakout from compression."""
        if launch_score < 30:  # Not enough compression
            return {'is_breaking': False, 'direction': None, 'strength': 0}
        
        try:
            # Look at recent consolidation range
            recent = df.iloc[-10]  # Last 10 candles (excluding current)
            current = df.iloc[-1]
            
            consolidation_high = recent['high'].max()
            consolidation_low = recent['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Breakout conditions
            breaking_up = (
                current['close'] > consolidation_high and
                current['close'] > current['open'] and
                current['volume'] > recent['volume'].mean() * 1.2
            )
            
            breaking_down = (
                current['close'] < consolidation_low and
                current['close'] < current['open'] and
                current['volume'] > recent['volume'].mean() * 1.2
            )
            
            if breaking_up:
                # Calculate breakout strength
                breakout_distance = (current['close'] - consolidation_high) / consolidation_range
                volume_ratio = current['volume'] / recent['volume'].mean()
                strength = min(100, 50 + (breakout_distance * 100) + (volume_ratio * 10))
                
                return {
                    'is_breaking': True,
                    'direction': 'UP',
                    'strength': strength,
                    'breakout_level': consolidation_high
                }
            elif breaking_down:
                return {
                    'is_breaking': True,
                    'direction': 'DOWN',
                    'strength': 60,
                    'breakout_level': consolidation_low
                }
            
            return {'is_breaking': False, 'direction': None, 'strength': 0}
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout: {e}")
            return {'is_breaking': False, 'direction': None, 'strength': 0}
    
    def _calculate_permissions(self, macro_strength: int, local_strength: int, 
                              launch_analysis: Dict, breakout_data: Dict,
                              market_regime: str) -> Dict:
        """Calculate trading permissions based on all factors."""
        permissions = {
            'allow_long': True,
            'allow_short': True,
            'risk_multiplier': 1.0,
            'confidence_boost': 0,
            'notes': []
        }
        
        launch_score = launch_analysis.get('score', 0)
        is_launch_ready = launch_analysis.get('is_ready', False)
        
        # 1. Basic trend filter
        if macro_strength > 40 or local_strength > 60:
            permissions['allow_short'] = False
            permissions['notes'].append("Strong uptrend - no shorts")
        
        if macro_strength < -40 or local_strength < -60:
            permissions['allow_long'] = False
            permissions['notes'].append("Strong downtrend - no longs")
        
        # 2. Launch override logic
        if is_launch_ready:
            # Adjust permissions based on launch strength
            if launch_score >= 70:
                # Strong launch signal - aggressive override
                permissions['allow_long'] = True
                permissions['allow_short'] = False
                permissions['risk_multiplier'] = self.risk_config['strong_launch_multiplier']
                permissions['confidence_boost'] = 15
                permissions['notes'].append(f"Strong launch detected (score: {launch_score})")
                
            elif launch_score >= 50:
                # Moderate launch signal - conservative override
                if macro_strength < 20:  # Only override in neutral/weak trends
                    permissions['allow_long'] = True
                    permissions['risk_multiplier'] = self.risk_config['launch_risk_multiplier']
                    permissions['confidence_boost'] = 10
                    permissions['notes'].append(f"Launch detected (score: {launch_score})")
                    
            elif launch_score >= 40:
                # Weak launch signal - very conservative
                permissions['risk_multiplier'] = self.risk_config['weak_launch_multiplier']
                permissions['confidence_boost'] = 5
                permissions['notes'].append(f"Early launch setup (score: {launch_score})")
        
        # 3. Breakout confirmation boost
        if breakout_data.get('is_breaking') and breakout_data.get('direction') == 'UP':
            permissions['confidence_boost'] += 10
            permissions['notes'].append(f"Breakout confirmed (strength: {breakout_data.get('strength', 0)})")
        
        # 4. Market regime adjustment
        if market_regime == 'BEARISH':
            permissions['risk_multiplier'] *= 0.7
            permissions['notes'].append("Bearish market - reducing risk")
        elif market_regime == 'BULLISH':
            permissions['confidence_boost'] += 5
            permissions['notes'].append("Bullish market - boosting confidence")
        
        return permissions
    
    def _scan_patterns_with_permissions(self, symbol: str, df: pd.DataFrame, tf: str,
                                       curr_price: float, permissions: Dict,
                                       launch_analysis: Dict, breakout_data: Dict) -> Optional[Dict]:
        """Scan for patterns with permission filtering."""
        
        # 1. FVG Retrace
        fvg_signal = self._scan_for_fvg_retrace(df, curr_price)
        if fvg_signal:
            direction = fvg_signal['dir']
            if ((direction == 'LONG' and permissions['allow_long']) or
                (direction == 'SHORT' and permissions['allow_short'])):
                return self._build_signal(
                    symbol=symbol,
                    pattern=fvg_signal,
                    current=df.iloc[-1],
                    timeframe=tf,
                    launch_analysis=launch_analysis,
                    breakout_data=breakout_data,
                    permissions=permissions
                )
        
        # 2. Order Block Retest
        ob_signal = self._scan_for_ob_retest(df, curr_price)
        if ob_signal:
            direction = ob_signal['dir']
            if ((direction == 'LONG' and permissions['allow_long']) or
                (direction == 'SHORT' and permissions['allow_short'])):
                return self._build_signal(
                    symbol=symbol,
                    pattern=ob_signal,
                    current=df.iloc[-1],
                    timeframe=tf,
                    launch_analysis=launch_analysis,
                    breakout_data=breakout_data,
                    permissions=permissions
                )
        
        # 3. Liquidity Sweep
        sfp_signal = self._check_sfp_live(df)
        if sfp_signal:
            direction = sfp_signal['dir']
            if ((direction == 'LONG' and permissions['allow_long']) or
                (direction == 'SHORT' and permissions['allow_short'])):
                return self._build_signal(
                    symbol=symbol,
                    pattern=sfp_signal,
                    current=df.iloc[-1],
                    timeframe=tf,
                    launch_analysis=launch_analysis,
                    breakout_data=breakout_data,
                    permissions=permissions
                )
        
        return None
    
    def _scan_for_fvg_retrace(self, df, curr_price):
        """Scan for Fair Value Gap retracement patterns."""
        for i in range(3, 15):
            # Bullish FVG: Candle 1 high < Candle 3 low
            c1_high = df['high'].iloc[-i-2]
            c3_low = df['low'].iloc[-i]
            
            if c3_low > c1_high: 
                gap_size = (c3_low - c1_high) / df['close'].iloc[-i-1]
                if gap_size > self.min_fvg_size:
                    if curr_price < c3_low and curr_price > c1_high:
                        if df['close'].iloc[-1] > df['open'].iloc[-1]:
                            return {
                                'name': 'Bullish FVG Retrace',
                                'dir': 'LONG',
                                'conf': 75,
                                'stop_level': c1_high * 0.995
                            }
            
            # Bearish FVG: Candle 1 low > Candle 3 high
            c1_low = df['low'].iloc[-i-2]
            c3_high = df['high'].iloc[-i]
            
            if c3_high < c1_low: 
                gap_size = (c1_low - c3_high) / df['close'].iloc[-i-1]
                if gap_size > self.min_fvg_size:
                    if curr_price > c3_high and curr_price < c1_low:
                        if df['close'].iloc[-1] < df['open'].iloc[-1]:
                            return {
                                'name': 'Bearish FVG Retrace',
                                'dir': 'SHORT',
                                'conf': 75,
                                'stop_level': c1_low * 1.005
                            }
        return None
    
    def _scan_for_ob_retest(self, df, curr_price):
        """Scan for Order Block retest patterns."""
        if len(df) < 55:
            return None
            
        # Bullish OB (Demand Zone)
        swing_low = df['low'].iloc[-50:-5].min()
        if abs(curr_price - swing_low) / swing_low < 0.005:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                return {
                    'name': 'Bullish OB Retest',
                    'dir': 'LONG',
                    'conf': 70,
                    'stop_level': swing_low * 0.998
                }
                
        # Bearish OB (Supply Zone)
        swing_high = df['high'].iloc[-50:-5].max()
        if abs(curr_price - swing_high) / swing_high < 0.005:
            if df['close'].iloc[-1] < df['open'].iloc[-1]:
                return {
                    'name': 'Bearish OB Retest',
                    'dir': 'SHORT',
                    'conf': 70,
                    'stop_level': swing_high * 1.002
                }
        return None
    
    def _check_sfp_live(self, df):
        """Check for Stop Hunt/Liquidity Sweep patterns."""
        if len(df) < 21:
            return None
            
        curr = df.iloc[-1]
        liq_high = df['high'].iloc[-20:-1].max()
        liq_low = df['low'].iloc[-20:-1].min()
        
        # Bearish liquidity sweep
        if curr['high'] > liq_high and curr['close'] < liq_high:
            return {
                'name': 'Bearish Liquidity Sweep',
                'dir': 'SHORT',
                'conf': 80,
                'stop_level': curr['high'] * 1.001
            }
            
        # Bullish liquidity sweep
        if curr['low'] < liq_low and curr['close'] > liq_low:
            return {
                'name': 'Bullish Liquidity Sweep',
                'dir': 'LONG',
                'conf': 80,
                'stop_level': curr['low'] * 0.999
            }
            
        return None
    
    def _build_signal(self, symbol: str, pattern: Dict, current: pd.Series,
                     timeframe: str, launch_analysis: Dict, breakout_data: Dict,
                     permissions: Dict) -> Dict:
        """Build complete trading signal with metadata."""
        
        # Base confidence
        confidence = pattern['conf']
        
        # Apply confidence boosts
        confidence += permissions['confidence_boost']
        
        # Risk multiplier
        risk_multiplier = permissions['risk_multiplier']
        
        # Pattern name enhancement
        pattern_name = pattern['name']
        is_launch = launch_analysis.get('is_ready', False)
        launch_score = launch_analysis.get('score', 0)
        
        if is_launch:
            if launch_score >= 70:
                pattern_name = f"🚀 {pattern_name} (Strong Launch)"
            elif launch_score >= 50:
                pattern_name = f"🚀 {pattern_name} (Launchpad)"
            else:
                pattern_name = f"⚡ {pattern_name} (Early Setup)"
        
        if breakout_data.get('is_breaking') and breakout_data.get('direction') == 'UP':
            pattern_name = f"💥 {pattern_name} (Breakout)"
        
        # Calculate dynamic stop loss
        stop_loss = self._calculate_dynamic_stop(
            pattern['stop_level'],
            pattern['dir'],
            is_launch,
            breakout_data.get('is_breaking', False)
        )
        
        # Build signal
        signal = {
            'strategy': 'Smart Money V4',
            'pattern_name': pattern_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': pattern['dir'],
            'entry_price': current['close'],
            'entry': current['close'],
            'confidence': min(95, confidence),  # Cap at 95%
            'technical_stop': stop_loss,
            'timestamp': current.get('timestamp', pd.Timestamp.now()).timestamp(),
            'risk_multiplier': risk_multiplier,
            'metadata': {
                'launch_score': launch_score,
                'launch_ready': is_launch,
                'breakout_strength': breakout_data.get('strength', 0),
                'breakout_confirmed': breakout_data.get('is_breaking', False),
                'trend_strength': self._get_trend_strength(pd.DataFrame([current])),
                'permission_notes': permissions.get('notes', [])
            }
        }
        
        # Calculate targets
        targets = self._calculate_targets(
            entry=current['close'],
            stop=stop_loss,
            direction=pattern['dir'],
            is_launch=is_launch
        )
        signal.update(targets)
        
        return signal
    
    def _calculate_dynamic_stop(self, base_stop: float, direction: str,
                               is_launch: bool, is_breakout: bool) -> float:
        """Calculate dynamic stop loss based on trade context."""
        if direction == 'LONG':
            if is_launch and not is_breakout:
                # Launch trades get wider stops (more room for volatility)
                return base_stop * 0.985  # 1.5% below
            elif is_breakout:
                # Breakout trades get tighter stops (clear invalidation)
                return base_stop * 0.995  # 0.5% below
            else:
                return base_stop  # Standard stop
        else:  # SHORT
            if is_launch:
                return base_stop * 1.015  # 1.5% above
            else:
                return base_stop
    
    def _calculate_targets(self, entry: float, stop: float, 
                          direction: str, is_launch: bool) -> Dict:
        """Calculate profit targets based on trade type."""
        if direction == 'LONG':
            stop_distance = entry - stop
            risk = stop_distance / entry
            
            # Different R:R based on trade type
            if is_launch:
                # Launch trades aim for bigger moves
                targets = [
                    entry + (stop_distance * 3),  # 3R
                    entry + (stop_distance * 5),  # 5R
                    entry + (stop_distance * 8)   # 8R for strong launches
                ]
            else:
                # Standard trades
                targets = [
                    entry + (stop_distance * 2),  # 2R
                    entry + (stop_distance * 3),  # 3R
                    entry + (stop_distance * 4)   # 4R
                ]
            
            return {
                'targets': targets,
                'risk_reward': 3.0 if is_launch else 2.0
            }
        else:  # SHORT
            stop_distance = stop - entry
            risk = stop_distance / entry
            
            targets = [
                entry - (stop_distance * 2),
                entry - (stop_distance * 3),
                entry - (stop_distance * 4)
            ]
            
            return {
                'targets': targets,
                'risk_reward': 2.0
            }
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate linear regression slope of a series."""
        if len(series) < 2:
            return 0
        try:
            x = np.arange(len(series))
            y = series.values
            slope, _ = np.polyfit(x, y, 1)
            return slope / np.mean(y)  # Normalized slope
        except:
            return 0
    
    def _get_atr_percentage(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            atr = true_range.rolling(window=period).mean().iloc[-1]
            return atr / df['close'].iloc[-1]
        except:
            return 0.02  # Default 2%
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _check_hidden_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Check for hidden bullish RSI divergence."""
        try:
            if len(df) < 30:
                return False
            
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'], period=14)
            
            # Find recent lows in price and RSI
            price_lows = []
            rsi_lows = []
            
            for i in range(3, 20):
                if (df['low'].iloc[-i] < df['low'].iloc[-i-1] and 
                    df['low'].iloc[-i] < df['low'].iloc[-i+1]):
                    price_lows.append((i, df['low'].iloc[-i]))
                
                if (rsi.iloc[-i] < rsi.iloc[-i-1] and 
                    rsi.iloc[-i] < rsi.iloc[-i+1]):
                    rsi_lows.append((i, rsi.iloc[-i]))
            
            # Check for divergence (price making lower lows, RSI making higher lows)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                price_trend = price_lows[0][1] < price_lows[1][1]  # Lower low
                rsi_trend = rsi_lows[0][1] > rsi_lows[1][1]  # Higher low
                
                if price_trend and rsi_trend:
                    return True
            
            return False
        except:
            return False
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Find recent support levels."""
        try:
            # Simple swing low detection
            support_levels = []
            for i in range(10, len(df) - 5):
                if (df['low'].iloc[i] < df['low'].iloc[i-5:i].min() and
                    df['low'].iloc[i] < df['low'].iloc[i+1:i+6].min()):
                    support_levels.append(df['low'].iloc[i])
            
            return sorted(list(set(support_levels)))
        except:
            return []
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Find recent resistance levels."""
        try:
            # Simple swing high detection
            resistance_levels = []
            for i in range(10, len(df) - 5):
                if (df['high'].iloc[i] > df['high'].iloc[i-5:i].max() and
                    df['high'].iloc[i] > df['high'].iloc[i+1:i+6].max()):
                    resistance_levels.append(df['high'].iloc[i])
            
            return sorted(list(set(resistance_levels)))
        except:
            return []
    
    def _get_market_regime(self, df: pd.DataFrame) -> str:
        """Determine overall market regime."""
        try:
            trend_strength = self._get_trend_strength(df)
            
            if trend_strength > 40:
                return 'BULLISH'
            elif trend_strength < -40:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'
    
    # ==================== CONFIGURATION ====================
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for this strategy."""
        return {
            'min_fvg_size': 0.003,
            'launch_threshold': 50,
            'max_position_risk': 0.02,
            'risk_multipliers': {
                'launch': 0.8,
                'strong_launch': 1.0,
                'weak_launch': 0.6
            },
            'enabled_timeframes': ['4h', '1h', '15m'],
            'min_data_length': 200
        }