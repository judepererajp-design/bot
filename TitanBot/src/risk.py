"""
TITAN-X RISK MANAGER (OPTIMIZED + SCALING)
------------------------------------------------------------------------------
Integrates with Position Optimizer AND Profit Optimizer.
Advanced risk calculation with multiple stop loss methodologies.
"""

import pandas as pd
import numpy as np
import logging
import math
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Try to import optimizers
try:
    from .position_optimizer import PositionOptimizer
    from .profit_optimizer import ProfitOptimizer
except ImportError:
    # Fallback for direct execution
    PositionOptimizer = None
    ProfitOptimizer = None

class RiskCalculator:
    def __init__(self, config: Dict[str, Any], 
                 optimizer: PositionOptimizer = None,
                 profit_optimizer: ProfitOptimizer = None):
        
        self.config = config
        self.logger = logging.getLogger("RiskManager")
        self.optimizer = optimizer
        self.profit_optimizer = profit_optimizer
        
        # Base Settings
        self.account_size = float(self.config.get('account_size', 10000.0))
        self.min_rr = float(self.config.get('min_rr', 1.3))
        self.max_rr = float(self.config.get('max_rr', 5.0))
        self.atr_period = int(self.config.get('atr_period', 14))
        self.atr_multiplier_sl = float(self.config.get('atr_multiplier_sl', 1.5))
        self.atr_multiplier_tp = float(self.config.get('atr_multiplier_tp', 2.5))
        
        # Advanced Settings
        self.use_kelly = bool(self.config.get('use_kelly', True))
        self.use_adaptive_stops = bool(self.config.get('use_adaptive_stops', True))
        self.max_position_size_pct = float(self.config.get('max_position_size_pct', 0.1))  # 10% max
        self.max_daily_loss_pct = float(self.config.get('max_daily_loss_pct', 0.02))  # 2% daily max
        self.correlation_adjustment = bool(self.config.get('correlation_adjustment', True))
        
        # Stop Loss Methodologies
        self.stop_methods = {
            'atr': self._calculate_atr_stop,
            'structure': self._calculate_structure_stop,
            'volatility': self._calculate_volatility_stop,
            'percent': self._calculate_percent_stop,
            'multi_timeframe': self._calculate_mtf_stop
        }
        
        # Risk Profiles for different strategies
        self.risk_profiles = {
            'default': {
                'sl_method': 'atr',
                'sl_multiplier': 1.5,
                'tp_multiplier': 2.5,
                'max_loss_pct': 0.01,  # 1% max loss per trade
                'confidence_multiplier': 1.0
            },
            'Institutional Breakout': {
                'sl_method': 'structure',
                'sl_multiplier': 1.2,
                'tp_multiplier': 3.0,
                'max_loss_pct': 0.012,
                'confidence_multiplier': 1.2
            },
            'Institutional Reversal': {
                'sl_method': 'atr',
                'sl_multiplier': 1.3,
                'tp_multiplier': 2.0,
                'max_loss_pct': 0.008,
                'confidence_multiplier': 1.1
            },
            'Smart Money': {
                'sl_method': 'structure',
                'sl_multiplier': 1.0,
                'tp_multiplier': 2.5,
                'max_loss_pct': 0.01,
                'confidence_multiplier': 1.15
            },
            'Price Action': {
                'sl_method': 'structure',
                'sl_multiplier': 1.4,
                'tp_multiplier': 2.2,
                'max_loss_pct': 0.009,
                'confidence_multiplier': 1.05
            },
            'Mean Reversion': {
                'sl_method': 'volatility',
                'sl_multiplier': 1.6,
                'tp_multiplier': 1.8,
                'max_loss_pct': 0.007,
                'confidence_multiplier': 0.9
            }
        }
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()

    def calculate_risk(self, signal: Dict[str, Any], mtf_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Comprehensive risk calculation with multiple methodologies.
        Returns enhanced signal with risk parameters or None if invalid.
        """
        try:
            # Reset daily tracking if new day
            self._reset_daily_tracking()
            
            # Extract signal data
            symbol = signal['symbol']
            tf = signal['timeframe']
            direction = signal['direction']
            strategy = signal.get('strategy', 'default')
            
            # Get entry price (handle both naming conventions)
            entry = float(signal.get('entry', signal.get('entry_price', 0)))
            if entry <= 0:
                self.logger.error(f"Invalid entry price for {symbol}: {entry}")
                return None
            
            # Get confidence
            confidence = float(signal.get('confidence', 50.0))
            
            # Get appropriate risk profile
            profile = self.risk_profiles.get(strategy, self.risk_profiles['default'])
            
            # ====================================================
            # STEP 1: STOP LOSS CALCULATION (Multiple Methods)
            # ====================================================
            stop_loss, stop_method, stop_metadata = self._calculate_stop_loss(
                entry=entry,
                direction=direction,
                profile=profile,
                mtf_data=mtf_data,
                signal=signal,
                tf=tf
            )
            
            # Validate stop loss
            if stop_loss <= 0:
                self.logger.warning(f"Invalid stop loss for {symbol}")
                return None
            
            # Ensure stop loss is valid for direction
            if direction == 'LONG' and stop_loss >= entry:
                stop_loss = entry * 0.995  # Force 0.5% below entry
            elif direction == 'SHORT' and stop_loss <= entry:
                stop_loss = entry * 1.005  # Force 0.5% above entry
            
            risk_distance = abs(entry - stop_loss)
            
            # ====================================================
            # STEP 2: TAKE PROFIT CALCULATION (Scaling)
            # ====================================================
            scaling_plan = {}
            if self.profit_optimizer:
                # Get volume profile from signal
                vp_data = signal.get('volume_profile', {})
                
                scaling_plan = self.profit_optimizer.calculate_targets(
                    entry=entry,
                    stop_loss=stop_loss,
                    direction=direction,
                    atr=self._calculate_atr_value(mtf_data, entry),
                    volume_profile=vp_data
                )
            
            # Determine final take profit
            if scaling_plan and 'tp3' in scaling_plan:
                # Use the runner target (tp3) as main TP for R:R calculation
                final_tp = scaling_plan['tp3']
            else:
                # Fallback to fixed multiplier
                tp_multiplier = profile['tp_multiplier']
                if direction == 'LONG':
                    final_tp = entry + (risk_distance * tp_multiplier)
                else:
                    final_tp = entry - (risk_distance * tp_multiplier)
            
            # Cap TP based on volatility
            max_tp_distance = self._calculate_max_tp_distance(mtf_data, entry)
            current_tp_distance = abs(final_tp - entry)
            
            if current_tp_distance > max_tp_distance:
                adjustment = max_tp_distance / current_tp_distance
                if direction == 'LONG':
                    final_tp = entry + ((final_tp - entry) * adjustment)
                else:
                    final_tp = entry - ((entry - final_tp) * adjustment)
            
            # ====================================================
            # STEP 3: RISK-REWARD VALIDATION
            # ====================================================
            reward_distance = abs(final_tp - entry)
            
            if risk_distance == 0:
                self.logger.error(f"Zero risk distance for {symbol}")
                return None
            
            risk_reward = reward_distance / risk_distance
            
            # Validate R:R
            if risk_reward < self.min_rr:
                self.logger.debug(f"Rejected {symbol}: R:R {risk_reward:.2f} < {self.min_rr}")
                return None
            
            if risk_reward > self.max_rr:
                self.logger.debug(f"Adjusting {symbol}: R:R {risk_reward:.2f} > {self.max_rr}")
                # Adjust TP to maintain max R:R
                if direction == 'LONG':
                    final_tp = entry + (risk_distance * self.max_rr)
                else:
                    final_tp = entry - (risk_distance * self.max_rr)
                reward_distance = abs(final_tp - entry)
                risk_reward = self.max_rr
            
            # ====================================================
            # STEP 4: POSITION SIZING (Kelly Criterion + Adjustments)
            # ====================================================
            position_size_data = self._calculate_position_size(
                entry=entry,
                stop_loss=stop_loss,
                confidence=confidence,
                profile=profile,
                strategy=strategy,
                symbol=symbol,
                mtf_data=mtf_data
            )
            
            if position_size_data is None:
                return None
            
            # ====================================================
            # STEP 5: DAILY RISK LIMITS
            # ====================================================
            trade_risk_amount = position_size_data['risk_amount']
            daily_risk_limit = self.account_size * self.max_daily_loss_pct
            
            if self.daily_pnl - trade_risk_amount < -daily_risk_limit:
                self.logger.warning(f"Daily risk limit reached for {symbol}")
                return None
            
            # ====================================================
            # STEP 6: FINALIZE SIGNAL
            # ====================================================
            # Update signal with risk parameters (both naming conventions)
            signal['entry'] = entry
            signal['entry_price'] = entry
            signal['stop'] = stop_loss
            signal['stop_loss'] = stop_loss
            signal['tp'] = final_tp
            signal['take_profit'] = final_tp
            
            # Add comprehensive risk plan
            signal['plan'] = {
                # Risk amounts
                'risk_amount': round(position_size_data['risk_amount'], 2),
                'position_size': float(position_size_data['position_size']),
                'risk_percent': round(position_size_data['risk_percent'], 2),
                
                # Risk metrics
                'risk_reward_ratio': round(risk_reward, 2),
                'risk_distance': float(risk_distance),
                'reward_distance': float(reward_distance),
                
                # Methodology
                'stop_method': stop_method,
                'stop_metadata': stop_metadata,
                'sizing_method': position_size_data['method'],
                
                # Advanced metrics
                'atr_value': float(self._calculate_atr_value(mtf_data, entry)),
                'volatility_pct': float(self._calculate_volatility_pct(mtf_data, entry)),
                'kelly_fraction': float(position_size_data.get('kelly_fraction', 0.0)),
                'confidence_adjustment': float(position_size_data.get('confidence_multiplier', 1.0)),
                
                # Scaling plan
                'scaling_plan': scaling_plan,
                
                # Limits
                'max_position_size': float(self.account_size * self.max_position_size_pct),
                'daily_risk_remaining': float(daily_risk_limit + self.daily_pnl)
            }
            
            # Add correlation adjustment if enabled
            if self.correlation_adjustment:
                corr_adjustment = self._calculate_correlation_adjustment(symbol, mtf_data)
                signal['plan']['correlation_adjustment'] = corr_adjustment
                signal['plan']['position_size'] *= corr_adjustment.get('size_multiplier', 1.0)
            
            # Log risk calculation
            self.logger.info(
                f"💰 Risk for {symbol}: Size ${position_size_data['risk_amount']:.2f} "
                f"({position_size_data['risk_percent']:.2f}%), R:R {risk_reward:.2f}"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Risk calculation error for {signal.get('symbol', 'unknown')}: {e}")
            return None

    def _calculate_stop_loss(self, entry: float, direction: str, profile: Dict, 
                           mtf_data: Dict, signal: Dict, tf: str) -> Tuple[float, str, Dict]:
        """
        Calculate stop loss using multiple methodologies and select best one.
        """
        stop_candidates = []
        
        # Try each stop method
        for method_name, method_func in self.stop_methods.items():
            try:
                stop, metadata = method_func(
                    entry=entry,
                    direction=direction,
                    profile=profile,
                    mtf_data=mtf_data,
                    signal=signal,
                    tf=tf
                )
                
                if stop > 0:
                    # Calculate quality score for this stop
                    quality_score = self._evaluate_stop_quality(
                        stop=stop,
                        entry=entry,
                        direction=direction,
                        method=method_name,
                        metadata=metadata,
                        mtf_data=mtf_data
                    )
                    
                    stop_candidates.append({
                        'stop': stop,
                        'method': method_name,
                        'metadata': metadata,
                        'quality': quality_score,
                        'distance_pct': abs(stop - entry) / entry
                    })
                    
            except Exception as e:
                self.logger.debug(f"Stop method {method_name} failed: {e}")
                continue
        
        # Select best stop loss
        if not stop_candidates:
            # Fallback to ATR stop
            atr_value = self._calculate_atr_value(mtf_data, entry)
            if direction == 'LONG':
                stop = entry - (atr_value * profile['sl_multiplier'])
            else:
                stop = entry + (atr_value * profile['sl_multiplier'])
            
            return stop, 'atr_fallback', {'atr_value': atr_value, 'multiplier': profile['sl_multiplier']}
        
        # Sort by quality score (descending) then distance (ascending for tighter stops)
        stop_candidates.sort(key=lambda x: (-x['quality'], x['distance_pct']))
        
        best_stop = stop_candidates[0]
        
        # Apply confidence adjustment
        confidence = signal.get('confidence', 50.0)
        confidence_multiplier = 1.0 - ((confidence - 50.0) / 100.0) * 0.2  # ±10% adjustment
        
        if direction == 'LONG':
            adjusted_stop = best_stop['stop'] * confidence_multiplier
            if adjusted_stop >= entry:
                adjusted_stop = entry * 0.995
        else:
            adjusted_stop = best_stop['stop'] / confidence_multiplier
            if adjusted_stop <= entry:
                adjusted_stop = entry * 1.005
        
        return adjusted_stop, best_stop['method'], best_stop['metadata']

    def _calculate_atr_stop(self, entry: float, direction: str, profile: Dict,
                          mtf_data: Dict, **kwargs) -> Tuple[float, Dict]:
        """ATR-based stop loss."""
        atr_value = self._calculate_atr_value(mtf_data, entry)
        multiplier = profile.get('sl_multiplier', self.atr_multiplier_sl)
        
        if direction == 'LONG':
            stop = entry - (atr_value * multiplier)
        else:
            stop = entry + (atr_value * multiplier)
        
        return stop, {'atr_value': atr_value, 'multiplier': multiplier}

    def _calculate_structure_stop(self, entry: float, direction: str, profile: Dict,
                                signal: Dict, **kwargs) -> Tuple[float, Dict]:
        """Structure-based stop loss (swing highs/lows)."""
        technical_stop = signal.get('technical_stop')
        if technical_stop and float(technical_stop) > 0:
            stop = float(technical_stop)
            
            # Add small buffer
            buffer = stop * 0.005
            if direction == 'LONG':
                stop -= buffer
            else:
                stop += buffer
            
            return stop, {'source': 'technical_stop', 'buffer_pct': 0.5}
        
        # Fallback to ATR if no technical stop
        return self._calculate_atr_stop(entry, direction, profile, kwargs.get('mtf_data', {}))

    def _calculate_volatility_stop(self, entry: float, direction: str, profile: Dict,
                                 mtf_data: Dict, **kwargs) -> Tuple[float, Dict]:
        """Volatility-based stop using standard deviation."""
        df = mtf_data.get(kwargs.get('tf', '1h'))
        if df is None or len(df) < 20:
            return self._calculate_atr_stop(entry, direction, profile, mtf_data)
        
        # Calculate rolling standard deviation
        returns = df['close'].pct_change().dropna()
        if len(returns) < 10:
            return self._calculate_atr_stop(entry, direction, profile, mtf_data)
        
        volatility = returns.std()
        multiplier = 2.0  # 2 standard deviations
        
        if direction == 'LONG':
            stop = entry * (1 - (volatility * multiplier))
        else:
            stop = entry * (1 + (volatility * multiplier))
        
        return stop, {'volatility': volatility, 'multiplier': multiplier, 'std_devs': 2}

    def _calculate_percent_stop(self, entry: float, direction: str, profile: Dict,
                              **kwargs) -> Tuple[float, Dict]:
        """Percentage-based stop loss."""
        percent_stop = profile.get('max_loss_pct', 0.01)
        
        if direction == 'LONG':
            stop = entry * (1 - percent_stop)
        else:
            stop = entry * (1 + percent_stop)
        
        return stop, {'percent': percent_stop * 100}

    def _calculate_mtf_stop(self, entry: float, direction: str, profile: Dict,
                          mtf_data: Dict, tf: str, **kwargs) -> Tuple[float, Dict]:
        """Multi-timeframe stop loss using higher timeframe structure."""
        # Get higher timeframe data
        higher_tf = self._get_higher_timeframe(tf)
        df_higher = mtf_data.get(higher_tf)
        
        if df_higher is None or len(df_higher) < 20:
            return self._calculate_atr_stop(entry, direction, profile, mtf_data)
        
        # Find recent swing points on higher timeframe
        if direction == 'LONG':
            # Look for recent swing low
            recent_lows = df_higher['low'].rolling(10).min().dropna()
            if len(recent_lows) > 0:
                swing_low = recent_lows.iloc[-1]
                stop = swing_low * 0.995  # 0.5% buffer below swing low
                return stop, {'method': 'mtf_swing_low', 'swing_price': swing_low, 'timeframe': higher_tf}
        else:
            # Look for recent swing high
            recent_highs = df_higher['high'].rolling(10).max().dropna()
            if len(recent_highs) > 0:
                swing_high = recent_highs.iloc[-1]
                stop = swing_high * 1.005  # 0.5% buffer above swing high
                return stop, {'method': 'mtf_swing_high', 'swing_price': swing_high, 'timeframe': higher_tf}
        
        # Fallback
        return self._calculate_atr_stop(entry, direction, profile, mtf_data)

    def _evaluate_stop_quality(self, stop: float, entry: float, direction: str,
                             method: str, metadata: Dict, mtf_data: Dict) -> float:
        """Evaluate the quality of a stop loss candidate."""
        score = 50.0  # Base score
        
        # 1. Distance score (20-80% of entry price range is ideal)
        distance_pct = abs(stop - entry) / entry
        if 0.02 <= distance_pct <= 0.08:  # 2-8% is optimal
            score += 30.0
        elif 0.01 <= distance_pct <= 0.10:  # 1-10% is acceptable
            score += 15.0
        else:
            score -= 20.0
        
        # 2. Method preference
        method_scores = {
            'structure': 20.0,
            'mtf_stop': 15.0,
            'volatility': 10.0,
            'atr': 5.0,
            'percent': 0.0
        }
        score += method_scores.get(method, 0.0)
        
        # 3. Volatility alignment
        atr_value = self._calculate_atr_value(mtf_data, entry)
        if atr_value > 0:
            actual_distance = abs(stop - entry)
            atr_multiple = actual_distance / atr_value
            
            if 1.0 <= atr_multiple <= 2.5:  # 1-2.5 ATR is optimal
                score += 20.0
            elif 0.5 <= atr_multiple <= 3.0:  # 0.5-3 ATR is acceptable
                score += 10.0
        
        # 4. Historical test (check if stop avoids recent price action)
        df = mtf_data.get('1h')
        if df is not None and len(df) >= 20:
            recent_data = df.iloc[-20:]
            
            if direction == 'LONG':
                recent_low = recent_data['low'].min()
                if stop < recent_low * 0.99:  # Stop below recent lows
                    score += 15.0
            else:
                recent_high = recent_data['high'].max()
                if stop > recent_high * 1.01:  # Stop above recent highs
                    score += 15.0
        
        return max(0.0, min(100.0, score))

    def _calculate_position_size(self, entry: float, stop_loss: float, 
                               confidence: float, profile: Dict,
                               strategy: str, symbol: str,
                               mtf_data: Dict) -> Optional[Dict]:
        """Calculate position size using multiple methodologies."""
        try:
            risk_distance = abs(entry - stop_loss)
            if risk_distance == 0:
                return None
            
            # Base risk percentage
            base_risk_pct = profile.get('max_loss_pct', 0.01)
            
            # Confidence adjustment
            confidence_multiplier = 1.0 + ((confidence - 50.0) / 50.0) * 0.5  # ±50% adjustment
            confidence_multiplier = max(0.5, min(2.0, confidence_multiplier))
            
            # Strategy multiplier
            strategy_multiplier = profile.get('confidence_multiplier', 1.0)
            
            # Volatility adjustment
            volatility_pct = self._calculate_volatility_pct(mtf_data, entry)
            volatility_multiplier = 1.0
            if volatility_pct > 0.05:  # High volatility
                volatility_multiplier = 0.7
            elif volatility_pct < 0.02:  # Low volatility
                volatility_multiplier = 1.3
            
            # Combined risk percentage
            adjusted_risk_pct = base_risk_pct * confidence_multiplier * strategy_multiplier * volatility_multiplier
            
            # Apply Kelly Criterion if enabled
            if self.use_kelly and self.optimizer:
                kelly_result = self.optimizer.calculate_position_size(
                    signal={'strategy': strategy, 'confidence': confidence},
                    account_size=self.account_size,
                    regime_data={},
                    current_portfolio=[],
                    stop_loss_distance=risk_distance
                )
                
                if kelly_result:
                    kelly_risk_pct = kelly_result.get('risk_percent', adjusted_risk_pct) / 100.0
                    # Use half-kelly for safety
                    kelly_risk_pct = kelly_risk_pct * 0.5
                    
                    # Blend with base risk (70% kelly, 30% base)
                    blended_risk_pct = (kelly_risk_pct * 0.7) + (adjusted_risk_pct * 0.3)
                    adjusted_risk_pct = blended_risk_pct
                    
                    kelly_fraction = kelly_result.get('kelly_fraction', 0.0)
                else:
                    kelly_fraction = 0.0
            else:
                kelly_fraction = 0.0
            
            # Apply hard limits
            adjusted_risk_pct = max(0.0025, min(self.max_position_size_pct, adjusted_risk_pct))
            
            # Calculate dollar amounts
            risk_amount = self.account_size * adjusted_risk_pct
            position_size = risk_amount / risk_distance
            
            # Apply maximum position size limit
            max_position_value = self.account_size * self.max_position_size_pct
            current_position_value = position_size * entry
            
            if current_position_value > max_position_value:
                position_size = max_position_value / entry
                risk_amount = position_size * risk_distance
                adjusted_risk_pct = risk_amount / self.account_size
            
            return {
                'risk_amount': round(risk_amount, 2),
                'position_size': float(position_size),
                'risk_percent': round(adjusted_risk_pct * 100, 4),
                'method': 'kelly_blended' if self.use_kelly else 'fixed_percentage',
                'confidence_multiplier': confidence_multiplier,
                'volatility_multiplier': volatility_multiplier,
                'kelly_fraction': kelly_fraction,
                'base_risk_pct': base_risk_pct * 100
            }
            
        except Exception as e:
            self.logger.error(f"Position sizing error: {e}")
            # Fallback to simple 1% risk
            risk_amount = self.account_size * 0.01
            position_size = risk_amount / risk_distance if risk_distance > 0 else 0
            
            return {
                'risk_amount': round(risk_amount, 2),
                'position_size': float(position_size),
                'risk_percent': 1.0,
                'method': 'fallback_fixed',
                'confidence_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'kelly_fraction': 0.0,
                'base_risk_pct': 1.0
            }

    def _calculate_atr_value(self, mtf_data: Dict, current_price: float) -> float:
        """Calculate ATR value from multi-timeframe data."""
        try:
            # Try 1h data first
            df = mtf_data.get('1h')
            if df is None:
                # Try any available dataframe
                for tf, data in mtf_data.items():
                    if data is not None and len(data) >= 20:
                        df = data
                        break
            
            if df is None or len(df) < 20:
                return current_price * 0.02  # 2% fallback
            
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.atr_period).mean().iloc[-1]
            
            if pd.isna(atr):
                return current_price * 0.02
            
            return float(atr)
            
        except Exception:
            return current_price * 0.02

    def _calculate_volatility_pct(self, mtf_data: Dict, current_price: float) -> float:
        """Calculate volatility as percentage of price."""
        atr_value = self._calculate_atr_value(mtf_data, current_price)
        if current_price > 0:
            return atr_value / current_price
        return 0.02  # 2% default

    def _calculate_max_tp_distance(self, mtf_data: Dict, entry: float) -> float:
        """Calculate maximum reasonable take profit distance."""
        atr_value = self._calculate_atr_value(mtf_data, entry)
        return atr_value * 6.0  # Max 6 ATRs for TP

    def _calculate_correlation_adjustment(self, symbol: str, mtf_data: Dict) -> Dict:
        """Calculate correlation-based position size adjustment."""
        # Simplified implementation
        # In full version, this would check correlation to BTC/ETH
        return {
            'size_multiplier': 1.0,
            'correlation_score': 50.0,
            'recommendation': 'NEUTRAL'
        }

    def _get_higher_timeframe(self, tf: str) -> str:
        """Map timeframe to higher timeframe."""
        mapping = {
            '15m': '1h',
            '1h': '4h',
            '4h': '1d',
            '1d': '1w'
        }
        return mapping.get(tf, '4h')

    def _reset_daily_tracking(self):
        """Reset daily PNL tracking if new day."""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today

    def record_trade_outcome(self, pnl: float):
        """Record trade outcome for daily tracking."""
        self.daily_pnl += pnl
        self.daily_trades += 1

    def can_trade(self) -> bool:
        """Check if trading is allowed based on daily limits."""
        daily_limit = self.account_size * self.max_daily_loss_pct
        return self.daily_pnl > -daily_limit