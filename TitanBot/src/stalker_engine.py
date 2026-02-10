"""
TITAN-X DYNAMIC STALKER ENGINE (ENHANCED)
------------------------------------------------------------------------------
Continuously monitors active crypto coins for PRE-SIGNAL setups.
- Filters out stablecoins, forex, non-crypto assets
- Focuses on real cryptocurrencies with volume
- Advanced setup detection with category awareness
"""

import asyncio
import logging
import time
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class StalkedCoin:
    symbol: str
    category: str
    added_at: float
    last_scan: float
    setup_score: int
    setup_type: str
    key_level: float
    timeframe: str
    urgency: str
    notes: List[str]
    volume_rank: int
    detection_confidence: float

class StalkerEngine:
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.logger = logging.getLogger("Stalker")
        self.stalked_coins: Dict[str, StalkedCoin] = {}
        
        # Config
        self.config = context.get('config', {}).get('stalker', {})
        self.max_watchlist = self.config.get('max_watchlist_size', 20)
        self.min_score = self.config.get('min_score', 40)
        
        # Throttle
        self.scan_interval = self.config.get('rescan_interval', 300)  # 5 minutes
        
        # Rate limiting
        self.fetch_semaphore = asyncio.Semaphore(3)
        
        # Filtering configuration
        self.exclude_patterns = [
            r'.*UP/USDT$',
            r'.*DOWN/USDT$',
            r'.*BULL/USDT$',
            r'.*BEAR/USDT$',
            r'^[A-Z]{3}/[A-Z]{3}$',  # Forex pairs
            r'USD[0-9]*/USDT$',
            r'^FDUSD/USDT$',
            r'^EUR/USDT$', r'^GBP/USDT$', r'^JPY/USDT$',
            r'^PAXG/USDT$',
        ]
        
        # Category priority (higher = more attention)
        self.category_priority = {
            'LAYER1': 1.2,
            'ETHEREUM': 1.3,
            'BITCOIN': 1.4,
            'LAYER2': 1.1,
            'DEFI': 1.0,
            'AI': 1.15,
            'GAMING': 0.9,
            'MEME': 0.7,  # Lower priority for meme coins
            'ORACLES': 1.0,
            'STORAGE': 0.9,
            'PRIVACY': 0.8,
            'RWA': 0.85,
            'EXCHANGE': 0.95,
            'LSD': 1.0,
            'BRIDGES': 0.9,
            'LARGE_CAP': 1.1,
            'ALTCAP': 0.8
        }
        
        # Setup type weights
        self.setup_weights = {
            'coiling': 65,
            'resistance_test': 75,
            'support_test': 75,
            'volume_anomaly': 60,
            'breakout_forming': 80,
            'wedge_forming': 70,
            'triangle_forming': 70,
            'divergence': 85
        }
        
        # Performance tracking
        self.scans_completed = 0
        self.setups_found = 0
        
    def _is_valid_crypto(self, symbol: str) -> bool:
        """Check if symbol is a valid cryptocurrency (not stablecoin/forex)."""
        symbol_upper = symbol.upper()
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, symbol_upper):
                return False
        
        # Additional manual exclusions
        invalid_bases = ['USD1', 'FDUSD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 
                        'CHF', 'PAXG', 'XAUT', 'BULL', 'BEAR', 'LONG', 'SHORT']
        base_symbol = symbol_upper.replace('/USDT', '')
        
        if base_symbol in invalid_bases:
            return False
        
        return True
    
    def _get_symbol_category(self, symbol: str) -> str:
        """Get category for symbol using scanner or fallback."""
        scanner = self.context.get('scanner')
        if scanner and hasattr(scanner, 'universe') and symbol in scanner.universe:
            return scanner.universe[symbol].category
        
        # Fallback categorization
        symbol_upper = symbol.upper().replace('/USDT', '')
        
        # Quick category check
        category_patterns = {
            'BITCOIN': ['BTC'],
            'ETHEREUM': ['ETH'],
            'LAYER1': ['SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'ATOM', 'NEAR', 'ALGO'],
            'LAYER2': ['ARB', 'OP', 'IMX', 'METIS'],
            'DEFI': ['UNI', 'AAVE', 'COMP', 'MKR'],
            'MEME': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK'],
            'AI': ['AGIX', 'FET', 'OCEAN', 'RNDR'],
            'GAMING': ['AXS', 'SAND', 'MANA', 'GALA'],
        }
        
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in symbol_upper:
                    return category
        
        return 'ALTCAP'
    
    def notify_signal_generated(self, symbol: str):
        """
        Called when a signal is generated for a symbol.
        Removes it from watchlist since we now have a signal.
        """
        if symbol in self.stalked_coins:
            del self.stalked_coins[symbol]
            self.logger.debug(f"Removed {symbol} from watchlist (signal generated)")
        
    async def run(self):
        """Main background loop."""
        self.logger.info("🕵️‍♂️ Stalker Engine Activated - Hunting for crypto setups...")
        while True:
            try:
                await self._scan_cycle()
                await asyncio.sleep(self.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stalker Cycle Error: {e}")
                await asyncio.sleep(60)

    async def _scan_cycle(self):
        """Main scanning cycle."""
        # 1. Get Universe from Scanner
        scanner = self.context.get('scanner')
        if not scanner or not scanner.universe: 
            await asyncio.sleep(60)
            return
        
        # Copy keys to avoid iteration issues
        symbols = list(scanner.universe.keys())
        
        for symbol in symbols:
            # SKIP NON-CRYPTO ASSETS
            if not self._is_valid_crypto(symbol):
                continue
            
            # Skip if already watching (we update existing ones differently)
            if symbol in self.stalked_coins: 
                await self._update_existing_stalk(symbol)
                continue
            
            # Use semaphore to limit concurrent API calls
            async with self.fetch_semaphore:
                data = await self._fetch_light_data(symbol)
                if not data: continue
                
                # Detect Setups
                setups = self._detect_setups(symbol, data)
                
                if setups:
                    self._process_new_setup(symbol, setups)
                
                # Small delay even with semaphore for extra safety
                await asyncio.sleep(0.2)

        # Cleanup old watchlist items
        self._cleanup_watchlist()
        
        self.scans_completed += 1
        if self.scans_completed % 10 == 0:
            self._log_performance()

    async def _update_existing_stalk(self, symbol: str):
        """Update an existing stalked coin."""
        if symbol not in self.stalked_coins:
            return
        
        stalk = self.stalked_coins[symbol]
        
        # Skip if recently updated
        if time.time() - stalk.last_scan < 300:  # 5 minutes
            return
        
        async with self.fetch_semaphore:
            data = await self._fetch_light_data(symbol)
            if not data:
                return
            
            # Check if setup is still valid or has improved
            setups = self._detect_setups(symbol, data)
            if setups:
                best_new = max(setups, key=lambda x: x['score'])
                if best_new['score'] > stalk.setup_score:
                    # Setup improved, update
                    self._process_new_setup(symbol, setups, is_update=True)
                elif best_new['score'] < stalk.setup_score * 0.7:
                    # Setup degraded significantly, consider removing
                    if time.time() - stalk.added_at > 3600:  # After 1 hour
                        del self.stalked_coins[symbol]
                        self.logger.debug(f"Removed {symbol} from watchlist (setup degraded)")
            else:
                # Setup no longer present
                if time.time() - stalk.added_at > 7200:  # After 2 hours
                    del self.stalked_coins[symbol]
                    self.logger.debug(f"Removed {symbol} from watchlist (setup disappeared)")
        
        stalk.last_scan = time.time()

    async def _fetch_light_data(self, symbol):
        """Fetch minimal data for stalker analysis."""
        api = self.context.get('api')
        if not api or not hasattr(api, 'exchange'):
            return None
            
        try:
            data = {}
            
            # Try to get 1h data first
            try:
                ohlcv_1h = await api.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                if ohlcv_1h:
                    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['1h'] = df_1h
                    
                    # For stalker, we only need 4h if 1h looks interesting
                    if self._looks_promising(df_1h):
                        ohlcv_4h = await api.exchange.fetch_ohlcv(symbol, '4h', limit=30)
                        if ohlcv_4h:
                            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            data['4h'] = df_4h
            except Exception as e:
                self.logger.debug(f"Failed to fetch {symbol} 1h: {e}")
            
            return data if data else None
            
        except Exception as e:
            self.logger.debug(f"Stalker fetch failed for {symbol}: {e}")
            return None

    def _looks_promising(self, df: pd.DataFrame) -> bool:
        """Quick filter to avoid unnecessary 4h fetches."""
        if df is None or len(df) < 20:
            return False
            
        # Check for recent volatility compression (coiling)
        recent = df.iloc[-10:]
        high_range = recent['high'].max() - recent['low'].min()
        avg_price = recent['close'].mean()
        
        if avg_price == 0:
            return False
        
        # Calculate average true range
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            current_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-20:-5].mean()
            
            if avg_atr == 0:
                return False
            
            # If ATR is < 60% of average, might be coiling
            is_coiling = current_atr < (avg_atr * 0.6)
            
            # Also check for approaching key levels
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].iloc[-20:-1].max()
            recent_low = df['low'].iloc[-20:-1].min()
            
            near_high = abs(current_price - recent_high) / recent_high < 0.02
            near_low = abs(current_price - recent_low) / recent_low < 0.02
            
            return is_coiling or near_high or near_low
            
        except:
            # Fallback to simple range check
            return (high_range / avg_price) < 0.02

    def _detect_setups(self, symbol, data) -> List[Dict]:
        setups = []
        
        # Get symbol category for scoring adjustment
        category = self._get_symbol_category(symbol)
        category_multiplier = self.category_priority.get(category, 1.0)
        
        for tf, df in data.items():
            if df is None or len(df) < 20: 
                continue
            
            # A. COILING (Volatility Squeeze)
            if self._is_coiling(df):
                base_score = self.setup_weights.get('coiling', 65)
                adjusted_score = min(100, int(base_score * category_multiplier))
                
                setups.append({
                    'type': 'coiling',
                    'tf': tf,
                    'score': adjusted_score,
                    'level': df['close'].iloc[-1],
                    'urgency': 'medium',
                    'note': 'Volatility compression, breakout imminent',
                    'confidence': 0.7
                })

            # B. KEY LEVEL APPROACH
            level_setup = self._check_key_levels(df, category_multiplier)
            if level_setup:
                level_setup['tf'] = tf
                setups.append(level_setup)
                
            # C. VOLUME ANOMALY
            vol_setup = self._check_volume(df, category_multiplier)
            if vol_setup:
                vol_setup['tf'] = tf
                setups.append(vol_setup)
            
            # D. PATTERN FORMATION (additional setups)
            pattern_setup = self._check_patterns(df, category_multiplier)
            if pattern_setup:
                pattern_setup['tf'] = tf
                setups.append(pattern_setup)

        return setups

    def _process_new_setup(self, symbol, setups, is_update=False):
        """Process and add new setup to watchlist."""
        if not setups:
            return
            
        # Pick best setup
        best = max(setups, key=lambda x: x['score'])
        
        if best['score'] < self.min_score: 
            return

        # Get category and volume rank
        category = self._get_symbol_category(symbol)
        scanner = self.context.get('scanner')
        volume_rank = 50  # Default
        
        if scanner and symbol in scanner.universe:
            volume_rank = scanner.universe[symbol].volume_rank
        
        # Calculate final score with category adjustment
        category_multiplier = self.category_priority.get(category, 1.0)
        final_score = min(100, int(best['score'] * category_multiplier))
        
        # Determine urgency based on score and setup type
        urgency = 'low'
        if final_score >= 80:
            urgency = 'high'
        elif final_score >= 65:
            urgency = 'medium'
        
        # Check if we need to make room in watchlist
        if len(self.stalked_coins) >= self.max_watchlist and not is_update:
            # Remove lowest score or oldest coin
            sorted_coins = sorted(self.stalked_coins.items(), 
                                key=lambda x: (x[1].setup_score, x[1].added_at))
            removed_symbol = sorted_coins[0][0]
            del self.stalked_coins[removed_symbol]
            self.logger.debug(f"Removed {removed_symbol} to make room for {symbol}")

        # Add/update to watchlist
        self.stalked_coins[symbol] = StalkedCoin(
            symbol=symbol,
            category=category,
            added_at=time.time(),
            last_scan=time.time(),
            setup_score=final_score,
            setup_type=best['type'],
            key_level=best['level'],
            timeframe=best['tf'],
            urgency=urgency,
            notes=[best['note']],
            volume_rank=volume_rank,
            detection_confidence=best.get('confidence', 0.5)
        )

        # Notify Telegram (only for new setups, not updates)
        if not is_update:
            asyncio.create_task(self._send_alert(symbol, best, category, final_score))
            self.setups_found += 1

    async def _send_alert(self, symbol, setup, category, score):
        """Send alert to Telegram."""
        telegram = self.context.get('telegram')
        if not telegram or not hasattr(telegram, 'app'): 
            return
        
        # Get category emoji
        category_emojis = {
            'LAYER1': '⚡',
            'ETHEREUM': '🔷',
            'BITCOIN': '₿',
            'LAYER2': '🔄',
            'DEFI': '🏦',
            'AI': '🤖',
            'GAMING': '🎮',
            'MEME': '🐸',
            'ORACLES': '🔗',
            'STORAGE': '💾'
        }
        
        category_emoji = category_emojis.get(category, '📊')
        
        urgency_icon = "🚨" if setup['urgency'] == 'high' else "🔍" if setup['urgency'] == 'medium' else "👀"
        
        msg = (
            f"{urgency_icon} <b>WATCHLIST: {symbol}</b> {category_emoji}\n"
            f"🏷️ <b>Category:</b> {category}\n"
            f"👀 <b>Setup:</b> {setup['type'].upper().replace('_', ' ')}\n"
            f"⏰ <b>TF:</b> {setup['tf']} | 📊 <b>Score:</b> {score}\n"
            f"🎯 <b>Level:</b> {setup['level']:.4f}\n"
            f"📈 <b>Confidence:</b> {setup.get('confidence', 0.5)*100:.0f}%\n"
            f"<i>{setup['note']}</i>"
        )
        
        # Add buttons
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            
            # Create tradingview chart URL
            tradingview_symbol = symbol.replace('/', '').replace('USDT', 'USDT')
            
            kb = [[
                InlineKeyboardButton("📊 Chart", url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{tradingview_symbol}"),
                InlineKeyboardButton("📈 Details", callback_data=f"stalker_detail:{symbol}")
            ]]
            
            await telegram.app.bot.send_message(
                chat_id=telegram.chat_id, 
                text=msg, 
                parse_mode='HTML',
                reply_markup=InlineKeyboardMarkup(kb),
                disable_notification=(setup['urgency'] != 'high')
            )
        except Exception as e:
            self.logger.error(f"Failed to send stalker alert: {e}")

    # --- DETECTION LOGIC ---

    def _is_coiling(self, df):
        """Check for volatility compression."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(14).mean()
            
            if len(atr) < 20:
                return False
            
            current_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-20:-5].mean()
            
            if avg_atr == 0:
                return False
            
            # Volatility is 60% of normal or less
            return current_atr < (avg_atr * 0.6)
        except:
            return False

    def _check_key_levels(self, df, category_multiplier=1.0):
        """Check proximity to key support/resistance levels."""
        if len(df) < 21:
            return None
        
        curr = df['close'].iloc[-1]
        high_20 = df['high'].iloc[-20:-1].max()
        low_20 = df['low'].iloc[-20:-1].min()
        
        if high_20 == 0 or low_20 == 0:
            return None
        
        # Check near resistance
        resistance_distance = abs(curr - high_20) / high_20
        if resistance_distance < 0.01:  # Within 1%
            base_score = self.setup_weights.get('resistance_test', 75)
            adjusted_score = min(100, int(base_score * category_multiplier))
            
            return {
                'type': 'resistance_test', 
                'score': adjusted_score, 
                'level': high_20, 
                'urgency': 'high' if resistance_distance < 0.005 else 'medium',
                'note': f'Testing 20-period resistance ({resistance_distance*100:.2f}% away)',
                'confidence': max(0.3, 1.0 - resistance_distance * 10)
            }
        
        # Check near support
        support_distance = abs(curr - low_20) / low_20
        if support_distance < 0.01:  # Within 1%
            base_score = self.setup_weights.get('support_test', 75)
            adjusted_score = min(100, int(base_score * category_multiplier))
            
            return {
                'type': 'support_test', 
                'score': adjusted_score, 
                'level': low_20, 
                'urgency': 'high' if support_distance < 0.005 else 'medium',
                'note': f'Testing 20-period support ({support_distance*100:.2f}% away)',
                'confidence': max(0.3, 1.0 - support_distance * 10)
            }
        
        return None

    def _check_volume(self, df, category_multiplier=1.0):
        """Check for volume anomalies."""
        if len(df) < 21:
            return None
        
        curr_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-20:-1].mean()
        
        if avg_vol == 0:
            return None
        
        volume_ratio = curr_vol / avg_vol
        
        if volume_ratio > 3.0:
            base_score = self.setup_weights.get('volume_anomaly', 60)
            adjusted_score = min(100, int(base_score * category_multiplier))
            
            return {
                'type': 'volume_anomaly', 
                'score': adjusted_score, 
                'level': df['close'].iloc[-1], 
                'urgency': 'high' if volume_ratio > 5.0 else 'medium',
                'note': f'{volume_ratio:.1f}x volume spike ({curr_vol:.0f} vs avg {avg_vol:.0f})',
                'confidence': min(0.9, volume_ratio / 10.0)
            }
        
        return None

    def _check_patterns(self, df, category_multiplier=1.0):
        """Check for emerging patterns."""
        if len(df) < 30:
            return None
        
        # Check for wedge/triangle formation
        highs = df['high'].iloc[-20:]
        lows = df['low'].iloc[-20:]
        
        # Simple wedge detection (converging highs and lows)
        high_slope = (highs.iloc[-1] - highs.iloc[0]) / 20
        low_slope = (lows.iloc[-1] - lows.iloc[0]) / 20
        
        # If highs are decreasing and lows are increasing = symmetrical pattern
        if high_slope < -0.001 and low_slope > 0.001:
            base_score = self.setup_weights.get('wedge_forming', 70)
            adjusted_score = min(100, int(base_score * category_multiplier))
            
            return {
                'type': 'wedge_forming', 
                'score': adjusted_score, 
                'level': df['close'].iloc[-1], 
                'urgency': 'medium',
                'note': 'Symmetrical wedge forming (converging price action)',
                'confidence': 0.6
            }
        
        return None

    def _cleanup_watchlist(self):
        """Remove stale coins from watchlist."""
        now = time.time()
        to_remove = []
        
        for symbol, coin in self.stalked_coins.items():
            # Remove if older than 24 hours
            if now - coin.added_at > 86400:
                to_remove.append(symbol)
            # Remove if not updated in 4 hours
            elif now - coin.last_scan > 14400:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del self.stalked_coins[symbol]
            self.logger.debug(f"Removed {symbol} from watchlist (stale)")

    def get_watchlist_text(self):
        """Get formatted watchlist for display."""
        if not self.stalked_coins: 
            return "📭 Watchlist Empty"
        
        # Sort by urgency and score
        sorted_coins = sorted(
            self.stalked_coins.values(), 
            key=lambda x: (0 if x.urgency == 'high' else 1 if x.urgency == 'medium' else 2, -x.setup_score)
        )
        
        # Category emojis
        category_emojis = {
            'LAYER1': '⚡',
            'ETHEREUM': '🔷',
            'BITCOIN': '₿',
            'LAYER2': '🔄',
            'DEFI': '🏦',
            'AI': '🤖',
            'GAMING': '🎮',
            'MEME': '🐸'
        }
        
        txt = "<b>🕵️‍♂️ ACTIVE CRYPTO WATCHLIST</b>\n"
        txt += "────────────────────────\n"
        
        for coin in sorted_coins[:15]:  # Limit to 15 for readability
            urgency_icon = "🚨" if coin.urgency == 'high' else "🔍" if coin.urgency == 'medium' else "👀"
            category_emoji = category_emojis.get(coin.category, '📊')
            
            txt += (
                f"{urgency_icon} <b>{coin.symbol}</b> {category_emoji} ({coin.timeframe})\n"
                f"   └ {coin.setup_type} @ {coin.key_level:.4f} (Score: {coin.setup_score})\n"
            )
        
        if len(sorted_coins) > 15:
            txt += f"\n... and {len(sorted_coins) - 15} more coins"
            
        return txt

    def _log_performance(self):
        """Log stalker performance."""
        self.logger.info(
            f"Stalker Stats: {len(self.stalked_coins)} active, "
            f"{self.setups_found} setups found, "
            f"{self.scans_completed} scans completed"
        )
        
        # Log category distribution in watchlist
        if self.stalked_coins:
            categories = {}
            for coin in self.stalked_coins.values():
                categories[coin.category] = categories.get(coin.category, 0) + 1
            
            self.logger.debug(f"Watchlist categories: {categories}")

    def get_stats(self) -> Dict[str, Any]:
        """Get stalker statistics."""
        categories = {}
        for coin in self.stalked_coins.values():
            categories[coin.category] = categories.get(coin.category, 0) + 1
        
        return {
            'active_stalks': len(self.stalked_coins),
            'setups_found': self.setups_found,
            'scans_completed': self.scans_completed,
            'category_distribution': categories,
            'avg_setup_score': sum(c.setup_score for c in self.stalked_coins.values()) / len(self.stalked_coins) if self.stalked_coins else 0
        }