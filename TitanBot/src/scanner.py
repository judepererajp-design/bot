"""
TITAN-X MARKET SCHEDULER (FULL VERSION WITH ENHANCED FILTERING)
------------------------------------------------------------------------------
Intelligent Priority Queue with cryptocurrency categorization.
- Filters out stablecoins, forex, leveraged tokens
- Categorizes coins into sectors (L1, L2, DeFi, Meme, AI, Gaming, etc.)
- Adaptive scanning based on category performance
"""

import time
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict

class ScanTier(Enum):
    HOT = 300      # 5 Minutes (High Priority)
    NORMAL = 900   # 15 Minutes (Standard)
    COLD = 3600    # 60 Minutes (Background)

@dataclass
class TokenState:
    symbol: str
    tier: ScanTier
    category: str = "UNKNOWN"
    last_scan: float = 0.0
    activity_score: int = 0
    consecutive_inactive: int = 0
    last_signal_time: float = 0.0
    scan_count: int = 0
    volume_rank: int = 0
    category_score: float = 1.0  # Score based on category performance
    
    def is_due(self) -> bool:
        """Check if token is due for scanning."""
        elapsed = time.time() - self.last_scan
        return elapsed > self.tier.value
    
    def get_priority_score(self) -> float:
        """Calculate priority score for sorting."""
        base_score = 1000 - self.tier.value  # HOT: 700, NORMAL: 100, COLD: -2600
        
        # Activity bonus
        activity_bonus = self.activity_score * 50
        
        # Time decay - promote if not scanned recently
        time_since_scan = time.time() - self.last_scan
        time_bonus = min(time_since_scan / self.tier.value * 100, 200)
        
        # Inactivity penalty
        inactive_penalty = self.consecutive_inactive * 20
        
        # Category bonus (some categories get priority)
        category_bonus = self._get_category_bonus()
        
        # Volume rank bonus (higher volume = higher priority)
        volume_bonus = max(0, (100 - self.volume_rank) * 2)
        
        return (base_score + activity_bonus + time_bonus + 
                category_bonus + volume_bonus - inactive_penalty)
    
    def _get_category_bonus(self) -> float:
        """Get bonus points based on category."""
        category_bonuses = {
            "LAYER1": 100,
            "ETHEREUM": 90,
            "BITCOIN": 80,
            "LAYER2": 70,
            "DEFI": 60,
            "AI": 75,
            "GAMING": 50,
            "MEME": 30,  # Lower bonus for meme coins
            "ORACLES": 55,
            "STORAGE": 45,
            "PRIVACY": 40,
            "RWA": 35,
            "EXCHANGE": 50,
            "LSD": 45,
            "BRIDGES": 40,
            "LARGE_CAP": 60,
            "ALTCAP": 20
        }
        return category_bonuses.get(self.category, 0) * self.category_score

class MarketScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("Scanner")
        self.universe: Dict[str, TokenState] = {}
        
        # Configuration
        self.tier_distribution = {
            'hot': 0.20,    # Top 20%
            'normal': 0.30, # Next 30%
            'cold': 0.50    # Bottom 50%
        }
        
        # Category management
        self.category_performance: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"signals": 0, "scans": 0, "success_rate": 0.5}
        )
        
        # Performance tracking
        self.total_scans = 0
        self.scans_per_hour = 0
        self.last_stats_update = time.time()
        
        # Batch processing
        self.batch_size = self.config.get('batch_size', 20)
        self.min_interval = self.config.get('min_interval', 1.0)
        
        # Symbol history for pattern detection
        self.symbol_history: Dict[str, deque] = {}
        self.history_length = 50
        
        # Adaptive scanning
        self.adaptive_enabled = self.config.get('adaptive_scanning', True)
        
        # Exclusion patterns for non-crypto assets
        self.exclude_patterns = [
            r'.*UP/USDT$',
            r'.*DOWN/USDT$',
            r'.*BULL/USDT$',
            r'.*BEAR/USDT$',
            r'.*LONG/USDT$',
            r'.*SHORT/USDT$',
            r'^[A-Z]{3}/[A-Z]{3}$',  # Forex pairs
            r'.*/\d+[A-Z]+$',        # Perpetual futures
            r'.*_[A-Z]+$',           # Cross pairs
            r'USD[0-9]*/USDT$',      # USD stablecoin variants
            r'^EUR/USDT$', r'^GBP/USDT$', r'^JPY/USDT$', r'^AUD/USDT$',  # Forex
            r'^FDUSD/USDT$', r'^BUSD/USDT$', r'^TUSD/USDT$', r'^USDP/USDT$',  # Stablecoins
            r'^PAXG/USDT$',  # Gold token
        ]
        
        # Category patterns
        self.category_patterns = {
            'BITCOIN': ['BTC'],
            'ETHEREUM': ['ETH'],
            'LAYER1': ['SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'ATOM', 'NEAR', 'ALGO', 'FTM', 'EGLD', 'KAS', 'SUI', 'APT', 'SEI'],
            'LAYER2': ['ARB', 'OP', 'IMX', 'METIS', 'BOBA', 'MNT', 'STRK', 'MANTA', 'ZRO'],
            'DEFI': ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'SUSHI', 'LDO', 'CAKE', 'RAY', 'INJ', 'JUP', 'PYTH'],
            'MEME': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'COQ', 'TURBO', 'POPCAT'],
            'AI': ['AGIX', 'FET', 'OCEAN', 'RNDR', 'TAO', 'AKT', 'PRIME', 'NMR', 'PAAL', 'AITECH', 'OLAS', 'GLM'],
            'GAMING': ['AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'ILV', 'YGG', 'PIXEL', 'BEAM', 'MAGIC', 'PRIME', 'PORTAL'],
            'RWA': ['ONDO', 'POLYX', 'CFG', 'TRU', 'MPL', 'PROPC', 'LABS'],
            'PRIVACY': ['XMR', 'ZEC', 'DASH', 'SCRT', 'ZEN'],
            'STORAGE': ['FIL', 'AR', 'STORJ', 'SC', 'BLZ', 'HNT'],
            'ORACLES': ['LINK', 'BAND', 'TRB', 'API3', 'PYTH', 'UMA'],
            'EXCHANGE': ['BNB', 'FTT', 'OKB', 'HT', 'KCS', 'GT', 'MX', 'LEO', 'CRO'],
            'LSD': ['LDO', 'RPL', 'SWISE', 'FXS', 'ANKR', 'RETH', 'STETH'],
            'BRIDGES': ['STG', 'MULTI', 'CELER', 'SYN', 'ANY'],
            'SOCIAL': ['LENS', 'GAL', 'RALLY', 'BAND'],
            'NFT': ['BLUR', 'LOOKS', 'X2Y2'],
        }
        
        # Exclusion list for problematic symbols
        self.excluded_symbols: Set[str] = set()
        
    def _categorize_symbol(self, symbol: str) -> str:
        """Categorize symbol into crypto sector."""
        symbol_upper = symbol.upper().replace('/USDT', '')
        
        # Check exact matches first
        for category, patterns in self.category_patterns.items():
            if symbol_upper in patterns:
                return category
        
        # Check pattern matches
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern in symbol_upper:
                    return category
        
        # Check for new listings or unusual patterns
        if len(symbol_upper) <= 4:
            # Likely major coin if short ticker
            return 'LARGE_CAP'
        elif symbol_upper.endswith('FI') or 'DEFI' in symbol_upper:
            return 'DEFI'
        elif symbol_upper.endswith('AI') or 'INTEL' in symbol_upper:
            return 'AI'
        elif any(game in symbol_upper for game in ['GAME', 'PLAY', 'Guild']):
            return 'GAMING'
        else:
            return 'ALTCAP'
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid cryptocurrency (not stablecoin/forex)."""
        symbol_upper = symbol.upper()
        
        # Must end with /USDT
        if not symbol_upper.endswith('/USDT'):
            return False
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, symbol_upper):
                return False
        
        # Additional manual exclusions
        invalid_bases = ['USD1', 'FDUSD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 
                        'CHF', 'PAXG', 'XAUT', 'XAUT', 'BULL', 'BEAR']
        base_symbol = symbol_upper.replace('/USDT', '')
        
        if base_symbol in invalid_bases:
            return False
        
        # Must be reasonable length
        if len(base_symbol) < 2 or len(base_symbol) > 10:
            return False
        
        return True
    
    def initialize(self, tickers: List[Any]):
        """
        Bootstraps the scheduler with pre-filtered tickers.
        Safe against data format errors.
        """
        self.universe.clear()
        self.symbol_history.clear()
        
        valid_tickers = []
        
        # Robust Unpacking Loop
        for item in tickers:
            try:
                # We expect a tuple of 3: (symbol, volume, category)
                # But we handle bad data gracefully just in case.
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    symbol, volume, category = item
                    if self._is_valid_symbol(symbol):
                        valid_tickers.append((symbol, volume, category))
                else:
                    # Silently skip malformed data to keep logs clean
                    continue
            except Exception:
                continue
        
        # Sort by volume
        valid_tickers.sort(key=lambda x: x[1], reverse=True)
        total = len(valid_tickers)
        
        for i, (symbol, volume, category) in enumerate(valid_tickers):
            # Assign Initial Tiers based on Volume Rank
            if i < (total * self.tier_distribution['hot']):
                tier = ScanTier.HOT
            elif i < (total * (self.tier_distribution['hot'] + self.tier_distribution['normal'])):
                tier = ScanTier.NORMAL
            else:
                tier = ScanTier.COLD
                
            self.universe[symbol] = TokenState(
                symbol=symbol, 
                tier=tier,
                category=category,
                last_scan=time.time() - (i * 10),  # Stagger initial scans
                volume_rank=i
            )
            
            # Initialize history
            self.symbol_history[symbol] = deque(maxlen=self.history_length)
            
        self.logger.info(f"✅ Scheduler initialized with {len(valid_tickers)} crypto assets "
                        f"(excluded {len(tickers) - len(valid_tickers)} items)")
        
        # Log category distribution
        self._log_category_distribution()
    
    def _log_category_distribution(self):
        """Log distribution of symbols across categories."""
        category_counts = defaultdict(int)
        for token in self.universe.values():
            category_counts[token.category] += 1
        
        self.logger.info("📊 Category Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {category}: {count} symbols")
    
    def get_due_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Returns a batch of symbols that need scanning.
        Uses priority scoring with category optimization.
        """
        if limit is None:
            limit = self.batch_size
            
        now = time.time()
        
        # Filter due tokens
        due_tokens = []
        for token in self.universe.values():
            if token.is_due():
                due_tokens.append(token)
        
        if not due_tokens:
            return []
        
        # Sort by priority score (descending)
        due_tokens.sort(key=lambda x: x.get_priority_score(), reverse=True)
        
        # Apply category-based adaptive limits
        if self.adaptive_enabled:
            limit = self._calculate_adaptive_limit(limit, due_tokens)
        
        # Ensure category diversity in selection
        selected_tokens = self._ensure_category_diversity(due_tokens, limit)
        
        tasks = []
        for token in selected_tokens:
            # Update token state
            token.last_scan = now
            token.scan_count += 1
            
            # Determine scan mode based on activity and category
            scan_mode = self._determine_scan_mode(token)
            
            # Add historical context
            history = list(self.symbol_history.get(token.symbol, []))
            
            tasks.append({
                'symbol': token.symbol,
                'tier': token.tier.name,
                'category': token.category,
                'mode': scan_mode['mode'],
                'timeframe': scan_mode['timeframe'],
                'priority_score': token.get_priority_score(),
                'activity_score': token.activity_score,
                'volume_rank': token.volume_rank,
                'history': history[-10:] if len(history) > 10 else history,
                'scan_count': token.scan_count
            })
            
            # Update history
            self.symbol_history[token.symbol].append({
                'timestamp': now,
                'mode': scan_mode['mode'],
                'tier': token.tier.name,
                'category': token.category
            })
        
        self.total_scans += len(tasks)
        
        # Update stats periodically
        if now - self.last_stats_update > 300:  # Every 5 minutes
            self._update_performance_stats()
            self.last_stats_update = now
        
        # Log batch composition
        if tasks:
            self._log_batch_composition(tasks)
        
        return tasks
    
    def _determine_scan_mode(self, token: TokenState) -> Dict[str, str]:
        """Determine scan mode based on token characteristics."""
        # Higher activity = deeper scan
        if token.activity_score >= 3:
            return {'mode': 'DEEP', 'timeframe': '15m'}
        elif token.activity_score >= 1:
            return {'mode': 'STANDARD', 'timeframe': '1h'}
        else:
            # Category-specific defaults
            if token.category in ['LAYER1', 'ETHEREUM', 'BITCOIN']:
                return {'mode': 'STANDARD', 'timeframe': '1h'}
            elif token.category == 'MEME':
                return {'mode': 'LIGHT', 'timeframe': '1h'}  # Memes need faster scans
            else:
                return {'mode': 'LIGHT', 'timeframe': '4h'}
    
    def _ensure_category_diversity(self, due_tokens: List[TokenState], limit: int) -> List[TokenState]:
        """Ensure we get tokens from different categories."""
        if len(due_tokens) <= limit:
            return due_tokens[:limit]
        
        # Group by category
        by_category = defaultdict(list)
        for token in due_tokens:
            by_category[token.category].append(token)
        
        # Calculate fair distribution
        total_categories = len(by_category)
        base_per_category = max(1, limit // total_categories)
        
        selected = []
        
        # Take from each category
        for category, tokens in by_category.items():
            # Adjust based on category performance
            category_multiplier = self._get_category_multiplier(category)
            take_count = min(int(base_per_category * category_multiplier), len(tokens))
            selected.extend(tokens[:take_count])
        
        # If we need more, take highest priority regardless of category
        if len(selected) < limit:
            remaining_needed = limit - len(selected)
            # Get tokens not yet selected
            selected_symbols = {t.symbol for t in selected}
            remaining = [t for t in due_tokens if t.symbol not in selected_symbols]
            remaining.sort(key=lambda x: x.get_priority_score(), reverse=True)
            selected.extend(remaining[:remaining_needed])
        
        # Ensure we don't exceed limit
        return selected[:limit]
    
    def _get_category_multiplier(self, category: str) -> float:
        """Get multiplier for category based on performance."""
        perf = self.category_performance.get(category, {"success_rate": 0.5})
        success_rate = perf["success_rate"]
        
        # Categories with higher success rates get more slots
        if success_rate > 0.6:
            return 1.5
        elif success_rate > 0.55:
            return 1.2
        elif success_rate < 0.45:
            return 0.7
        elif success_rate < 0.4:
            return 0.5
        else:
            return 1.0
    
    def _calculate_adaptive_limit(self, base_limit: int, due_tokens: List[TokenState]) -> int:
        """
        Adjust batch limit based on market conditions and system load.
        """
        # Reduce during high volatility (more symbols need attention)
        if len(due_tokens) > 100:
            return max(5, base_limit // 2)
        
        # Increase during quiet periods
        if len(due_tokens) < 20:
            return min(base_limit * 2, 50)
        
        # Category-based adjustment
        high_priority_categories = ['LAYER1', 'ETHEREUM', 'BITCOIN', 'AI']
        high_priority_count = sum(1 for t in due_tokens if t.category in high_priority_categories)
        
        if high_priority_count > len(due_tokens) * 0.3:  # >30% high priority
            return min(base_limit + 5, 30)
        
        return base_limit
    
    def update_task_result(self, symbol: str, result: Dict[str, Any]):
        """
        Feedback Loop: Update token based on scan results.
        """
        if symbol not in self.universe:
            return
            
        token = self.universe[symbol]
        
        # Extract results
        signals_found = len(result.get('signals', []))
        error = result.get('error', False)
        
        # Update category performance
        self._update_category_performance(token.category, signals_found > 0)
        
        # Update token activity
        if signals_found > 0:
            token.activity_score = min(token.activity_score + signals_found * 2, 10)
            token.consecutive_inactive = 0
            token.last_signal_time = time.time()
            
            # Promote to HOT if activity is high
            if token.activity_score >= 3 and token.tier != ScanTier.HOT:
                old_tier = token.tier.name
                token.tier = ScanTier.HOT
                self.logger.debug(f"📈 Promoted {symbol} ({token.category}) from {old_tier} to HOT")
                
        else:
            # No signals found
            token.consecutive_inactive += 1
            
            # Decay activity score
            if token.consecutive_inactive >= 3:
                token.activity_score = max(0, token.activity_score - 1)
                token.consecutive_inactive = 0
                
                # Demote if inactive for too long
                if token.activity_score == 0 and token.tier == ScanTier.HOT:
                    token.tier = ScanTier.NORMAL
                    self.logger.debug(f"📉 Demoted {symbol} from HOT to NORMAL (inactivity)")
        
        # Handle errors
        if error:
            self.logger.warning(f"Scan error for {symbol}, adding to watch list")
            token.activity_score = min(token.activity_score + 1, 10)  # Increase to monitor
        
        # Update history with result
        if symbol in self.symbol_history and self.symbol_history[symbol]:
            last_entry = self.symbol_history[symbol][-1]
            last_entry['signals'] = signals_found
            last_entry['error'] = error
    
    def _update_category_performance(self, category: str, signal_found: bool):
        """Update category performance statistics."""
        if category not in self.category_performance:
            self.category_performance[category] = {
                "signals": 0,
                "scans": 0,
                "success_rate": 0.5
            }
        
        perf = self.category_performance[category]
        perf["scans"] += 1
        
        if signal_found:
            perf["signals"] += 1
        
        # Calculate success rate (EMA smoothing)
        current_rate = perf["signals"] / perf["scans"] if perf["scans"] > 0 else 0.5
        perf["success_rate"] = (perf["success_rate"] * 0.9) + (current_rate * 0.1)
    
    def update_status(self, symbol: str, activity_detected: bool):
        """
        Simple feedback method for backward compatibility.
        """
        if symbol not in self.universe:
            return
            
        token = self.universe[symbol]
        
        if activity_detected:
            token.activity_score = min(token.activity_score + 1, 10)
            token.consecutive_inactive = 0
            
            if token.activity_score >= 2 and token.tier != ScanTier.HOT:
                token.tier = ScanTier.HOT
        else:
            token.consecutive_inactive += 1
            
            if token.consecutive_inactive >= 5:
                token.activity_score = max(0, token.activity_score - 1)
                token.consecutive_inactive = 0
                
                if token.activity_score == 0 and token.tier == ScanTier.HOT:
                    token.tier = ScanTier.NORMAL
    
    def update_universe(self, tickers: List[Tuple[str, float, str]]):
        """
        Refreshes the universe with new tickers.
        """
        current_set = set(self.universe.keys())
        new_symbols = {symbol for symbol, _, _ in tickers}
        
        # Add New
        added_count = 0
        for symbol, volume, category in tickers:
            if symbol not in current_set and self._is_valid_symbol(symbol):
                # Find volume rank among new symbols
                volume_rank = sorted(tickers, key=lambda x: x[1], reverse=True).index((symbol, volume, category))
                
                self.universe[symbol] = TokenState(
                    symbol=symbol,
                    tier=ScanTier.NORMAL,
                    category=category,
                    volume_rank=volume_rank
                )
                self.symbol_history[symbol] = deque(maxlen=self.history_length)
                added_count += 1
        
        # Remove Old (if not in new list)
        removed_count = 0
        for symbol in list(self.universe.keys()):
            if symbol not in new_symbols:
                # Check if inactive for a while
                token = self.universe[symbol]
                if time.time() - token.last_scan > 86400:  # 24 hours
                    del self.universe[symbol]
                    if symbol in self.symbol_history:
                        del self.symbol_history[symbol]
                    removed_count += 1
        
        if added_count > 0 or removed_count > 0:
            self.logger.info(f"Updated universe: +{added_count}, -{removed_count} symbols")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Returns comprehensive scheduler statistics.
        """
        now = time.time()
        
        # Tier counts
        hot = sum(1 for t in self.universe.values() if t.tier == ScanTier.HOT)
        normal = sum(1 for t in self.universe.values() if t.tier == ScanTier.NORMAL)
        cold = sum(1 for t in self.universe.values() if t.tier == ScanTier.COLD)
        
        # Category analysis
        category_counts = defaultdict(int)
        for token in self.universe.values():
            category_counts[token.category] += 1
        
        # Activity analysis
        active_symbols = sum(1 for t in self.universe.values() if t.activity_score > 0)
        highly_active = sum(1 for t in self.universe.values() if t.activity_score >= 3)
        
        # Recent activity
        recent_signals = sum(1 for t in self.universe.values() 
                           if t.last_signal_time > now - 3600)  # Last hour
        
        # Performance metrics
        scans_per_hour = self.scans_per_hour
        
        # Queue analysis
        due_now = sum(1 for t in self.universe.values() if t.is_due())
        
        # Top categories by success rate
        top_categories = sorted(
            [(cat, perf["success_rate"]) 
             for cat, perf in self.category_performance.items() 
             if perf["scans"] >= 10],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_monitored": len(self.universe),
            "tier_1_hot": hot,
            "tier_2_normal": normal,
            "tier_3_cold": cold,
            "category_distribution": dict(category_counts),
            "active_symbols": active_symbols,
            "highly_active": highly_active,
            "recent_signals": recent_signals,
            "scans_per_hour": scans_per_hour,
            "total_scans": self.total_scans,
            "due_now": due_now,
            "top_categories": top_categories,
            "excluded_symbols": len(self.excluded_symbols)
        }
    
    def get_top_active_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the most active symbols for monitoring.
        """
        active_tokens = [t for t in self.universe.values() if t.activity_score > 0]
        active_tokens.sort(key=lambda x: x.activity_score, reverse=True)
        
        result = []
        for token in active_tokens[:limit]:
            result.append({
                'symbol': token.symbol,
                'category': token.category,
                'activity_score': token.activity_score,
                'tier': token.tier.name,
                'last_signal_hours': (time.time() - token.last_signal_time) / 3600 if token.last_signal_time > 0 else None,
                'scan_count': token.scan_count,
                'volume_rank': token.volume_rank
            })
        
        return result
    
    def _log_batch_composition(self, tasks: List[Dict[str, Any]]):
        """Log composition of the current batch."""
        category_counts = defaultdict(int)
        for task in tasks:
            category_counts[task['category']] += 1
        
        if category_counts:
            composition_str = ", ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
            self.logger.debug(f"Batch composition: {composition_str}")
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        now = time.time()
        
        if hasattr(self, 'last_stats_scans') and hasattr(self, 'last_stats_time'):
            elapsed_hours = (now - self.last_stats_time) / 3600
            if elapsed_hours > 0:
                scans_since_last = self.total_scans - self.last_stats_scans
                self.scans_per_hour = scans_since_last / elapsed_hours
            else:
                self.scans_per_hour = 0
        else:
            self.scans_per_hour = 0
        
        self.last_stats_scans = self.total_scans
        self.last_stats_time = now
    
    async def maintenance_cycle(self):
        """
        Periodic maintenance to clean up and optimize.
        """
        while True:
            try:
                now = time.time()
                
                # Clean up old history
                symbols_to_clean = []
                for symbol, history in self.symbol_history.items():
                    if history and now - history[0].get('timestamp', 0) > 86400 * 7:  # 7 days
                        symbols_to_clean.append(symbol)
                
                for symbol in symbols_to_clean:
                    if symbol in self.symbol_history:
                        self.symbol_history[symbol].clear()
                
                # Rebalance tiers based on long-term activity
                rebalanced = 0
                for token in self.universe.values():
                    # If symbol has been HOT for 24+ hours without signals, demote
                    if token.tier == ScanTier.HOT and now - token.last_signal_time > 86400:
                        token.tier = ScanTier.NORMAL
                        rebalanced += 1
                
                if rebalanced > 0:
                    self.logger.debug(f"Rebalanced {rebalanced} symbols based on activity")
                
                # Update category scores based on recent performance
                self._update_category_scores()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Maintenance cycle error: {e}")
                await asyncio.sleep(300)
    
    def _update_category_scores(self):
        """Update category performance scores."""
        for symbol, token in self.universe.items():
            perf = self.category_performance.get(token.category, {"success_rate": 0.5})
            token.category_score = perf["success_rate"]
    
    async def shutdown(self):
        """Cleanup before shutdown."""
        self.logger.info("Scanner shutdown initiated")
        
        # Clear data structures
        self.universe.clear()
        self.symbol_history.clear()
        self.excluded_symbols.clear()
        self.category_performance.clear()