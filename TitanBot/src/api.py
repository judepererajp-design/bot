"""
TITAN-X EXCHANGE INTERFACE (ENHANCED & FIXED)
- Filters out stablecoins, forex, leveraged tokens
- Adds cryptocurrency categorization
- Improved symbol filtering
"""

import ccxt.async_support as ccxt
import asyncio
import logging
import pandas as pd
import re
from typing import List, Dict, Optional, Any, Set, Tuple

class ExchangeInterface:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ExchangeAPI")
        self.exchange = None
        
        # Concurrency Control
        self.semaphore = asyncio.Semaphore(5)
        
        # Symbol categorization data
        self.symbol_categories = {}
        self.category_patterns = {
            'LAYER1': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'ATOM', 'NEAR', 'ALGO'],
            'LAYER2': ['ARB', 'OP', 'IMX', 'METIS', 'BOBA', 'MNT'],
            'DEFI': ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'SUSHI', 'LDO', 'AVAX'],
            'MEME': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME'],
            'AI': ['AGIX', 'FET', 'OCEAN', 'RNDR', 'TAO', 'AKT', 'PRIME', 'NMR'],
            'GAMING': ['AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'ILV', 'YGG', 'PIXEL'],
            'RWA': ['ONDO', 'POLYX', 'CFG', 'TRU', 'MPL'],
            'PRIVACY': ['XMR', 'ZEC', 'DASH', 'SCRT'],
            'STORAGE': ['FIL', 'AR', 'STORJ'],
            'ORACLES': ['LINK', 'BAND', 'TRB', 'API3'],
            'EXCHANGE': ['BNB', 'FTT', 'OKB', 'HT', 'KCS', 'GT', 'MX'],
            'STABLECOINS': ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'FDUSD', 'USD1', 'EUR', 'GBP'],
            'LEVERAGED': ['UP', 'DOWN', 'BULL', 'BEAR', 'LONG', 'SHORT']
        }
        
        # Exclusion patterns
        self.exclude_patterns = [
            r'.*UP/USDT$',
            r'.*DOWN/USDT$',
            r'.*BULL/USDT$',
            r'.*BEAR/USDT$',
            r'.*LONG/USDT$',
            r'.*SHORT/USDT$',
            r'^[A-Z]{3}/[A-Z]{3}$',  # Forex pairs like EUR/USDT
            r'.*/\d+[A-Z]+$',        # Perpetual futures
            r'.*_[A-Z]+$'            # Cross pairs
        ]

    async def connect(self):
        """Initializes the exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.config['name'])
            self.exchange = exchange_class({
                'apiKey': self.config.get('api_key', ''),
                'secret': self.config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            if self.config.get('testnet', False):
                self.exchange.set_sandbox_mode(True)
                
            await self.exchange.load_markets()
            self.logger.info(f"✅ Connected to {self.config['name'].upper()}")
            
        except Exception as e:
            self.logger.critical(f"❌ Exchange Connection Failed: {e}")
            raise

    async def fetch_active_tickers(self, min_volume: float, max_count: int) -> List[Tuple[str, float, str]]:
        """
        Fetches all tickers and filters top N by volume.
        Enhanced filtering to remove stablecoins, forex, and leveraged tokens.
        Returns: List of (symbol, volume, category) tuples.
        """
        try:
            self.logger.info("Fetching market data with enhanced filtering...")
            tickers = await self.exchange.fetch_tickers()
            
            candidates = []
            self.symbol_categories = {}
            
            for symbol, data in tickers.items():
                # Apply filters
                if not self._is_valid_symbol(symbol):
                    continue
                
                # Volume filter
                quote_vol = data.get('quoteVolume', 0)
                if quote_vol and quote_vol >= min_volume:
                    # Get category
                    category = self._categorize_symbol(symbol)
                    
                    candidates.append({
                        'symbol': symbol,
                        'volume': quote_vol,
                        'category': category,
                        'price': data.get('last', 0),
                        'change': data.get('percentage', 0)
                    })
                    
                    # Store category for later use
                    self.symbol_categories[symbol] = category
            
            # Sort by volume desc
            candidates.sort(key=lambda x: x['volume'], reverse=True)
            
            # Apply category diversity (ensure we get different types)
            final_list_dicts = self._diversify_selection(candidates, max_count)
            
            # Log categories
            category_stats = {}
            for item in final_list_dicts:
                cat = item['category']
                category_stats[cat] = category_stats.get(cat, 0) + 1
            
            self.logger.info(f"✅ Refined Universe: {len(final_list_dicts)} assets")
            self.logger.info(f"📊 Categories: {category_stats}")
            
            # --- CRITICAL FIX: Return Tuples (Symbol, Volume, Category) ---
            # The Scanner expects this format to unpack correctly.
            result_tuples = [(item['symbol'], item['volume'], item['category']) for item in final_list_dicts]
            return result_tuples

        except Exception as e:
            self.logger.error(f"Ticker Fetch Error: {e}")
            return []

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for trading analysis."""
        symbol_upper = symbol.upper()
        
        # Must end with /USDT
        if not symbol_upper.endswith('/USDT'):
            return False
        
        # Remove USDT suffix for checking
        base_symbol = symbol_upper.replace('/USDT', '')
        
        # Exclude leveraged tokens
        if any(leveraged in base_symbol for leveraged in self.category_patterns['LEVERAGED']):
            return False
        
        # Exclude stablecoins
        if any(stablecoin in base_symbol for stablecoin in self.category_patterns['STABLECOINS']):
            return False
        
        # Exclude by regex patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, symbol_upper):
                return False
        
        # Must have reasonable length (not too short or too long)
        if len(base_symbol) < 2 or len(base_symbol) > 10:
            return False
        
        # Must be mostly letters (avoid numbers)
        if sum(1 for c in base_symbol if c.isalpha()) / len(base_symbol) < 0.7:
            return False
        
        return True

    def _categorize_symbol(self, symbol: str) -> str:
        """Categorize symbol into crypto sectors."""
        symbol_upper = symbol.upper().replace('/USDT', '')
        
        # Check each category
        for category, patterns in self.category_patterns.items():
            if category in ['STABLECOINS', 'LEVERAGED']:
                continue  # Already filtered out
                
            for pattern in patterns:
                if pattern in symbol_upper:
                    return category
        
        # Default categories based on market cap or other heuristics
        if symbol_upper == 'BTC':
            return 'BITCOIN'
        elif symbol_upper == 'ETH':
            return 'ETHEREUM'
        elif len(symbol_upper) <= 4:  # Likely major coin
            return 'LARGE_CAP'
        else:
            return 'ALTCAP'

    def _diversify_selection(self, candidates: List[Dict], max_count: int) -> List[Dict]:
        """Ensure diversity across categories in final selection."""
        if len(candidates) <= max_count:
            return candidates
        
        # Group by category
        by_category = {}
        for candidate in candidates:
            cat = candidate['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(candidate)
        
        # Calculate fair distribution
        total_categories = len(by_category)
        base_per_category = max(1, max_count // total_categories)
        
        selected = []
        
        # Take from each category
        for category, items in by_category.items():
            take_count = min(base_per_category, len(items))
            selected.extend(items[:take_count])
        
        # If we need more, take top volume regardless of category
        if len(selected) < max_count:
            remaining_needed = max_count - len(selected)
            # Get all candidates not yet selected
            all_candidates_dict = {c['symbol']: c for c in candidates}
            selected_symbols = {c['symbol'] for c in selected}
            remaining = [all_candidates_dict[sym] for sym in all_candidates_dict 
                        if sym not in selected_symbols]
            remaining.sort(key=lambda x: x['volume'], reverse=True)
            selected.extend(remaining[:remaining_needed])
        
        # Sort final selection by volume
        selected.sort(key=lambda x: x['volume'], reverse=True)
        return selected[:max_count]

    async def fetch_ohlcv_batch(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple timeframes."""
        async with self.semaphore:
            tasks = []
            for tf in timeframes:
                tasks.append(self._fetch_single_tf(symbol, tf))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data_map = {}
            for tf, res in zip(timeframes, results):
                if isinstance(res, pd.DataFrame) and not res.empty:
                    data_map[tf] = res
            
            return data_map

    async def _fetch_single_tf(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Helper to fetch one timeframe."""
        try:
            # Fetch 100 candles (enough for patterns + EMA)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            self.logger.debug(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None

    def get_symbol_category(self, symbol: str) -> str:
        """Get category for a symbol."""
        return self.symbol_categories.get(symbol, 'UNKNOWN')

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Exchange connection closed.")