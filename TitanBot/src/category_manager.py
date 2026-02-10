"""
TITAN-X CATEGORY MANAGER
Advanced cryptocurrency categorization and sector analysis
"""

import logging
import re
from typing import Dict, List, Set, Any
from datetime import datetime, timedelta

class CategoryManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("CategoryManager")
        
        # Enhanced category definitions
        self.categories = {
            'BITCOIN': {
                'symbols': ['BTC'],
                'description': 'Bitcoin Ecosystem',
                'weight': 1.0,
                'volatility_profile': 'MEDIUM'
            },
            'ETHEREUM': {
                'symbols': ['ETH'],
                'description': 'Ethereum Ecosystem',
                'weight': 0.9,
                'volatility_profile': 'HIGH'
            },
            'LAYER1': {
                'patterns': ['SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'ATOM', 'NEAR', 'ALGO', 'FTM', 'EGLD'],
                'description': 'Layer 1 Blockchains',
                'weight': 0.8,
                'volatility_profile': 'HIGH'
            },
            'LAYER2': {
                'patterns': ['ARB', 'OP', 'IMX', 'METIS', 'BOBA', 'MNT', 'STRK', 'ZKSYNC'],
                'description': 'Layer 2 Scaling',
                'weight': 0.85,
                'volatility_profile': 'VERY_HIGH'
            },
            'DEFI': {
                'patterns': ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'SUSHI', 'LDO', 'CAKE', 'RAY'],
                'description': 'Decentralized Finance',
                'weight': 0.75,
                'volatility_profile': 'HIGH'
            },
            'MEME': {
                'patterns': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'COQ', 'TURBO'],
                'description': 'Meme Coins',
                'weight': 0.4,
                'volatility_profile': 'EXTREME'
            },
            'AI': {
                'patterns': ['AGIX', 'FET', 'OCEAN', 'RNDR', 'TAO', 'AKT', 'PRIME', 'NMR', 'PAAL', 'AITECH'],
                'description': 'Artificial Intelligence',
                'weight': 0.85,
                'volatility_profile': 'HIGH'
            },
            'GAMING': {
                'patterns': ['AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'ILV', 'YGG', 'PIXEL', 'BEAM', 'MAGIC'],
                'description': 'Gaming & Metaverse',
                'weight': 0.7,
                'volatility_profile': 'HIGH'
            },
            'RWA': {
                'patterns': ['ONDO', 'POLYX', 'CFG', 'TRU', 'MPL', 'PROPC', 'LABS'],
                'description': 'Real World Assets',
                'weight': 0.6,
                'volatility_profile': 'MEDIUM'
            },
            'PRIVACY': {
                'patterns': ['XMR', 'ZEC', 'DASH', 'SCRT', 'ZEN'],
                'description': 'Privacy Coins',
                'weight': 0.5,
                'volatility_profile': 'MEDIUM'
            },
            'STORAGE': {
                'patterns': ['FIL', 'AR', 'STORJ', 'SC', 'BLZ'],
                'description': 'Decentralized Storage',
                'weight': 0.65,
                'volatility_profile': 'HIGH'
            },
            'ORACLES': {
                'patterns': ['LINK', 'BAND', 'TRB', 'API3', 'PYTH', 'UMA'],
                'description': 'Oracle Networks',
                'weight': 0.75,
                'volatility_profile': 'HIGH'
            },
            'EXCHANGE': {
                'patterns': ['BNB', 'FTT', 'OKB', 'HT', 'KCS', 'GT', 'MX', 'LEO', 'CRO'],
                'description': 'Exchange Tokens',
                'weight': 0.7,
                'volatility_profile': 'MEDIUM'
            },
            'LSD': {
                'patterns': ['LDO', 'RPL', 'SWISE', 'FXS', 'ANKR'],
                'description': 'Liquid Staking Derivatives',
                'weight': 0.8,
                'volatility_profile': 'MEDIUM'
            },
            'BRIDGES': {
                'patterns': ['STG', 'MULTI', 'CELER', 'SYN'],
                'description': 'Cross-Chain Bridges',
                'weight': 0.65,
                'volatility_profile': 'HIGH'
            }
        }
        
        # Exclusion lists
        self.exclude_categories = ['STABLECOINS', 'FOREX', 'LEVERAGED']
        self.exclude_patterns = [
            r'^[A-Z]{3}/[A-Z]{3}$',  # Forex pairs
            r'.*UP$', r'.*DOWN$', r'.*BULL$', r'.*BEAR$',  # Leveraged
            r'.*3[SL]$',  # 3x leveraged
            r'USD[0-9]*$', r'EUR$', r'GBP$', r'JPY$',  # Stablecoins/forex
        ]
        
        # Sector correlations
        self.sector_correlations = {
            'LAYER1': ['LAYER2', 'DEFI'],
            'ETHEREUM': ['LAYER2', 'DEFI', 'LSD'],
            'MEME': [],  # Low correlation with fundamentals
            'AI': ['GAMING', 'COMPUTE'],
            'GAMING': ['METAVERSE', 'NFT']
        }
        
        # Performance tracking
        self.sector_performance = {}
        self.last_update = datetime.now()
        
    def categorize_symbol(self, symbol: str) -> str:
        """Categorize a symbol into appropriate sector."""
        # Remove pair suffix
        base_symbol = symbol.split('/')[0].upper()
        
        # Check exact matches first
        for category, data in self.categories.items():
            if 'symbols' in data and base_symbol in data['symbols']:
                return category
        
        # Check pattern matches
        for category, data in self.categories.items():
            if 'patterns' in data:
                for pattern in data['patterns']:
                    if pattern in base_symbol:
                        return category
        
        # Check for new listings or unusual patterns
        if len(base_symbol) <= 4:
            return 'LARGE_CAP'
        elif base_symbol.endswith('FI'):
            return 'DEFI'
        elif base_symbol.endswith('AI'):
            return 'AI'
        else:
            return 'ALTCAP'
    
    def should_include_symbol(self, symbol: str) -> bool:
        """Determine if symbol should be included in analysis."""
        symbol_upper = symbol.upper()
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, symbol_upper):
                return False
        
        # Check for stablecoin/forex patterns
        if any(x in symbol_upper for x in ['USD1', 'FDUSD', 'TUSD', 'USDP', 'DAI', 'BUSD']):
            return False
        
        # Check for forex
        if re.match(r'^[A-Z]{3}/USDT$', symbol_upper):
            base = symbol_upper.replace('/USDT', '')
            if base in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD']:
                return False
        
        return True
    
    def get_sector_health(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze health of different sectors."""
        sector_counts = {}
        sector_symbols = {}
        
        for symbol in symbols:
            category = self.categorize_symbol(symbol)
            if category not in sector_counts:
                sector_counts[category] = 0
                sector_symbols[category] = []
            sector_counts[category] += 1
            sector_symbols[category].append(symbol)
        
        # Calculate diversity score
        total_symbols = len(symbols)
        diversity_score = len(sector_counts) / len(self.categories) * 100
        
        return {
            'sector_distribution': sector_counts,
            'sector_symbols': sector_symbols,
            'diversity_score': diversity_score,
            'recommended_sectors': self._get_recommended_sectors(sector_counts)
        }
    
    def _get_recommended_sectors(self, sector_counts: Dict[str, int]) -> List[str]:
        """Get recommended sectors based on current distribution."""
        if not sector_counts:
            return ['LAYER1', 'DEFI', 'AI']  # Default recommendations
        
        # Find underrepresented sectors
        avg_count = sum(sector_counts.values()) / len(sector_counts) if sector_counts else 0
        underrepresented = []
        
        for category in self.categories:
            if category not in self.exclude_categories:
                current_count = sector_counts.get(category, 0)
                if current_count < avg_count * 0.5:  # Less than half of average
                    underrepresented.append(category)
        
        return underrepresented[:3]  # Top 3 recommendations
    
    def get_trading_recommendations(self, symbol: str, category: str) -> Dict[str, Any]:
        """Get trading recommendations based on category."""
        recommendations = {
            'position_size_multiplier': 1.0,
            'risk_adjustment': 1.0,
            'holding_period': 'MEDIUM',
            'entry_strategy': 'STANDARD'
        }
        
        # Adjust based on category volatility
        category_data = self.categories.get(category, {})
        volatility = category_data.get('volatility_profile', 'MEDIUM')
        
        if volatility == 'EXTREME':
            recommendations['position_size_multiplier'] = 0.5
            recommendations['risk_adjustment'] = 0.7
            recommendations['holding_period'] = 'SHORT'
        elif volatility == 'VERY_HIGH':
            recommendations['position_size_multiplier'] = 0.7
            recommendations['risk_adjustment'] = 0.8
        elif volatility == 'HIGH':
            recommendations['position_size_multiplier'] = 0.8
            recommendations['risk_adjustment'] = 0.9
        elif volatility == 'LOW':
            recommendations['position_size_multiplier'] = 1.2
            recommendations['risk_adjustment'] = 1.1
            recommendations['holding_period'] = 'LONG'
        
        # Specific category adjustments
        if category == 'MEME':
            recommendations['entry_strategy'] = 'AGGRESSIVE'
            recommendations['holding_period'] = 'VERY_SHORT'
        elif category in ['LAYER1', 'ETHEREUM']:
            recommendations['entry_strategy'] = 'CONSERVATIVE'
            recommendations['holding_period'] = 'LONG'
        
        return recommendations
    
    def update_sector_performance(self, symbol: str, pnl: float):
        """Update sector performance tracking."""
        category = self.categorize_symbol(symbol)
        
        if category not in self.sector_performance:
            self.sector_performance[category] = {
                'total_pnl': 0.0,
                'trade_count': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        stats = self.sector_performance[category]
        stats['total_pnl'] += pnl
        stats['trade_count'] += 1
        
        if pnl > 0:
            stats['winning_trades'] += 1
        else:
            stats['losing_trades'] += 1
        
        # Clean old data periodically
        if datetime.now() - self.last_update > timedelta(hours=24):
            self._clean_old_data()
            self.last_update = datetime.now()
    
    def _clean_old_data(self):
        """Clean old performance data."""
        # Keep only last 30 days of data
        pass  # Implement if using time-series data
    
    def get_sector_insights(self) -> Dict[str, Any]:
        """Get insights about sector performance."""
        insights = {
            'best_performing': [],
            'worst_performing': [],
            'most_active': [],
            'recommendations': []
        }
        
        for category, stats in self.sector_performance.items():
            if stats['trade_count'] > 0:
                win_rate = (stats['winning_trades'] / stats['trade_count']) * 100
                avg_pnl = stats['total_pnl'] / stats['trade_count']
                
                if win_rate > 60:
                    insights['best_performing'].append({
                        'category': category,
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl
                    })
                elif win_rate < 40:
                    insights['worst_performing'].append({
                        'category': category,
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl
                    })
                
                if stats['trade_count'] >= 5:  # Active category
                    insights['most_active'].append({
                        'category': category,
                        'trade_count': stats['trade_count']
                    })
        
        # Sort and limit
        insights['best_performing'].sort(key=lambda x: x['win_rate'], reverse=True)
        insights['worst_performing'].sort(key=lambda x: x['win_rate'])
        insights['most_active'].sort(key=lambda x: x['trade_count'], reverse=True)
        
        # Generate recommendations
        if insights['best_performing']:
            best_cat = insights['best_performing'][0]['category']
            insights['recommendations'].append(f"Focus on {best_cat} sector (high win rate)")
        
        if insights['worst_performing']:
            worst_cat = insights['worst_performing'][0]['category']
            insights['recommendations'].append(f"Reduce exposure to {worst_cat} sector")
        
        return insights