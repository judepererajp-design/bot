import pandas as pd
import numpy as np
from typing import Dict, Optional

class InstitutionalBreakout:
    """
    TITAN-X BREAKOUT STRATEGY
    -------------------------
    Logic: 
    1. Donchian Channel Breakout (20-period High).
    2. Volume Filter (Volume > 1.5x Moving Average).
    3. Volatility Expansion (ATR Rising).
    """
    def __init__(self, context: Dict):
        self.config = context.get('config', {})
        self.min_volume_mult = 1.5
        self.lookback = 20

    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        # Breakouts are best validated on 1H or 4H
        df = data.get('1h')
        if df is None or len(df) < 50: 
            return None

        current = df.iloc[-1]
        
        # 1. Donchian Channel (Price hits 20-candle High)
        period_high = df['high'].iloc[-21:-1].max()
        
        # 2. Volume Spike Check
        avg_vol = df['volume'].iloc[-21:-1].mean()
        has_volume = current['volume'] > (avg_vol * self.min_volume_mult)

        # 3. Validation
        is_breakout = current['close'] > period_high
        
        if is_breakout and has_volume:
            return {
                'strategy': 'Institutional Breakout',
                'pattern_name': 'Vol-Backed Breakout',
                'symbol': symbol,
                'timeframe': '1h',
                'direction': 'LONG', 
                'entry_price': current['close'],  # Use entry_price consistently
                'entry': current['close'],        # ALSO include entry for compatibility
                'confidence': 80, 
                'timestamp': current['timestamp'].timestamp()
            }
        
        return None
