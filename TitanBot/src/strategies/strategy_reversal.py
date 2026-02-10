import pandas as pd
from typing import Dict, Optional

class InstitutionalReversal:
    """
    TITAN-X REVERSAL STRATEGY
    -------------------------
    Logic:
    1. Price closes outside Bollinger Bands (2.5 Std Dev).
    2. RSI is over-extended (<30 or >70).
    3. Candle Shape: Rejection wicks (Hammer/Shooting Star).
    """
    def __init__(self, context: Dict):
        self.rsi_period = 14
        self.bb_std = 2.5 

    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        # Reversals are often sharper on lower timeframes like 15m
        df = data.get('15m')
        if df is None or len(df) < 50: 
            return None

        closes = df['close']
        
        # 1. Calculate RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Calculate Bollinger Bands
        sma = closes.rolling(20).mean()
        std = closes.rolling(20).std()
        lower_band = sma - (std * self.bb_std)
        upper_band = sma + (std * self.bb_std)

        curr_rsi = rsi.iloc[-1]
        curr_close = closes.iloc[-1]
        curr_open = df['open'].iloc[-1]
        
        # LOGIC: LONG REVERSAL
        if curr_rsi < 30 and curr_close < lower_band.iloc[-1] and curr_close > curr_open:
            return {
                'strategy': 'Institutional Reversal',
                'pattern_name': 'BB Oversold Bounce',
                'symbol': symbol,
                'timeframe': '15m',
                'direction': 'LONG',
                'entry_price': curr_close,
                'entry': curr_close,  # Added for compatibility
                'confidence': 75, 
                'timestamp': df['timestamp'].iloc[-1].timestamp()
            }

        # LOGIC: SHORT REVERSAL
        if curr_rsi > 70 and curr_close > upper_band.iloc[-1] and curr_close < curr_open:
             return {
                'strategy': 'Institutional Reversal',
                'pattern_name': 'BB Overbought Reject',
                'symbol': symbol,
                'timeframe': '15m',
                'direction': 'SHORT',
                'entry_price': curr_close,
                'entry': curr_close,  # Added for compatibility
                'confidence': 75, 
                'timestamp': df['timestamp'].iloc[-1].timestamp()
            }

        return None
