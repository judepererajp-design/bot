import sys
import os
import asyncio
import pandas as pd
import logging

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy_manager import StrategyManager

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Backtest")

class BacktestEngine:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            logger.error(f"❌ Data file not found: {csv_path}")
            sys.exit(1)
            
        logger.info(f"📂 Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Ensure timestamp is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Mock the context required by StrategyManager
        # We pass an empty config because we aren't connecting to live APIs
        self.context = {
            'config': {},
            'geo_detector': None, # Disable complex detectors for basic backtest
            'harm_detector': None,
            'order_flow': None,
            'correlation': None
        } 
        
        self.strategies = StrategyManager(self.context)
        self.results = []
        
    async def run(self):
        logger.info("🔄 Starting Simulation...")
        
        # We need a window of data (e.g., 50 candles) to form indicators
        window_size = 50
        
        # Iterate through the CSV
        for i in range(window_size, len(self.df)):
            # 1. Slice "Historical" Data to look like "Live" data
            subset = self.df.iloc[i-window_size:i].copy()
            current_candle = self.df.iloc[i]
            
            # 2. Package it nicely for the strategy
            # Strategies expect a dict of timeframes. We simulate 1h data.
            data_packet = {
                '1h': subset,  
                '15m': subset # Using same data for demo purposes, strictly should be different CSV
            }
            
            # 3. Ask Strategy Manager: "Is there a signal here?"
            try:
                # We mock symbol as BTC/USDT
                signals = await self.strategies.run_analysis("BTC/USDT", data_packet)
                
                # 4. If Signal, Calculate Result
                for sig in signals:
                    self._simulate_trade(sig, current_candle, i)
            except Exception as e:
                logger.error(f"Error at index {i}: {e}")
                
        self._print_stats()

    def _simulate_trade(self, signal, candle, index):
        entry = candle['close']
        
        # Look forward 12 candles (e.g., 12 hours) to see result
        future_window = 12
        if index + future_window >= len(self.df): return
        
        future = self.df.iloc[index+1 : index+future_window+1]

        if signal['direction'] == 'LONG':
            # Highest price in next 12 hours
            max_price = future['high'].max()
            potential_return = (max_price - entry) / entry
        else:
            # Lowest price in next 12 hours
            min_price = future['low'].min()
            potential_return = (entry - min_price) / entry
            
        self.results.append({
            'timestamp': candle['timestamp'],
            'strategy': signal['pattern_name'],
            'direction': signal['direction'],
            'entry': entry,
            'max_return_pct': round(potential_return * 100, 2)
        })

    def _print_stats(self):
        if not self.results:
            logger.info("⚠️ No signals generated.")
            return
            
        res_df = pd.DataFrame(self.results)
        print("\n" + "="*40)
        print("📊 BACKTEST RESULTS")
        print("="*40)
        print(res_df.to_string(index=False))
        print("-" * 40)
        print(f"Total Trades: {len(res_df)}")
        print(f"Avg Max Potential: {res_df['max_return_pct'].mean():.2f}%")
        
        # Simple Win Rate (assuming we target 1% gain)
        wins = len(res_df[res_df['max_return_pct'] > 1.0])
        print(f"Win Rate (>1% gain): {wins}/{len(res_df)} ({(wins/len(res_df))*100:.1f}%)")

if __name__ == "__main__":
    # Create a dummy CSV if it doesn't exist to prevent crash on first run
    if not os.path.exists("data/backtest_data.csv"):
        print("⚠️ No data found. Please place a CSV file at 'data/backtest_data.csv'")
        print("Format: timestamp,open,high,low,close,volume")
    else:
        engine = BacktestEngine("data/backtest_data.csv") 
        asyncio.run(engine.run())