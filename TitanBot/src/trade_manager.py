"""
TITAN-X ADVANCED TRADE MANAGER
------------------------------------------------------------------------------
Dynamic Position Management.
1. Breakeven Triggers (Protect Capital).
2. Chandelier Trailing Stops (Capture Trends).
3. Time-Based Exits (Kill Dead Money).
4. Order Flow Reversal Checks.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from .telegram_policy import TelegramPolicy

class TradeManager:
    def __init__(self, context: Dict[str, Any]):
        self.db = context['db']
        self.api = context['api']
        self.telegram = context['telegram']
        self.order_flow = context['order_flow']
        self.logger = logging.getLogger("TradeManager")
        
        # Policy Engine (Governance)
        self.policy = TelegramPolicy(self.db)
        
        # Runtime State (Tracks Highest/Lowest price for Trailing Stops)
        # Structure: { trade_id: { 'highest': float, 'lowest': float, 'entry_ts': float } }
        self.trade_state = {}
        
        # FIX: Add cleanup timer
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Clean up every hour

    async def run_loop(self):
        """Main loop that checks active signals every 15 seconds."""
        self.logger.info("🟢 Advanced Trade Monitor Active")
        while True:
            try:
                # FIX: Periodic cleanup of old trade states
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    await self._cleanup_old_states()
                    self.last_cleanup = time.time()
                
                # 1. Fetch ONLY ACTIVE (Tracked) trades from DB
                active_signals = await self.db.get_recent_signals(limit=50)
                
                if not active_signals:
                    # Clear memory if no active trades to prevent leaks
                    self.trade_state.clear()
                    await asyncio.sleep(30)
                    continue

                # 2. Batch Fetch Current Prices
                symbols = list(set([s['symbol'] for s in active_signals]))
                tickers = await self.api.exchange.fetch_tickers(symbols)

                # 3. Analyze Each Trade
                for signal in active_signals:
                    await self._monitor_signal(signal, tickers)

                await asyncio.sleep(15)

            except Exception as e:
                self.logger.error(f"Monitor Loop Error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_old_states(self):
        """Removes trade states for trades that are no longer active."""
        try:
            # Get all active trade IDs from database
            active_trade_ids = set()
            active_signals = await self.db.get_recent_signals(limit=100)
            for signal in active_signals:
                trade_id = signal.get('trade_id')
                if trade_id:
                    active_trade_ids.add(trade_id)
            
            # Remove states for non-active trades
            to_remove = []
            for trade_id in self.trade_state:
                if trade_id not in active_trade_ids:
                    to_remove.append(trade_id)
            
            for trade_id in to_remove:
                del self.trade_state[trade_id]
                
            if to_remove:
                self.logger.debug(f"Cleaned up {len(to_remove)} old trade states")
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    async def _monitor_signal(self, signal: Dict, tickers: Dict):
        """Checks one specific trade for management events."""
        trade_id = signal['trade_id']
        symbol = signal['symbol']
        direction = signal['direction']
        
        # Data Normalization
        entry = float(signal.get('entry', signal.get('entry_price', 0)))
        stop = float(signal.get('stop', signal.get('stop_loss', 0)))
        tp = float(signal.get('tp', signal.get('take_profit', 0)))
        timestamp = float(signal.get('timestamp', time.time()))
        
        if symbol not in tickers: 
            return
            
        current_price = tickers[symbol]['last']
        
        # --- 0. INITIALIZE RUNTIME STATE ---
        if trade_id not in self.trade_state:
            self.trade_state[trade_id] = {
                'highest': current_price, # Track Highest High (for Long Trailing)
                'lowest': current_price,  # Track Lowest Low (for Short Trailing)
                'last_update': time.time()
            }
        
        # Update State
        state = self.trade_state[trade_id]
        state['highest'] = max(state['highest'], current_price)
        state['lowest'] = min(state['lowest'], current_price)
        state['last_update'] = time.time()
        
        # Calculate Risk Unit (1R) - Used for dynamic calculations
        risk_unit = abs(entry - stop)
        if risk_unit == 0: 
            risk_unit = entry * 0.01 # Fallback
        
        # ====================================================
        # STRATEGY 1: TIME-BASED EXIT (Dead Money)
        # ====================================================
        # If trade is > 12 hours old AND profit is < 0.3R, Kill it.
        hours_active = (time.time() - timestamp) / 3600
        
        if hours_active > 12:
            current_pnl = (current_price - entry) if direction == 'LONG' else (entry - current_price)
            if current_pnl < (risk_unit * 0.3):
                if await self.policy.should_send(trade_id, "EXIT"):
                    msg = (
                        f"⏳ <b>TIME EXIT: {symbol}</b>\n"
                        f"Trade active for {int(hours_active)}h with no momentum.\n"
                        f"Closing to free up capital."
                    )
                    await self.telegram.send_update(trade_id, msg)
                    await self.policy.mark_sent(trade_id, "EXIT")
                    await self.db.deactivate_trade(trade_id)
                    # FIX: Remove from state tracking immediately
                    if trade_id in self.trade_state:
                        del self.trade_state[trade_id]
                    return # Stop processing this trade

        # ====================================================
        # STRATEGY 2: BREAK-EVEN TRIGGER
        # ====================================================
        # If price moves 0.8R in favor -> Alert to move to Entry
        if await self.policy.should_send(trade_id, "MOVE_TO_BE"):
            should_alert = False
            
            if direction == 'LONG' and current_price >= (entry + (risk_unit * 0.8)):
                should_alert = True
            elif direction == 'SHORT' and current_price <= (entry - (risk_unit * 0.8)):
                should_alert = True
                
            if should_alert:
                msg = f"🔒 <b>SECURE RISK: {symbol}</b>\nPrice moved 0.8R in profit.\nMove Stop Loss to Entry <code>{entry:.4f}</code>."
                await self.telegram.send_update(trade_id, msg)
                await self.policy.mark_sent(trade_id, "MOVE_TO_BE")

        # ====================================================
        # STRATEGY 3: CHANDELIER TRAILING STOP (Trend Capture)
        # ====================================================
        # Trail behind the peak by 2.5R. If price crosses, Exit.
        if await self.policy.should_send(trade_id, "EXIT"):
            trail_hit = False
            trail_price = 0.0
            
            if direction == 'LONG':
                # Stop is Highest High minus 2.5x Risk
                trail_price = state['highest'] - (risk_unit * 2.5)
                # Only activate trailing if we are actively in profit (above entry)
                if current_price < trail_price and current_price > entry:
                    trail_hit = True
            else: # SHORT
                # Stop is Lowest Low plus 2.5x Risk
                trail_price = state['lowest'] + (risk_unit * 2.5)
                if current_price > trail_price and current_price < entry:
                    trail_hit = True
            
            if trail_hit:
                msg = f"📉 <b>TRAILING STOP HIT: {symbol}</b>\nTrend reversal detected at <code>{current_price:.4f}</code>.\nLocking in profit."
                await self.telegram.send_update(trade_id, msg)
                await self.policy.mark_sent(trade_id, "EXIT")
                await self.db.deactivate_trade(trade_id)
                # FIX: Remove from state tracking immediately
                if trade_id in self.trade_state:
                    del self.trade_state[trade_id]
                return

        # ====================================================
        # STRATEGY 4: ORDER FLOW FLIP (Early Warning)
        # ====================================================
        if await self.policy.should_send(trade_id, "FLOW_WARNING"):
            # Only check if hovering near entry (within 0.5% drift)
            if abs(current_price - entry) / entry < 0.005:
                of_data = await self.order_flow.analyze(symbol)
                warning = None
                
                if direction == 'LONG' and of_data['imbalance_score'] < 20: 
                    warning = f"⚠️ <b>FLOW ALERT: {symbol}</b>\nAggressive selling detected near entry."
                elif direction == 'SHORT' and of_data['imbalance_score'] > 80:
                    warning = f"⚠️ <b>FLOW ALERT: {symbol}</b>\nAggressive buying detected near entry."
                
                if warning:
                    await self.telegram.send_update(trade_id, warning)
                    await self.policy.mark_sent(trade_id, "FLOW_WARNING")

        # ====================================================
        # STRATEGY 5: HARD TP/SL (Safety Net)
        # ====================================================
        if await self.policy.should_send(trade_id, "EXIT"):
            exit_msg = None
            if direction == 'LONG':
                if current_price >= tp: 
                    exit_msg = f"✅ <b>TARGET SMASHED: {symbol}</b>\nPrice: {current_price:.4f}"
                elif current_price <= stop: 
                    exit_msg = f"🛑 <b>STOP LOSS: {symbol}</b>\nPrice: {current_price:.4f}"
            else:
                if current_price <= tp: 
                    exit_msg = f"✅ <b>TARGET SMASHED: {symbol}</b>\nPrice: {current_price:.4f}"
                elif current_price >= stop: 
                    exit_msg = f"🛑 <b>STOP LOSS: {symbol}</b>\nPrice: {current_price:.4f}"
            
            if exit_msg:
                await self.telegram.send_update(trade_id, exit_msg)
                await self.policy.mark_sent(trade_id, "EXIT")
                await self.db.deactivate_trade(trade_id)
                # FIX: Remove from state tracking immediately
                if trade_id in self.trade_state:
                    del self.trade_state[trade_id]
