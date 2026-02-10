"""
TITAN-X TELEGRAM INTERFACE (FULL SUITE)
------------------------------------------------------------------------------
Features:
1. Professional UI (Formatter Integration)
2. Interactive Signal Management (Track/Ignore)
3. Dynamic Stalker Watchlist (/watchlist)
4. Live System Dashboard
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, 
    CommandHandler, 
    CallbackQueryHandler, 
    ContextTypes
)

# Import the Pro Formatter
try:
    from .telegram_formatter import TelegramFormatter
except ImportError:
    # Fallback if formatter not found yet
    class TelegramFormatter:
        def format_signal(self, s): return str(s)
        def format_dashboard(self, s): return "Dashboard Active"

class TelegramInterface:
    def __init__(self, config: Dict[str, Any]):
        self.token = config.get('bot_token', '')
        self.chat_id = config.get('chat_id', '')
        self.enabled = bool(self.token and self.chat_id)
        
        self.logger = logging.getLogger("Telegram")
        self.app = None
        self.engine_ref = None
        
        # UI & State
        self.formatter = TelegramFormatter()
        self.dashboard_message_id = None
        self.last_dashboard_update = datetime.min
        self.session_start = datetime.now()
        
        # Local Stats (reset on restart)
        self.stats = {
            'signals_sent': 0,
            'alerts_sent': 0
        }

    def set_engine(self, engine):
        self.engine_ref = engine

    async def start(self):
        """Boot the Telegram Bot with better error handling."""
        if not self.enabled: 
            self.logger.warning("Telegram disabled - no bot token or chat ID")
            return

        try:
            self.logger.info("Starting Telegram bot...")
            
            # Add timeout and retry logic
            self.app = Application.builder().token(self.token).build()
            
            # --- COMMANDS ---
            self.app.add_handler(CommandHandler("start", self._cmd_dashboard))
            self.app.add_handler(CommandHandler("dashboard", self._cmd_dashboard))
            self.app.add_handler(CommandHandler("status", self._cmd_dashboard))
            self.app.add_handler(CommandHandler("watchlist", self._cmd_watchlist))
            self.app.add_handler(CommandHandler("help", self._cmd_help))

            # --- INTERACTIVITY ---
            self.app.add_handler(CallbackQueryHandler(self._handle_button))
            
            await self.app.initialize()
            await self.app.start()
            
            # Start polling with error handling
            try:
                await self.app.updater.start_polling(
                    poll_interval=1.0,
                    timeout=10,
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES
                )
            except Exception as poll_error:
                self.logger.error(f"Polling error: {poll_error}")
                # Try with longer timeout
                await asyncio.sleep(5)
                await self.app.updater.start_polling(
                    poll_interval=2.0,
                    timeout=30,
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES
                )
            
            self.logger.info("✅ Telegram UI Active")
            
            # Send/Pin Dashboard on Boot (with retry)
            await self._retry_update_dashboard()

        except Exception as e:
            self.logger.error(f"Telegram Boot Error: {e}")
            self.enabled = False
            
            # Try to restart after delay
            await asyncio.sleep(10)
            self.logger.info("Retrying Telegram connection...")
            await self._retry_start()

    async def _retry_start(self, max_retries=3):
        """Retry starting Telegram bot."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Telegram retry attempt {attempt + 1}/{max_retries}")
                await self.start()
                if self.enabled:
                    return
            except Exception as e:
                self.logger.error(f"Retry {attempt + 1} failed: {e}")
                await asyncio.sleep(5 * (attempt + 1))
        
        self.logger.warning("Telegram failed to start after all retries")

    async def _retry_update_dashboard(self, max_retries=3):
        """Retry dashboard update with backoff."""
        for attempt in range(max_retries):
            try:
                await self.update_dashboard(force=True)
                return
            except Exception as e:
                self.logger.warning(f"Dashboard update failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2 * (attempt + 1))

    async def stop(self):
        if self.app:
            await self.app.stop()
            await self.app.shutdown()

    # =========================================================================
    # 📤 SENDING METHODS
    # =========================================================================

    async def send_signal(self, signal: Dict[str, Any]):
        """Sends a formatted Trade Signal card."""
        if not self.enabled: return
        
        try:
            # 1. Format
            msg = self.formatter.format_signal(signal)
            trade_id = signal.get('trade_id', 'UNKNOWN')

            # 2. Buttons (Manual Interaction)
            keyboard = [
                [
                    InlineKeyboardButton("✅ Track / Enter", callback_data=f"TRACK:{trade_id}"),
                    InlineKeyboardButton("❌ Ignore", callback_data=f"IGNORE:{trade_id}")
                ],
                [
                    InlineKeyboardButton("📊 Chart", url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{signal['symbol'].replace('/','')}")
                ]
            ]
            
            # 3. Send
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML',
                reply_markup=InlineKeyboardMarkup(keyboard),
                disable_web_page_preview=True
            )
            
            self.stats['signals_sent'] += 1
            await self.update_dashboard() # Update stats immediately
            
        except Exception as e:
            self.logger.error(f"Signal Send Error: {e}")

    async def send_update(self, trade_id: str, message_text: str):
        """Sends an update for a tracked trade."""
        if not self.enabled: return
        # Simple alert wrapper
        await self._send_raw(message_text)

    async def update_dashboard(self, force=False):
        """Updates the pinned Live Dashboard."""
        if not self.enabled or not self.engine_ref: return
        
        now = datetime.now()
        # Rate limit: Max once per 2 minutes unless forced
        if not force and (now - self.last_dashboard_update).total_seconds() < 120:
            return

        try:
            # 1. Gather Stats
            db = self.engine_ref.db
            
            # Active Trades (Tracked)
            active_count = 0
            async with db.conn.execute("SELECT COUNT(*) FROM trade_state WHERE is_active=1") as cursor:
                row = await cursor.fetchone()
                active_count = row[0] if row else 0
            
            # Watchlist Count (Stalker)
            watchlist_count = len(self.engine_ref.stalker.stalked_coins) if hasattr(self.engine_ref, 'stalker') else 0

            # Market Regime
            regime = "UNKNOWN"
            if hasattr(self.engine_ref, 'regime_detector'):
                regime = self.engine_ref.regime_detector.current_regime.value

            stats_payload = {
                'signals_today': await db.get_today_signal_count(),
                'active_trades': active_count,
                'watchlist_count': watchlist_count, # <--- NEW
                'market_regime': regime,
                'uptime': str(now - self.session_start).split('.')[0],
                'win_rate': 0.0, # Placeholder
                'avg_rr': 0.0,
                'total_pnl': 0.0
            }

            # 2. Format
            dash_text = self.formatter.format_dashboard(stats_payload)
            
            # 3. Buttons
            kb = [[InlineKeyboardButton("🔄 Refresh Stats", callback_data="REFRESH_DASH")]]

            # 4. Edit or Send
            if self.dashboard_message_id:
                try:
                    await self.app.bot.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self.dashboard_message_id,
                        text=dash_text,
                        parse_mode='HTML',
                        reply_markup=InlineKeyboardMarkup(kb)
                    )
                except Exception:
                    self.dashboard_message_id = None # Message deleted, resend

            if not self.dashboard_message_id:
                msg = await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=dash_text,
                    parse_mode='HTML',
                    reply_markup=InlineKeyboardMarkup(kb)
                )
                self.dashboard_message_id = msg.message_id
                try:
                    await self.app.bot.pin_chat_message(self.chat_id, msg.message_id)
                except:
                    pass # Helper might not have pin rights

            self.last_dashboard_update = now

        except Exception as e:
            self.logger.error(f"Dashboard Error: {e}")

    # =========================================================================
    # 🎮 HANDLERS
    # =========================================================================

    async def _cmd_dashboard(self, update, context):
        await self.update_dashboard(force=True)

    async def _cmd_watchlist(self, update, context):
        """Shows the Stalker Engine's active watchlist."""
        if not self.engine_ref or not hasattr(self.engine_ref, 'stalker'):
            await update.message.reply_text("🕵️‍♂️ Stalker Engine not loaded.")
            return

        # Get formatted text from Stalker
        msg = self.engine_ref.stalker.get_watchlist_text()
        
        # If empty, just text. If full, maybe a button to charts?
        await update.message.reply_text(msg, parse_mode='HTML', disable_web_page_preview=True)

    async def _cmd_help(self, update, context):
        msg = (
            "<b>🤖 TITAN-X COMMANDS</b>\n"
            "/dashboard - View Live Stats\n"
            "/watchlist - View Stalker Candidates\n"
            "/status - System Health"
        )
        await update.message.reply_text(msg, parse_mode='HTML')

    async def _handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        data = query.data
        await query.answer()

        if data == "REFRESH_DASH":
            await self.update_dashboard(force=True)

        elif data.startswith("TRACK:"):
            # User wants to track this trade
            trade_id = data.split(":")[1]
            if self.engine_ref:
                await self.engine_ref.db.activate_trade(trade_id)
                
                # Remove buttons to prevent double-click
                await query.edit_message_reply_markup(reply_markup=None)
                await self.app.bot.send_message(
                    self.chat_id, 
                    f"📌 <b>Trade {trade_id[:6]} Tracked</b>\nYou will receive management updates.", 
                    parse_mode='HTML'
                )
                await self.update_dashboard(force=True)

        elif data.startswith("IGNORE:"):
            # User rejects trade
            trade_id = data.split(":")[1]
            if self.engine_ref:
                await self.engine_ref.db.deactivate_trade(trade_id)
                await query.edit_message_reply_markup(reply_markup=None)
                await self.app.bot.send_message(
                    self.chat_id, 
                    f"🚫 <b>Trade Ignored</b>", 
                    parse_mode='HTML'
                )

    async def _send_raw(self, text):
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='HTML')
        except:
            pass
