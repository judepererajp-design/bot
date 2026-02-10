"""
TITAN-X TELEGRAM MESSAGE FORMATTER (ULTIMATE EDITION)
------------------------------------------------------------------------------
The Presentation Layer for the Titan-X Institutional Engine.
Handles formatting for:
1. Trade Signals (with Scaling Plans & Kelly Sizing)
2. Institutional Scorecards (6-Factor Breakdown)
3. Live Dashboards (Regime & Stats)
4. Alerts (Management & Stalker)
"""

from typing import Dict, Any, List
from datetime import datetime

class TelegramFormatter:
    """
    Renders complex trading data into clean, emoji-rich Telegram HTML.
    """
    
    @staticmethod
    def format_signal(signal: Dict[str, Any]) -> str:
        """
        Renders a full Trade Signal Card.
        """
        # --- 1. HEADER DATA ---
        symbol = signal.get('symbol', 'UNKNOWN').replace(':USDT', '')
        direction = signal.get('direction', 'FLAT')
        timeframe = signal.get('timeframe', 'M15')
        pattern = signal.get('pattern_name', 'Unknown Pattern')
        
        # --- 2. PRICE DATA ---
        entry = float(signal.get('entry', signal.get('entry_price', 0)))  # Handle both keys
        stop = float(signal.get('stop', signal.get('stop_loss', 0)))      # Handle both keys
        
        # CORRECTED: Calculate Stop Distance % based on direction
        if entry > 0:
            if direction == 'LONG':
                # For LONG: Stop is below entry
                stop_dist_pct = ((stop - entry) / entry) * 100  # Negative value
            else:  # SHORT
                # For SHORT: Stop is above entry
                stop_dist_pct = ((stop - entry) / entry) * 100  # Positive value
        else:
            stop_dist_pct = 0.0

        # --- 3. PLAN DATA (Optimizers) ---
        plan = signal.get('plan', {})
        scaling = plan.get('scaling_plan', {})
        size_str = plan.get('sizing_method', 'Standard 1.0%')
        rr_ratio = plan.get('risk_reward_ratio', 0.0)
        
        # --- 4. VISUAL LOGIC ---
        if direction == 'LONG':
            header_icon = "🟢"
            stop_icon = "🔻"  # Stop is below entry (negative distance)
            target_icon = "🎯"
        else:
            header_icon = "🔴"
            stop_icon = "🔺"  # Stop is above entry (positive distance)
            target_icon = "🎯"

        # Format Prices (Strip trailing zeros)
        def fmt(n): 
            return f"{float(n):.8f}".rstrip('0').rstrip('.')

        # --- 5. BUILD MAIN BODY ---
        lines = []
        
        # Title
        lines.append(f"<b>───────────────</b>")
        lines.append(f"<b>{header_icon} {direction} | {symbol}</b>")
        lines.append(f"⏰ {timeframe} • 🎯 {pattern}")
        lines.append(f"<b>───────────────</b>")
        
        # Entry & Stop (FIXED: Show correct distance direction)
        lines.append(f"📈 <b>Entry:</b> <code>{fmt(entry)}</code>")
        lines.append(f"🛡️  <b>Stop:</b>  <code>{fmt(stop)}</code> ({stop_icon}{abs(stop_dist_pct):.2f}%)")
        
        # Scaling Targets (The Pro Part)
        if scaling:
            lines.append(f"\n🎯 <b>Scaling Targets:</b>")
            if 'tp1' in scaling:
                lines.append(f"   1️⃣ <code>{fmt(scaling['tp1'])}</code> ({scaling.get('desc_1', 'Target 1')})")
            if 'tp2' in scaling:
                lines.append(f"   2️⃣ <code>{fmt(scaling['tp2'])}</code> ({scaling.get('desc_2', 'Target 2')})")
            if 'tp3' in scaling:
                lines.append(f"   3️⃣ <code>{fmt(scaling['tp3'])}</code> ({scaling.get('desc_3', 'Runner')})")
        else:
            # Fallback for old/simple signals
            tp = float(signal.get('tp', signal.get('take_profit', 0)))  # Handle both keys
            if tp > 0:
                lines.append(f"🎯 <b>Target:</b> <code>{fmt(tp)}</code>")

        # Risk & Sizing
        if rr_ratio > 0:
            lines.append(f"")
            lines.append(f"📊 <b>R:R:</b> {rr_ratio:.2f}")
        if size_str:
            lines.append(f"💰 <b>Size:</b> {size_str}")

        # --- 6. SCORECARD SECTION ---
        sc = signal.get('scorecard', {})
        final_score = sc.get('final_score', 50)
        
        # Score Emoji
        if final_score >= 80: s_emoji = "💎"
        elif final_score >= 65: s_emoji = "⭐"
        elif final_score >= 50: s_emoji = "📊"
        else: s_emoji = "⚠️"
        
        lines.append(f"{s_emoji} <b>Score:</b> {final_score}/100")
        
        # Institutional Breakdown (If available)
        comps = sc.get('components', {})
        if comps:
            lines.append(f"\n<b>Institutional Breakdown:</b>")
            # We map the internal keys to nice display names
            key_mapping = {
                'technical':   '📐 Technical',
                'volume':      '🌊 Volume',
                'orderflow':   '⚡ Orderflow',
                'correlation': '🔗 Beta/Corr',
                'sentiment':   '🤖 Sentiment',
                'derivatives': '📊 Funding'
            }
            
            for k, v in comps.items():
                display_name = key_mapping.get(k, k.title())
                # Add check/cross based on score
                mark = "✅" if v >= 60 else "⚠️" if v >= 40 else "❌"
                lines.append(f"{mark} {display_name}: <b>{v}</b>")

        # --- 7. FOOTER ---
        trade_id = signal.get('trade_id', '')
        if trade_id:
            lines.append(f"\n🔑 <b>ID:</b> <code>{trade_id[:8]}</code>")

        return "\n".join(lines)

    @staticmethod
    def format_dashboard(stats: Dict[str, Any]) -> str:
        """
        Renders the Live System Dashboard.
        """
        # Regime Formatting
        regime_raw = stats.get('market_regime', 'UNKNOWN')
        regime_map = {
            'STRONG_UPTREND':   '🚀 Strong Uptrend (Aggressive)',
            'WEAK_UPTREND':     '📈 Weak Uptrend (Normal)',
            'STRONG_DOWNTREND': '🩸 Strong Downtrend (Aggressive)',
            'WEAK_DOWNTREND':   '📉 Weak Downtrend (Normal)',
            'RANGING':          '🦀 Ranging (Defensive)',
            'HIGH_VOLATILITY':  '🌪️ High Volatility (Reduced Size)',
            'LOW_VOLATILITY':   '💤 Low Volatility (Accumulation)'
        }
        regime_display = regime_map.get(regime_raw, regime_raw)

        # Uptime
        uptime = stats.get('uptime', '0s')

        # Counts
        signals = stats.get('signals_today', 0)
        active = stats.get('active_trades', 0)
        watching = stats.get('watchlist_count', 0)

        # Build
        dash = (
            f"<b>🤖 TITAN-X INSTITUTIONAL ENGINE</b>\n"
            f"<b>───────────────────────</b>\n"
            f"🌍 <b>Market Regime:</b>\n"
            f"   {regime_display}\n\n"
            
            f"<b>📊 LIVE SESSION STATS</b>\n"
            f"📨 Signals Generated: <b>{signals}</b>\n"
            f"📌 Active Positions: <b>{active}</b>\n"
            f"👀 Stalker Watchlist: <b>{watching}</b>\n\n"
            
            f"<b>⚙️ SYSTEM HEALTH</b>\n"
            f"✅ Scanner Active\n"
            f"✅ Risk Engine Online\n"
            f"⏱️ Uptime: {uptime}\n"
            f"\n<i>Waiting for high-probability institutional setups...</i>"
        )
        return dash

    @staticmethod
    def format_alert(alert_type: str, symbol: str, data: Dict[str, Any] = None) -> str:
        """
        Renders formatted Alerts for Trade Management & Stalker.
        """
        symbol = symbol.replace(':USDT', '')
        data = data or {}
        msg_text = data.get('message', '')
        
        # Color/Emoji Templates
        templates = {
            # --- Trade Manager Alerts ---
            'flow_warning': {
                'emoji': '⚠️', 'color': '🟡', 'title': 'ORDER FLOW WARNING'
            },
            'breakeven': {
                'emoji': '🔒', 'color': '🟢', 'title': 'RISK SECURED'
            },
            'chandelier_exit': {
                'emoji': '📉', 'color': '🟠', 'title': 'TRAILING STOP HIT'
            },
            'time_exit': {
                'emoji': '⏳', 'color': '🔵', 'title': 'TIME EXIT (DEAD MONEY)'
            },
            'take_profit': {
                'emoji': '✅', 'color': '🟢', 'title': 'TARGET SMASHED'
            },
            'stop_loss': {
                'emoji': '🛑', 'color': '🔴', 'title': 'STOP LOSS HIT'
            },
            
            # --- Stalker Alerts ---
            'watchlist_new': {
                'emoji': '👀', 'color': '⚪', 'title': 'ADDED TO WATCHLIST'
            },
            'watchlist_alert': {
                'emoji': '🚨', 'color': '🔴', 'title': 'SETUP TRIGGERING'
            },
            
            # --- System Alerts ---
            'trade_tracked': {
                'emoji': '📌', 'color': '⚪', 'title': 'TRACKING STARTED'
            },
            'trade_ignored': {
                'emoji': '🗑️', 'color': '⚪', 'title': 'SIGNAL IGNORED'
            }
        }

        # Fallback
        t = templates.get(alert_type, {'emoji': 'ℹ️', 'color': '⚪', 'title': 'ALERT'})
        
        # Construct
        formatted = (
            f"<b>{t['color']} {t['title']}</b>\n"
            f"<b>{t['emoji']} {symbol}</b>\n"
            f"────────────────\n"
            f"{msg_text}"
        )
        
        return formatted

    @staticmethod
    def format_compact_signal(signal: Dict[str, Any]) -> str:
        """
        Minimalist format for high-frequency logs.
        """
        sym = signal.get('symbol', '').split('/')[0]
        dire = "LONG" if signal.get('direction') == 'LONG' else "SHRT"
        icon = "🟢" if dire == "LONG" else "🔴"
        entry = float(signal.get('entry', signal.get('entry_price', 0)))
        
        return f"{icon} {dire} {sym} @ {entry:.4f} | {signal.get('pattern_name')}"
