import aiosqlite
import logging
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

class AsyncDatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.db_path = config.get('path', 'data/titan.db')
        self.logger = logging.getLogger("Database")
        self.conn = None
        self.lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    async def connect(self):
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode=WAL;")
            await self.conn.execute("PRAGMA synchronous=NORMAL;")
            await self.conn.execute("PRAGMA busy_timeout=5000;")
            await self._init_schema()
            self.logger.info(f"✅ Database Connected: {self.db_path}")
        except Exception as e:
            self.logger.critical(f"❌ Database Connection Failed: {e}")
            raise

    async def _init_schema(self):
        # 1. Signals Table
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                pattern_name TEXT,
                timeframe TEXT,
                direction TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                timestamp REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. Create indexes for signals table
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_trade_id ON signals (trade_id)
        ''')
        
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals (symbol, timestamp)
        ''')
        
        # 3. Trade State Table
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trade_state (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                be_sent INTEGER DEFAULT 0,
                flow_warning_sent INTEGER DEFAULT 0,
                exit_sent INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 4. Create indexes for trade_state table
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_trade_state_active ON trade_state (is_active)
        ''')
        
        await self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_trade_state_trade_id ON trade_state (trade_id)
        ''')
        
        await self.conn.commit()
        self.logger.info("✅ Database schema initialized")

    # --- CORE METHODS ---

    async def save_signal(self, s: Dict[str, Any]) -> bool:
        """Saves signal and initializes trade state."""
        async with self.lock:
            trade_id = s.get('trade_id')
            if not trade_id:
                self.logger.error("No trade_id in signal")
                return False
            
            # Check if signal already exists
            async with self.conn.execute('SELECT 1 FROM signals WHERE trade_id = ?', (trade_id,)) as cursor:
                if await cursor.fetchone():
                    return False

            try:
                await self.conn.execute('BEGIN TRANSACTION')
                
                # Insert Signal
                await self.conn.execute('''
                    INSERT INTO signals 
                    (trade_id, symbol, pattern_name, timeframe, direction, 
                     entry_price, stop_loss, take_profit, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    s.get('symbol', ''),
                    s.get('pattern_name', ''),
                    s.get('timeframe', ''),
                    s.get('direction', ''),
                    s.get('entry', s.get('entry_price', 0)),
                    s.get('stop', s.get('stop_loss', 0)),
                    s.get('tp', s.get('take_profit', 0)),
                    s.get('confidence', 50.0),
                    s.get('timestamp', datetime.now().timestamp())
                ))
                
                # Initialize Trade State (INACTIVE by default)
                await self.conn.execute('''
                    INSERT OR IGNORE INTO trade_state (trade_id, symbol)
                    VALUES (?, ?)
                ''', (trade_id, s.get('symbol', '')))
                
                await self.conn.commit()
                return True
            except Exception as e:
                await self.conn.rollback()
                self.logger.error(f"Failed to save signal: {e}")
                return False

    # --- STATE MANAGEMENT METHODS ---

    async def activate_trade(self, trade_id: str):
        """User clicked 'Track'."""
        async with self.lock:
            await self.conn.execute(
                "UPDATE trade_state SET is_active=1 WHERE trade_id=?",
                (trade_id,)
            )
            await self.conn.commit()

    async def deactivate_trade(self, trade_id: str):
        """User clicked 'Ignore' or 'Stop Tracking'."""
        async with self.lock:
            await self.conn.execute(
                "UPDATE trade_state SET is_active=0 WHERE trade_id=?",
                (trade_id,)
            )
            await self.conn.commit()

    async def is_trade_active(self, trade_id: str) -> bool:
        async with self.lock:
            async with self.conn.execute(
                "SELECT is_active FROM trade_state WHERE trade_id=?",
                (trade_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row and row['is_active'] == 1

    async def can_send_message(self, trade_id: str, message_type: str) -> bool:
        """Checks if message type was already sent."""
        async with self.lock:
            column_map = {
                'MOVE_TO_BE': 'be_sent',
                'FLOW_WARNING': 'flow_warning_sent',
                'EXIT': 'exit_sent'
            }
            col = column_map.get(message_type)
            if not col:
                return False

            async with self.conn.execute(
                f"SELECT {col} FROM trade_state WHERE trade_id=?",
                (trade_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row and row[col] == 0

    async def mark_message_sent(self, trade_id: str, message_type: str):
        """Marks a message type as sent to prevent duplicates."""
        async with self.lock:
            column_map = {
                'MOVE_TO_BE': 'be_sent',
                'FLOW_WARNING': 'flow_warning_sent',
                'EXIT': 'exit_sent'
            }
            col = column_map.get(message_type)
            if not col:
                return

            await self.conn.execute(
                f"UPDATE trade_state SET {col}=1, updated_at=CURRENT_TIMESTAMP WHERE trade_id=?",
                (trade_id,)
            )
            await self.conn.commit()
    
    async def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Fetches active signals."""
        async with self.lock:
            query = '''
                SELECT s.* 
                FROM signals s
                JOIN trade_state t ON s.trade_id = t.trade_id
                WHERE t.is_active = 1
                ORDER BY s.timestamp DESC 
                LIMIT ?
            '''
            async with self.conn.execute(query, (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_today_signal_count(self) -> int:
        start_ts = datetime.utcnow().replace(hour=0, minute=0, second=0).timestamp()
        async with self.lock:
            async with self.conn.execute(
                "SELECT COUNT(*) FROM signals WHERE timestamp >= ?",
                (start_ts,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.logger.info("Database connection closed")