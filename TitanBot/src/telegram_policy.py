import logging

class TelegramPolicy:
    """
    GOVERNANCE LAYER
    ----------------
    Decides IF a message is allowed to be sent.
    Rules:
    1. Trade must be marked ACTIVE (User clicked 'Track').
    2. Message type must not have been sent before (Anti-Spam).
    """
    def __init__(self, db_manager):
        self.db = db_manager
        self.logger = logging.getLogger("TelegramPolicy")

    async def should_send(self, trade_id: str, message_type: str) -> bool:
        # 1. Check Active Status
        if not await self.db.is_trade_active(trade_id):
            return False
            
        # 2. Check Duplication
        return await self.db.can_send_message(trade_id, message_type)

    async def mark_sent(self, trade_id: str, message_type: str):
        await self.db.mark_message_sent(trade_id, message_type)