import logging
from datetime import datetime, timedelta

class CircuitBreaker:
    """
    Prevents catastrophic loss by halting trading if thresholds are hit.
    """
    def __init__(self, max_drawdown_pct=0.10, max_losses_per_hour=3):
        self.logger = logging.getLogger("CircuitBreaker")
        self.max_dd = max_drawdown_pct
        self.max_losses_1h = max_losses_per_hour
        
        self.starting_balance = 0
        self.current_balance = 0
        self.recent_losses = [] 
        self.is_tripped = False

    def initialize(self, balance):
        self.starting_balance = balance
        self.current_balance = balance

    def record_pnl(self, pnl_amount):
        """Call this whenever a trade closes"""
        self.current_balance += pnl_amount
        
        if pnl_amount < 0:
            self._record_loss()
            
        self._check_drawdown()

    def _record_loss(self):
        now = datetime.now()
        # Remove losses older than 1 hour
        self.recent_losses = [t for t in self.recent_losses if t > now - timedelta(hours=1)]
        self.recent_losses.append(now)
        
        if len(self.recent_losses) >= self.max_losses_1h:
            self._trip(f"Velocity Limit: {len(self.recent_losses)} losses in 1 hour.")

    def _check_drawdown(self):
        if self.starting_balance <= 0: return
        
        dd = (self.starting_balance - self.current_balance) / self.starting_balance
        if dd >= self.max_dd:
            self._trip(f"Max Drawdown Limit: {dd*100:.2f}% loss.")

    def _trip(self, reason):
        if not self.is_tripped:
            self.is_tripped = True
            self.logger.critical(f"🔥 CIRCUIT BREAKER TRIPPED: {reason}")
            self.logger.critical("⛔ SYSTEM HALTED. MANUAL INTERVENTION REQUIRED.")

    def can_trade(self):
        return not self.is_tripped