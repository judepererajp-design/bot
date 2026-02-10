"""
TITAN-X ENGINE: FULL INSTITUTIONAL SUITE (BUG-FIXED)
Original functionality with all features, just fixing the import/initialization bugs
"""

import asyncio
import logging
import signal
import sys
import time
import hashlib
import traceback
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Import everything - using absolute imports to avoid circular issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Now import everything
try:
    from src.config_loader import ConfigManager
    from src.database import AsyncDatabaseManager
    from src.api import ExchangeInterface
    from src.scanner import MarketScheduler
    from src.telegram_bot import TelegramInterface
    from src.health_server import HealthServer
    from src.stalker_engine import StalkerEngine
    from src.detector_geometric import GeometricPatternDetector
    from src.detector_harmonic import HarmonicPatternDetector
    from src.analysis_mtf import MultiTimeframeAnalyzer
    from src.analyzer_regime import MarketRegimeDetector
    from src.analyzer_volume import VolumeProfileAnalyzer
    from src.analyzer_orderflow import OrderFlowAnalyzer
    from src.analyzer_correlation import CorrelationAnalyzer
    from src.analyzer_sentiment import SentimentEngine
    from src.analyzer_derivatives import DerivativesAnalyzer
    from src.decision_engine import DecisionEngine
    from src.strategy_manager import StrategyManager
    from src.risk import RiskCalculator
    from src.position_optimizer import PositionOptimizer
    from src.profit_optimizer import ProfitOptimizer
    from src.trade_manager import TradeManager
    from src.telegram_policy import TelegramPolicy
    from src.circuit_breaker import CircuitBreaker
except ImportError as e:
    print(f"Import error - trying fallback: {e}")
    # Try relative imports
    from .config_loader import ConfigManager
    from .database import AsyncDatabaseManager
    from .api import ExchangeInterface
    from .scanner import MarketScheduler
    from .telegram_bot import TelegramInterface
    from .health_server import HealthServer
    from .stalker_engine import StalkerEngine
    from .detector_geometric import GeometricPatternDetector
    from .detector_harmonic import HarmonicPatternDetector
    from .analysis_mtf import MultiTimeframeAnalyzer
    from .analyzer_regime import MarketRegimeDetector
    from .analyzer_volume import VolumeProfileAnalyzer
    from .analyzer_orderflow import OrderFlowAnalyzer
    from .analyzer_correlation import CorrelationAnalyzer
    from .analyzer_sentiment import SentimentEngine
    from .analyzer_derivatives import DerivativesAnalyzer
    from .decision_engine import DecisionEngine
    from .strategy_manager import StrategyManager
    from .risk import RiskCalculator
    from .position_optimizer import PositionOptimizer
    from .profit_optimizer import ProfitOptimizer
    from .trade_manager import TradeManager
    from .telegram_policy import TelegramPolicy
    from .circuit_breaker import CircuitBreaker

class TitanEngine:
    def __init__(self):
        # Setup logging first
        self.logger = self._setup_logging()
        self.logger.info("🚀 TITAN-X INSTITUTIONAL ENGINE - FULL POWER MODE")
        
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Thread pool for CPU-intensive tasks
        workers = self.config.get('system', {}).get('cpu_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="TitanCalc")
        
        # Initialize ALL components (like original)
        self._init_components()
        
        # State
        self.is_running = False
        self.start_time = time.time()
        self.signals_generated = 0
        self.signals_sent = 0
        self.errors = 0
        
        # Queues for pipeline
        self.analysis_queue = asyncio.Queue(maxsize=1000)
        self.signal_queue = asyncio.Queue(maxsize=100)
        
    def _setup_logging(self):
        """Setup professional logging."""
        import colorlog
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s: %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            }
        ))
        logger = colorlog.getLogger("TitanEngine")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _init_components(self):
        """Initialize all engine components."""
        # Core infrastructure
        self.db = AsyncDatabaseManager(self.config.get('database', {}))
        self.api = ExchangeInterface(self.config.get('exchange', {}))
        self.telegram = TelegramInterface(self.config.get('telegram', {}))
        self.health_server = HealthServer(port=8080)
        self.scanner = MarketScheduler(self.config.get('scanning', {}))
        
        # Pattern detectors
        self.geo_detector = GeometricPatternDetector(self.config.get('patterns', {}))
        self.harm_detector = HarmonicPatternDetector(self.config.get('patterns', {}))
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        
        # Institutional analyzers
        self.regime_detector = MarketRegimeDetector(self.config.get('system', {}))
        self.volume_analyzer = VolumeProfileAnalyzer(self.config.get('volume_profile', {}))
        self.order_flow = OrderFlowAnalyzer(self.api)
        self.correlation = CorrelationAnalyzer(self.api)
        self.derivatives = DerivativesAnalyzer(self.api)
        self.sentiment = SentimentEngine(self.config.get('system', {}))
        
        # Risk & optimization
        self.position_optimizer = PositionOptimizer(self.config.get('risk', {}))
        self.profit_optimizer = ProfitOptimizer(self.config.get('risk', {}))
        self.risk_manager = RiskCalculator(
            self.config.get('risk', {}),
            optimizer=self.position_optimizer,
            profit_optimizer=self.profit_optimizer
        )
        
        # Decision engine
        self.decision_engine = DecisionEngine()
        
        # Strategy manager context
        strat_context = {
            'geo_detector': self.geo_detector,
            'harm_detector': self.harm_detector,
            'order_flow': self.order_flow,
            'correlation': self.correlation,
            'config': self.config
        }
        self.strategy_manager = StrategyManager(strat_context)
        
        # Trade manager
        trade_context = {
            'db': self.db,
            'api': self.api,
            'telegram': self.telegram,
            'order_flow': self.order_flow,
            'config': self.config
        }
        self.trade_manager = TradeManager(trade_context)
        
        # Stalker engine
        stalker_context = {
            'config': self.config.get('stalker', {}),
            'scanner': self.scanner,
            'api': self.api,
            'telegram': self.telegram
        }
        self.stalker = StalkerEngine(stalker_context)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            max_drawdown_pct=0.10,
            max_losses_per_hour=3
        )
        
        # Build shared context
        self.context = {
            'db': self.db,
            'api': self.api,
            'telegram': self.telegram,
            'geo_detector': self.geo_detector,
            'harm_detector': self.harm_detector,
            'order_flow': self.order_flow,
            'correlation': self.correlation,
            'derivatives': self.derivatives,
            'volume_analyzer': self.volume_analyzer,
            'sentiment': self.sentiment,
            'regime_detector': self.regime_detector,
            'strategy_manager': self.strategy_manager,
            'decision_engine': self.decision_engine,
            'risk_manager': self.risk_manager,
            'scanner': self.scanner,
            'stalker': self.stalker,
            'trade_manager': self.trade_manager,
            'circuit_breaker': self.circuit_breaker,
            'config': self.config
        }
        
        # Link engine to telegram
        self.telegram.set_engine(self)
        
    async def start(self):
        """Start the full institutional engine."""
        self.is_running = True
        
        try:
            self.logger.info("🔌 Connecting to all services...")
            
            # Connect to database
            await self.db.connect()
            
            # Connect to exchange
            await self.api.connect()
            
            # Start health server
            await self.health_server.start()
            
            # Start telegram
            await self.telegram.start()
            
            # Initialize market graph
            await self.correlation.update_macro_data()
            
            # Load tickers
            tickers = await self.api.fetch_active_tickers(
                min_volume=self.config['scanning']['min_volume'],
                max_count=self.config['scanning']['max_tickers']
            )
            self.scanner.initialize(tickers)
            
            self.logger.info(f"📈 Monitoring {len(tickers)} assets with full institutional analysis")
            self.logger.info("✅ All systems operational")
            
            # Start ALL background tasks
            background_tasks = [
                self._producer_loop(),
                self._consumer_loop(),
                self._dispatcher_loop(),
                self._stalker_loop(),
                self._trade_monitor_loop(),
                self._maintenance_loop(),
                self._dashboard_update_loop()
            ]
            
            await asyncio.gather(*background_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.critical(f"Engine failed to start: {e}")
            traceback.print_exc()
            await self.shutdown()
    
    async def _producer_loop(self):
        """Producer: Get symbols to analyze."""
        while self.is_running:
            try:
                tasks = self.scanner.get_due_tasks(limit=10)
                for task in tasks:
                    await self.analysis_queue.put(task)
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"Producer error: {e}")
                await asyncio.sleep(5)
    
    async def _consumer_loop(self):
        """Consumer: Full institutional analysis pipeline."""
        while self.is_running:
            try:
                task = await self.analysis_queue.get()
                symbol = task['symbol']
                
                # 1. Fetch multi-timeframe data
                ohlcv = await self.api.fetch_ohlcv_batch(symbol, ['15m', '1h', '4h'])
                if not ohlcv:
                    continue
                
                # 2. Market regime analysis
                regime_data = self.regime_detector.analyze(ohlcv['1h'])
                
                # 3. Run ALL strategies
                signals = await self.strategy_manager.run_analysis(symbol, ohlcv, regime_data)
                
                # 4. Institutional validation for each signal
                for signal in signals:
                    # Validate with MTF
                    if not self.mtf_analyzer.confirm_trend(signal, ohlcv):
                        continue
                    
                    # Volume profile
                    volume_data = self.volume_analyzer.analyze(ohlcv['1h'])
                    
                    # Order flow
                    order_flow_data = await self.order_flow.analyze(symbol)
                    
                    # Correlation
                    correlation_data = await self.correlation.analyze(symbol, ohlcv['1h'])
                    
                    # Derivatives
                    derivatives_data = await self.derivatives.analyze(symbol)
                    
                    # Sentiment (AI)
                    sentiment_data = await self.sentiment.analyze(
                        symbol, 
                        signal['timeframe'], 
                        f"{signal['pattern_name']} detected"
                    )
                    
                    # Risk calculation with Kelly sizing
                    signal['volume_profile'] = volume_data
                    risk_signal = self.risk_manager.calculate_risk(signal, ohlcv)
                    if not risk_signal:
                        continue
                    
                    # Institutional scoring
                    scorecard = self.decision_engine.generate_scorecard(
                        tech_signal=risk_signal,
                        volume_profile=volume_data,
                        order_flow=order_flow_data,
                        correlation=correlation_data,
                        sentiment=sentiment_data,
                        derivatives=derivatives_data
                    )
                    
                    # Filter by confidence
                    min_conf = self.config['system'].get('min_confidence', 70.0)
                    if scorecard['final_score'] >= min_conf:
                        risk_signal['scorecard'] = scorecard
                        risk_signal['trade_id'] = hashlib.sha256(
                            f"{symbol}{time.time()}".encode()
                        ).hexdigest()[:12]
                        
                        # Queue for dispatch
                        await self.signal_queue.put(risk_signal)
                        self.signals_generated += 1
                
                self.analysis_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Consumer error: {e}")
                self.analysis_queue.task_done()
    
    async def _dispatcher_loop(self):
        """Dispatch validated signals."""
        while self.is_running:
            try:
                signal = await self.signal_queue.get()
                
                # Save to database
                saved = await self.db.save_signal(signal)
                if saved:
                    # Send to telegram
                    await self.telegram.send_signal(signal)
                    
                    # Update stalker
                    if self.stalker:
                        self.stalker.notify_signal_generated(signal['symbol'])
                    
                    self.signals_sent += 1
                    
                    self.logger.info(
                        f"🚨 SIGNAL: {signal['symbol']} {signal['direction']} | "
                        f"Score: {signal['scorecard']['final_score']} | "
                        f"Pattern: {signal['pattern_name']}"
                    )
                
                self.signal_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Dispatcher error: {e}")
                self.signal_queue.task_done()
    
    async def _stalker_loop(self):
        """Run stalker engine."""
        if self.stalker:
            await self.stalker.run()
        else:
            await asyncio.sleep(300)  # Sleep if disabled
    
    async def _trade_monitor_loop(self):
        """Monitor active trades."""
        await self.trade_manager.run_loop()
    
    async def _maintenance_loop(self):
        """Hourly maintenance."""
        while self.is_running:
            await asyncio.sleep(3600)
            
            # Update macro data
            await self.correlation.update_macro_data()
            
            # Refresh tickers
            tickers = await self.api.fetch_active_tickers(
                min_volume=self.config['scanning']['min_volume'],
                max_count=self.config['scanning']['max_tickers']
            )
            self.scanner.update_universe(tickers)
            
            self.logger.info("✅ Hourly maintenance completed")
    
    async def _dashboard_update_loop(self):
        """Update dashboard periodically."""
        while self.is_running:
            await asyncio.sleep(120)  # Every 2 minutes
            try:
                await self.telegram.update_dashboard()
            except Exception as e:
                self.logger.debug(f"Dashboard update failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("🛑 Shutting down...")
        self.is_running = False
        
        # Shutdown all components
        tasks = []
        if self.api:
            tasks.append(self.api.close())
        if self.db:
            tasks.append(self.db.close())
        if self.telegram:
            tasks.append(self.telegram.stop())
        if self.health_server:
            tasks.append(self.health_server.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        if self.executor:
            self.executor.shutdown(wait=False)
        
        runtime = time.time() - self.start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        self.logger.info(f"📊 Session: {self.signals_sent} signals in {hours}h {minutes}m")
        self.logger.info("👋 Shutdown complete")
        sys.exit(0)

async def main():
    """Main entry point."""
    engine = TitanEngine()
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.shutdown()))
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.shutdown()
    except Exception as e:
        print(f"💀 Critical error: {e}")
        traceback.print_exc()
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())