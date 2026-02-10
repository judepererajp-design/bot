import sys
import os
import asyncio
import logging
import warnings

# 1. HARD PATH FIX (Crucial for Mac/Linux)
# Forces Python to look in the current directory for modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 2. CONFIGURE BOOT LOGGING
# We set this up before importing anything else to catch import errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# --- SILENCE NOISY LIBRARIES ---
# This prevents the "HTTP Request: POST..." spam
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
# -------------------------------

boot_logger = logging.getLogger("BOOT")

# 3. ENVIRONMENT OPTIMIZATION
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- MAC OS COMPATIBILITY FIX ---
# We force standard asyncio because uvloop conflicts with httpx on Mac
try:
    # import uvloop
    # if hasattr(asyncio, 'set_event_loop_policy'):
    #     asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    #     boot_logger.info("🚀 High-Performance 'uvloop' activated")
    boot_logger.info("⚠️ uvloop disabled for macOS stability (fixes httpx error)")
except Exception as e:
    boot_logger.debug(f"uvloop skip: {e}")

# 4. IMPORT ENGINE (With Error Trapping)
try:
    # This import triggers the loading of all other src files
    from src.main import TitanEngine
except ImportError as e:
    boot_logger.critical(f"❌ IMPORT ERROR: {e}")
    boot_logger.critical("---------------------------------------------------")
    boot_logger.critical("CHECK FOLDER STRUCTURE:")
    boot_logger.critical(f"1. Root path: {current_dir}")
    boot_logger.critical("2. Does 'src' folder exist? (Must be lowercase)")
    boot_logger.critical("3. Does 'src/__init__.py' exist?")
    boot_logger.critical("---------------------------------------------------")
    raise SystemExit(1)

async def main():
    try:
        boot_logger.info("⚡ Initializing Titan-X Institutional Engine...")
        bot = TitanEngine()
        await bot.start()
    except asyncio.CancelledError:
        boot_logger.info("🛑 Task Cancelled. Shutting down.")
    except Exception as e:
        boot_logger.critical(f"💀 CRITICAL RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        boot_logger.info("👋 Manual Shutdown Received. Goodbye.")
        raise SystemExit(0)