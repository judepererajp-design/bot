from aiohttp import web
import logging
import asyncio

class HealthServer:
    def __init__(self, port=8080):
        self.port = port
        self.logger = logging.getLogger("HealthMonitor")
        self.app = web.Application()
        self.app.add_routes([
            web.get('/', self.handle_root),
            web.get('/health', self.handle_health),
        ])
        self.runner = None
        self.site = None
        
        # Shared State to track bot performance
        self.stats = {
            'status': 'booting',
            'signals_found': 0,
            'errors': 0
        }

    async def start(self):
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await self.site.start()
            self.stats['status'] = 'active'
            self.logger.info(f"🚑 Health Server listening on port {self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start Health Server: {e}")

    async def handle_root(self, request):
        return web.Response(text="Titan-X Institutional Engine is Running.")

    async def handle_health(self, request):
        # Returns JSON status
        return web.json_response(self.stats)

    async def stop(self):
        if self.site:
            await self.runner.cleanup()