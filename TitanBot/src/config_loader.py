"""
TITAN-X CONFIG MANAGER (INSTITUTIONAL)
------------------------------------------------------------------------------
Handles loading, validation, and hot-reloading of configuration.
Supports YAML format with .env file overrides.
Enforces Strict Type Checking to prevent "Fat Finger" errors.
"""

import os
import yaml
import logging
import sys
from typing import Dict, Any, Union
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, config_path: str = "config/settings.yaml"):
        # 1. Load Environment Variables (Secure Keys)
        load_dotenv()
        
        # 2. Resolve Paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_path, config_path)
        self.logger = logging.getLogger("ConfigManager")
        self._config = {}

    def load_config(self) -> Dict[str, Any]:
        """Loads, merges, and validates the configuration."""
        if not os.path.exists(self.config_path):
            self.logger.critical(f"❌ Configuration file missing: {self.config_path}")
            raise SystemExit(1)

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # 3. Inject Secure Environment Variables
            self._inject_env_vars()
            
            # 4. Validate Types & Logic
            self._validate_schema()
            
            self.logger.info(f"✅ Configuration loaded from {os.path.basename(self.config_path)}")
            return self._config

        except Exception as e:
            self.logger.critical(f"❌ Config Load Error: {e}")
            raise SystemExit(1)

    def _inject_env_vars(self):
        """Overrides yaml config with secure .env variables and logs it."""
        overrides = {
            'TITAN_EXCHANGE_KEY': ('exchange', 'api_key'),
            'TITAN_EXCHANGE_SECRET': ('exchange', 'api_secret'),
            'TITAN_TELEGRAM_TOKEN': ('telegram', 'bot_token'),
            'TITAN_TELEGRAM_CHAT_ID': ('telegram', 'chat_id'),
            'TITAN_GEMINI_KEY': ('system', 'gemini_api_key'),
        }

        for env_var, (section, key) in overrides.items():
            val = os.getenv(env_var)
            if val:
                # Ensure section exists
                if section not in self._config:
                    self._config[section] = {}
                
                # Apply override
                self._config[section][key] = val
                # Mask keys in logs for security
                masked_val = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "***"
                self.logger.info(f"🔑 ENV Override: {section}.{key} set to {masked_val}")

    def _validate_schema(self):
        """Ensures the config file isn't garbage."""
        
        # A. Check Critical Sections
        required_sections = ['system', 'exchange', 'telegram', 'scanning', 'risk', 'patterns']
        for sec in required_sections:
            if sec not in self._config:
                self.logger.critical(f"❌ Missing critical config section: '{sec}'")
                raise SystemExit(1)

        # B. Type Enforcement & Safety Checks
        try:
            # Risk Checks
            risk = self._config['risk']
            risk['risk_per_trade'] = float(risk.get('risk_per_trade', 0.01))
            risk['account_size'] = float(risk.get('account_size', 1000))
            
            if risk['risk_per_trade'] > 0.05:
                self.logger.warning("⚠️  HIGH RISK SETTING: Risk per trade is > 5%")
            
            if risk['risk_per_trade'] <= 0:
                self.logger.critical("❌ Risk per trade must be positive.")
                raise ValueError("Invalid risk_per_trade")

            # System Checks
            sys_conf = self._config['system']
            sys_conf['cpu_workers'] = int(sys_conf.get('cpu_workers', 4))
            sys_conf['min_confidence'] = float(sys_conf.get('min_confidence', 70.0))

            # Exchange Checks
            if not self._config['exchange'].get('name'):
                self._config['exchange']['name'] = 'binance'

        except ValueError as e:
            self.logger.critical(f"❌ Configuration Type Error: {e}")
            raise SystemExit(1)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)