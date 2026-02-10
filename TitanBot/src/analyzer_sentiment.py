"""
TITAN-X SENTIMENT ENGINE (LLM POWERED)
------------------------------------------------------------------------------
Uses Google Gemini AI to analyze market sentiment.
"""

import logging
import os
import asyncio
import aiohttp
import json
from typing import Dict, Any
from datetime import datetime, timedelta

class SentimentEngine:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("gemini_api_key", os.getenv("GEMINI_API_KEY"))
        self.logger = logging.getLogger("SentimentLLM")
        self.enabled = False
        self.timeout = 10
        self.cache = {}
        self.cache_duration = 3600
        self.model = None
        self.client = None
        
        # Check if API key exists
        if not self.api_key:
            self.logger.warning("⚠️ No Gemini API Key found. Sentiment Analysis Disabled.")
            return
        
        # Try to import and initialize Gemini
        try:
            # Try NEW library first (google-genai)
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
            self.enabled = True
            self.library_version = "new"
            self.logger.info("✅ Gemini AI Connected (New Library v2)")
        except ImportError:
            try:
                # Try OLD library (google-generativeai)
                import google.generativeai as genai_old
                genai_old.configure(api_key=self.api_key)
                self.model = genai_old.GenerativeModel('gemini-pro')
                self.enabled = True
                self.library_version = "old"
                self.logger.warning("⚠️ Using deprecated Gemini library. Consider updating to 'google-genai'")
            except ImportError:
                self.logger.warning("⚠️ No Gemini library found. Install: 'pip install google-genai'")
            except Exception as e:
                self.logger.error(f"❌ Gemini initialization failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Gemini initialization failed: {e}")

    async def analyze(self, symbol: str, timeframe: str, price_action_summary: str) -> Dict[str, Any]:
        """
        Analyzes sentiment using Gemini AI.
        Falls back to simple scoring if API is unavailable.
        """
        if not self.enabled:
            return self._simple_sentiment(symbol, timeframe, price_action_summary)
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{hash(price_action_summary[:100])}"
        cached = self._get_from_cache(cache_key)
        if cached:
            cached['cached'] = True
            return cached
        
        # Only run for high-confidence signals (save API credits)
        if len(price_action_summary) < 30:
            return self._simple_sentiment(symbol, timeframe, price_action_summary)
        
        try:
            # Use async HTTP call for better performance
            score, reason = await self._async_gemini_call(symbol, timeframe, price_action_summary)
            
            result = {
                'sentiment_score': score,
                'reasoning': reason,
                'cached': False,
                'source': 'gemini_api'
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Gemini timeout for {symbol}")
            return self._simple_sentiment(symbol, timeframe, price_action_summary)
        except Exception as e:
            self.logger.error(f"Gemini API Error: {e}")
            return self._simple_sentiment(symbol, timeframe, price_action_summary)
    
    async def _async_gemini_call(self, symbol: str, timeframe: str, summary: str):
        """
        Async implementation using aiohttp.
        """
        prompt = (
            f"Act as a hedge fund analyst. Analyze {symbol} on {timeframe} timeframe.\n"
            f"Context: {summary}\n"
            f"Task: Provide a Sentiment Score (0=Very Bearish, 100=Very Bullish) and a 1-sentence reason.\n"
            f"Format: Score|Reason\n"
            f"Example: 75|Strong momentum with volume support suggests bullish continuation."
            f"Example: 35|Weak volume on breakout suggests false move."
        )
        
        # Always use HTTP API for consistency
        return await self._http_gemini_call(prompt)
    
    async def _http_gemini_call(self, prompt: str):
        """Direct HTTP call to Gemini API."""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": 100,
                "temperature": 0.2
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                params=params, 
                headers=headers, 
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                    
                    if "|" in text:
                        score_str, reason = text.split("|", 1)
                        try:
                            score = float(score_str.strip())
                            # Clamp between 0-100
                            score = max(0, min(100, score))
                            return score, reason.strip()
                        except ValueError:
                            return 50, "Invalid score format"
                    else:
                        # Try to extract score from text
                        import re
                        numbers = re.findall(r'\b\d{1,3}\b', text)
                        if numbers:
                            try:
                                score = float(numbers[0])
                                score = max(0, min(100, score))
                                return score, text[:100]
                            except:
                                return 50, text[:100] if text else "No analysis"
                        return 50, text[:100] if text else "No analysis"
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
    
    def _simple_sentiment(self, symbol: str, timeframe: str, summary: str) -> Dict[str, Any]:
        """Fallback sentiment analysis without API."""
        # Simple rule-based sentiment
        score = 50.0  # Neutral baseline
        
        # Basic keyword detection
        summary_lower = summary.lower()
        
        bullish_keywords = ['bullish', 'breakout', 'strong', 'momentum', 'volume', 'support', 'bounce']
        bearish_keywords = ['bearish', 'reversal', 'weak', 'rejection', 'resistance', 'breakdown']
        
        bullish_count = sum(1 for word in bullish_keywords if word in summary_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in summary_lower)
        
        if bullish_count > bearish_count:
            score = 60.0 + (bullish_count * 5)
        elif bearish_count > bullish_count:
            score = 40.0 - (bearish_count * 5)
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Generate simple reasoning
        if score >= 70:
            reasoning = "Bullish bias detected in pattern"
        elif score >= 60:
            reasoning = "Slightly bullish characteristics"
        elif score <= 30:
            reasoning = "Bearish bias detected in pattern"
        elif score <= 40:
            reasoning = "Slightly bearish characteristics"
        else:
            reasoning = "Neutral market sentiment"
        
        return {
            'sentiment_score': score,
            'reasoning': reasoning,
            'cached': False,
            'source': 'rule_based'
        }
    
    def _get_from_cache(self, key: str):
        """Get cached result if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_duration):
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Add result to cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries
        if len(self.cache) > 100:
            # Remove oldest 20 entries
            sorted_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['timestamp'])
            for k in sorted_keys[:20]:
                del self.cache[k]