"""
llm_client.py - LLM API Client for Running Form Analysis Reports
Supports DeepSeek, OpenAI, and compatible APIs.
Configure via environment variables or direct constructor args.
"""

import os
import json
from typing import Optional, Dict, List, Callable
import urllib.request
import urllib.error
import ssl


# Default config
DEFAULT_CONFIG = {
    # DeepSeek
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    # OpenAI
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


class LLMClient:
    """
    Lightweight LLM client for running form analysis reports.
    No extra dependencies - uses urllib directly.
    """

    def __init__(self, provider: str = "deepseek",
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2048):
        """
        Args:
            provider: "deepseek" or "openai"
            api_key: API key (defaults to env var DEEPSEEK_API_KEY or OPENAI_API_KEY)
            model: Model name (defaults to provider default)
            base_url: API base URL (defaults to provider default)
            temperature: Response creativity (0.0-1.0, lower = more precise)
            max_tokens: Max response tokens
        """
        config = DEFAULT_CONFIG.get(provider, DEFAULT_CONFIG["deepseek"])
        self.provider = provider
        self.api_key = api_key or os.environ.get(config["api_key_env"], "")
        self.model = model or config["model"]
        self.base_url = (base_url or config["base_url"]).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Detect if running inside OpenClaw (use its model)
        self._use_openclaw_model = False
        self._openclaw_session_key = None

    @property
    def is_configured(self) -> bool:
        """Check if API key is available."""
        return bool(self.api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text."""
        if not self.api_key:
            raise ValueError(
                f"API key not configured. Set {DEFAULT_CONFIG[self.provider]['api_key_env']} "
                f"environment variable or pass api_key to LLMClient."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            # Create SSL context that handles modern certs
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {e.code}: {body[:500]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"API response parse error: {e}")

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        """Make the client callable - compatible with AIRunningCoach.llm_api_func."""
        return self.chat(system_prompt, user_prompt)


def create_llm_client(provider: str = "deepseek",
                      auto_fallback: bool = True) -> Optional[LLMClient]:
    """
    Create an LLM client if API key is available.
    Falls back to None if no key and auto_fallback is True.

    Args:
        provider: "deepseek" or "openai"
        auto_fallback: Return None if no API key configured

    Returns:
        LLMClient or None
    """
    client = LLMClient(provider=provider)
    if client.is_configured:
        print(f"  ✅ LLM configured: {provider}/{client.model}")
        return client
    elif auto_fallback:
        print(f"  ⚠️  No {provider} API key found. Using template report.")
        print(f"     Set {DEFAULT_CONFIG[provider]['api_key_env']}=your_key to enable AI reports.")
        return None
    else:
        raise ValueError(
            f"API key not configured. Set {DEFAULT_CONFIG[provider]['api_key_env']} "
            f"environment variable."
        )
