"""
NyayaSathi AI — LLM Router
============================
Routes generation requests to either:
  • **Fast Mode**   → Groq cloud API  (low-latency, needs GROQ_API_KEY)
  • **Private Mode** → Ollama local    (data never leaves the machine)

Architecture note:
  The router is intentionally thin — no prompt-engineering lives here.
  Callers build their own prompts and pass them in; the router just
  handles transport, retries, and error wrapping.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests

from utils.logger import get_logger
import config

logger = get_logger(__name__)


class LLMRouter:
    """Unified interface for routing prompts to Groq or Ollama."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ) -> None:
        # Resolve credentials: caller-provided → env-var / config defaults
        resolved_key = groq_api_key or config.GROQ_API_KEY
        resolved_ollama = ollama_base_url or config.OLLAMA_BASE_URL

        # Groq setup — lazy; only fail when actually called in fast mode.
        self._groq_client = None
        if resolved_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=resolved_key)
                logger.info("Groq client initialised (model: %s)", config.GROQ_MODEL)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not initialise Groq client: %s", exc)

        # Ollama — just store the base URL; connectivity checked on demand.
        self._ollama_url = resolved_ollama.rstrip("/")
        logger.info("Ollama base URL: %s (model: %s)", self._ollama_url, config.OLLAMA_MODEL)

    # ── Public API ────────────────────────────────────────────────────

    def generate_response(
        self,
        prompt: str,
        mode: str = "fast",
        system_prompt: Optional[str] = None,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate a text completion.

        Parameters
        ----------
        prompt : str
            The user / task prompt.
        mode : str
            ``"fast"`` → Groq  |  ``"private"`` → Ollama.
        system_prompt : str | None
            Optional system-level instruction prepended to the conversation.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        str
            Model-generated text.

        Raises
        ------
        RuntimeError
            On network, timeout, or API errors.
        """
        mode = mode.lower().strip()
        if mode == "fast":
            return self._call_groq(prompt, system_prompt, temperature, max_tokens)
        elif mode == "private":
            return self._call_ollama(prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown LLM mode: '{mode}'. Use 'fast' or 'private'.")

    def is_groq_available(self) -> bool:
        """Check whether the Groq client is initialised."""
        return self._groq_client is not None

    def is_ollama_available(self) -> bool:
        """Ping the Ollama server health endpoint."""
        try:
            resp = requests.get(f"{self._ollama_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ── Private: Groq ─────────────────────────────────────────────────

    def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self._groq_client is None:
            raise RuntimeError(
                "Groq API key not set. Export GROQ_API_KEY or switch to Private mode."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=config.GROQ_TIMEOUT,
            )
            elapsed = time.time() - start
            text = response.choices[0].message.content.strip()
            logger.info("Groq response in %.1fs (%d chars)", elapsed, len(text))
            return text

        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

    # ── Private: Ollama ───────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{self._ollama_url}/api/generate"
        payload: dict = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            start = time.time()
            resp = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
            resp.raise_for_status()
            elapsed = time.time() - start
            data = resp.json()
            text = data.get("response", "").strip()
            logger.info("Ollama response in %.1fs (%d chars)", elapsed, len(text))
            return text

        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {config.OLLAMA_TIMEOUT}s. "
                "Check if the model is loaded."
            )
        except requests.RequestException as exc:
            logger.error("Ollama request error: %s", exc)
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Ollama response parsing error: %s", exc)
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc
