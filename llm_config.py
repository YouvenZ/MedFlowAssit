"""
llm_config.py — Centralised LLM access via LiteLLM (Groq provider).

All agents call `llm_completion()` instead of touching Groq directly.
Switching provider later means changing ONE file.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from litellm import completion

load_dotenv()
import litellm
logger = logging.getLogger(__name__)
# litellm._turn_on_debug()
# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"       # LiteLLM prefix for Groq
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 1024

# Ensure the Groq key is available to LiteLLM
_groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_key")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key


def llm_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs: Any,
):
    """
    Thin wrapper around litellm.completion().

    Returns the full LiteLLM response object (OpenAI-compatible).
    """
    logger.debug("LiteLLM call  model=%s  msgs=%d  tools=%s",
                 model, len(messages), bool(tools))

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        call_kwargs["tools"] = tools

    call_kwargs.update(kwargs)

    response = completion(**call_kwargs)
    logger.debug("LiteLLM response  finish=%s",
                 response.choices[0].finish_reason if response.choices else "?")
    return response
