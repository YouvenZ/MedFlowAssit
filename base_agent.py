"""
base_agent.py — Abstract base for every agent in the system.

Provides:
  • A shared tool-calling agentic loop  (agent_loop)
  • Message management helpers
  • Tool dispatch plumbing

Concrete subclasses only need to supply:
  _build_system_prompt()   → str
  _build_tool_schemas()    → list[dict]
  available_function(name) → callable
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from llm_config import llm_completion, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract agent with a reusable tool-calling loop."""

    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 12,
    ):
        self.name = name
        self.model = model
        self.max_iterations = max_iterations
        self.messages: list[dict] = []
        self.tools = self._build_tool_schemas()
        logger.info("[%s] initialised  model=%s  tools=%d",
                     self.name, self.model, len(self.tools))

    # ── abstract hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _build_system_prompt(self) -> str:
        ...

    @abstractmethod
    def _build_tool_schemas(self) -> list[dict]:
        ...

    @abstractmethod
    def available_function(self, function_name: str) -> Any:
        ...

    # ── message helpers ───────────────────────────────────────────────────────

    def _ensure_system_prompt(self):
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(
                0, {"role": "system", "content": self._build_system_prompt()}
            )

    def add_context_message(self, role: str, content: str):
        """Inject an extra message (e.g. handoff context from orchestrator)."""
        self.messages.append({"role": role, "content": content})

    def reset(self):
        """Wipe conversation history."""
        logger.info("[%s] conversation reset", self.name)
        self.messages.clear()

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _llm_call(self) -> Any:
        return llm_completion(
            messages=self.messages,
            tools=self.tools or None,
            model=self.model,
        )

    # ── tool execution ────────────────────────────────────────────────────────

    def _execute_tool_call(self, tool_call) -> Any:
        name = tool_call.function.name
        args_str = tool_call.function.arguments
        logger.info("[%s] tool=%s", self.name, name)
        logger.debug("[%s] args=%s", self.name, args_str[:300])

        args = json.loads(args_str)

        print(f"\n{'─'*60}")
        print(f"  [{self.name}] TOOL  : {name}")
        print(f"  ARGS  : {json.dumps(args, indent=4)}")

        try:
            fn = self.available_function(name)
            result = fn(**args)
        except Exception as exc:
            logger.error("[%s] tool %s failed: %s", self.name, name, exc, exc_info=True)
            result = {"status": "error", "message": str(exc), "tool": name}

        print(f"  RESULT: {json.dumps(result, indent=4) if isinstance(result, dict) else result}")
        print(f"{'─'*60}")
        return result

    # ── agentic loop ──────────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        """
        Execute a full tool-calling loop for a single user turn.

        Returns the final assistant text.
        """
        logger.info("[%s] ── run ──  input=%s", self.name, user_input[:120])

        # Build / append messages
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user",   "content": user_input},
            ]
        else:
            self._ensure_system_prompt()
            self.messages.append({"role": "user", "content": user_input})

        response = self._llm_call()
        msg = response.choices[0].message

        # Append assistant message
        assistant_msg: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = msg.tool_calls
        self.messages.append(assistant_msg)

        # Tool-call loop
        iteration = 0
        while msg.tool_calls and iteration < self.max_iterations:
            iteration += 1
            logger.info("[%s] iteration %d/%d  tools=%d",
                        self.name, iteration, self.max_iterations, len(msg.tool_calls))

            for tc in msg.tool_calls:
                result = self._execute_tool_call(tc)
                self.messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tc.function.name,
                    "content":      json.dumps(result) if isinstance(result, dict) else str(result),
                })

            response = self._llm_call()
            msg = response.choices[0].message

            assistant_msg = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = msg.tool_calls
            self.messages.append(assistant_msg)

        if iteration >= self.max_iterations:
            logger.warning("[%s] max iterations reached (%d)", self.name, self.max_iterations)

        final = msg.content or ""
        logger.info("[%s] done  iterations=%d  reply_len=%d", self.name, iteration, len(final))
        return final
