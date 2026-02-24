"""
workflows/base.py — Abstract base class for all clinical workflows.

Every workflow inherits from ClinicalWorkflow and implements:
  • metadata (name, icon, description, input_types, protocol)
  • validate_input(data)  → bool
  • execute(data)         → WorkflowResult
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Standard result schema
# ══════════════════════════════════════════════════════════════════════════════

class WorkflowStatus(str, Enum):
    SUCCESS = "success"
    ERROR   = "error"


class InputType(str, Enum):
    TEXT  = "Text"
    IMAGE = "Image"
    PDF   = "PDF"
    CSV   = "CSV"


@dataclass
class WorkflowResult:
    """Standard return object for every workflow — matches the spec JSON."""
    workflow_id: str
    status: WorkflowStatus
    data: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "data": self.data,
            "artifacts": self.artifacts,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Base class
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalWorkflow(ABC):
    """Abstract base for chain-based clinical workflows."""

    # ── metadata (override in subclass) ───────────────────────────────────────
    workflow_id: str = ""
    name: str = ""
    icon: str = ""
    description: str = ""
    input_types: list[InputType] = []
    protocol: str = ""
    specialty: str = ""

    def __init__(self):
        self._run_id = str(uuid.uuid4())[:8]
        logger.info("[%s] workflow instantiated  run=%s", self.workflow_id, self._run_id)

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, data: dict[str, Any]) -> WorkflowResult:
        """
        Validate → execute → wrap in standard result.
        Subclasses should NOT override this; override execute() instead.
        """
        logger.info("[%s] run started  run=%s", self.workflow_id, self._run_id)
        try:
            errors = self.validate_input(data)
            if errors:
                return WorkflowResult(
                    workflow_id=self.workflow_id,
                    status=WorkflowStatus.ERROR,
                    data={"summary": "Validation failed", "errors": errors,
                          "metrics": {}, "protocol_adherence": False, "raw_output": ""},
                )

            result_data = self.execute(data)

            # Ensure standard keys
            result_data.setdefault("summary", "")
            result_data.setdefault("metrics", {})
            result_data.setdefault("protocol_adherence", True)
            result_data.setdefault("raw_output", "")

            logger.info("[%s] run completed  run=%s", self.workflow_id, self._run_id)
            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.SUCCESS,
                data=result_data,
            )

        except Exception as exc:
            logger.error("[%s] run failed: %s", self.workflow_id, exc, exc_info=True)
            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.ERROR,
                data={"summary": f"Workflow error: {exc}", "metrics": {},
                      "protocol_adherence": False, "raw_output": str(exc)},
            )

    # ── hooks for subclasses ──────────────────────────────────────────────────

    @abstractmethod
    def validate_input(self, data: dict[str, Any]) -> list[str]:
        """Return list of validation error strings, or empty list if OK."""
        ...

    @abstractmethod
    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Run the chain and return a dict with at least:
          summary, metrics, protocol_adherence, raw_output
        """
        ...

    # ── Card metadata for UI ─────────────────────────────────────────────────

    def card_info(self) -> dict:
        """Return the metadata the grid UI needs to render a card."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "input_types": [t.value for t in self.input_types],
            "protocol": self.protocol,
            "specialty": self.specialty,
        }
