"""
workflows/__init__.py — Workflow registry.

Import all workflow classes and expose a WORKFLOW_REGISTRY dict
keyed by workflow_id for the UI and API routes.
"""

from workflows.retina_counter import RetinaCounterWorkflow
from workflows.stroke_risk import StrokeRiskWorkflow
from workflows.medrec_guard import MedRecGuardWorkflow
from workflows.triage_flag import TriageFlagWorkflow
from workflows.growth_plotter import GrowthPlotterWorkflow
from workflows.consult_scribe import ConsultScribeWorkflow
from workflows.qsofa_calc import QSofaCalcWorkflow
from workflows.wells_pe import WellsPEWorkflow
from workflows.meld_calc import MeldCalcWorkflow
from workflows.gcs_calc import GCSCalcWorkflow
from workflows.lab_flagr import LabFlagrWorkflow
from workflows.rx_interact import RxInteractWorkflow
from workflows.vitals_trend import VitalsTrendWorkflow

# ── Registry: workflow_id → class ─────────────────────────────────────────────
WORKFLOW_CLASSES: dict[str, type] = {
    "retina_counter": RetinaCounterWorkflow,
    "stroke_risk":    StrokeRiskWorkflow,
    "medrec_guard":   MedRecGuardWorkflow,
    "triage_flag":    TriageFlagWorkflow,
    "growth_plotter": GrowthPlotterWorkflow,
    "consult_scribe": ConsultScribeWorkflow,
    "qsofa_calc":     QSofaCalcWorkflow,
    "wells_pe":       WellsPEWorkflow,
    "meld_calc":      MeldCalcWorkflow,
    "gcs_calc":       GCSCalcWorkflow,
    "lab_flagr":      LabFlagrWorkflow,
    "rx_interact":    RxInteractWorkflow,
    "vitals_trend":   VitalsTrendWorkflow,
}


def get_all_cards() -> list[dict]:
    """Return card_info() for every registered workflow (for the grid UI)."""
    return [cls().card_info() for cls in WORKFLOW_CLASSES.values()]


def run_workflow(workflow_id: str, data: dict) -> dict:
    """Instantiate and run a workflow by its id. Returns result dict."""
    cls = WORKFLOW_CLASSES.get(workflow_id)
    if cls is None:
        return {
            "workflow_id": workflow_id,
            "status": "error",
            "data": {"summary": f"Unknown workflow: {workflow_id}",
                     "metrics": {}, "protocol_adherence": False, "raw_output": ""},
            "artifacts": [],
        }
    wf = cls()
    result = wf.run(data)
    return result.to_dict()
