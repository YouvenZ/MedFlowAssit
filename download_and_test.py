#!/usr/bin/env python3
"""
download_and_test.py — Download model weights & test all API endpoints.

Usage:
    python download_and_test.py --download-only      # Pre-download HF models
    python download_and_test.py --test-llm           # Test Groq LLM API
    python download_and_test.py --test-api           # Test all Flask endpoints
    python download_and_test.py --test-workflows     # Test all 13 workflows
    python download_and_test.py --test-all           # Everything

Requires the Flask app to be running on localhost:5000 for API tests.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── Coloured output helpers ───────────────────────────────────────────────────
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
except ImportError:
    # Stub out colorama if not installed
    class _Stub:
        GREEN = YELLOW = RED = CYAN = MAGENTA = RESET = ""
    Fore = Style = _Stub()

PASS = f"{Fore.GREEN}PASS{Style.RESET_ALL}"
FAIL = f"{Fore.RED}FAIL{Style.RESET_ALL}"
SKIP = f"{Fore.YELLOW}SKIP{Style.RESET_ALL}"
INFO = f"{Fore.CYAN}INFO{Style.RESET_ALL}"

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:5000")

_results: list[dict] = []


def _log(status: str, name: str, detail: str = ""):
    sym = {"PASS": PASS, "FAIL": FAIL, "SKIP": SKIP, "INFO": INFO}.get(status, status)
    line = f"  [{sym}] {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    _results.append({"status": status, "name": name, "detail": detail})


# ══════════════════════════════════════════════════════════════════════════════
#  1.  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

MODELS_TO_DOWNLOAD = [
    {
        "name": "MedGemma 1.5 4B-IT (vision-language)",
        "repo_id": "google/medgemma-1.5-4b-it",
        "env_override": None,
    },
    {
        "name": "MedASR (medical speech-to-text)",
        "repo_id": os.getenv("MEDASR_MODEL", "google/medasr"),
        "env_override": "MEDASR_MODEL",
    },
]


def download_models():
    """Pre-download all HuggingFace model weights to the local cache."""
    print(f"\n{'='*70}")
    print("  MODEL DOWNLOAD — Pre-downloading HuggingFace model weights")
    print(f"{'='*70}\n")

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        _log("FAIL", "huggingface_hub import",
             "Install with: pip install huggingface-hub")
        return

    api = HfApi()

    for model in MODELS_TO_DOWNLOAD:
        name = model["name"]
        repo = model["repo_id"]
        print(f"  Downloading: {name}")
        print(f"  Repo: {repo}")

        try:
            # Check if model exists on HF Hub
            try:
                info = api.model_info(repo)
                size_mb = (info.siblings and
                           sum(s.size or 0 for s in info.siblings) / 1e6)
                print(f"  Size: ~{size_mb:.0f} MB" if size_mb else "  Size: unknown")
            except Exception:
                print(f"  (Could not fetch model info — may require auth)")

            t0 = time.time()
            path = snapshot_download(
                repo_id=repo,
                resume_download=True,
            )
            elapsed = time.time() - t0
            _log("PASS", f"Download {name}",
                 f"Cached at {path} ({elapsed:.1f}s)")

        except Exception as exc:
            _log("FAIL", f"Download {name}", str(exc))

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  2.  LLM API TESTS (Groq via LiteLLM)
# ══════════════════════════════════════════════════════════════════════════════

def test_llm_api():
    """Test the Groq LLM API endpoint via litellm."""
    print(f"\n{'='*70}")
    print("  LLM API TESTS — Groq via LiteLLM")
    print(f"{'='*70}\n")

    # Check API key
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_key")
    if not groq_key:
        _log("FAIL", "GROQ_API_KEY", "Not set in environment or .env file")
        return
    _log("PASS", "GROQ_API_KEY", f"Found ({groq_key[:8]}...)")

    # Test litellm import
    try:
        from litellm import completion
        _log("PASS", "litellm import", "OK")
    except ImportError:
        _log("FAIL", "litellm import", "pip install litellm")
        return

    # Test basic completion
    try:
        from llm_config import llm_completion, DEFAULT_MODEL
        _log("INFO", "LLM model", DEFAULT_MODEL)

        t0 = time.time()
        resp = llm_completion(
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": "What is the normal range for hemoglobin in adult males? Answer in one sentence."},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content.strip()
        _log("PASS", "LLM basic completion",
             f"{elapsed:.1f}s — {len(text)} chars: \"{text[:80]}...\"")
    except Exception as exc:
        _log("FAIL", "LLM basic completion", str(exc))

    # Test JSON extraction (used by workflows)
    try:
        t0 = time.time()
        resp = llm_completion(
            messages=[
                {"role": "system", "content": "You are a data extraction engine."},
                {"role": "user", "content": (
                    "Extract from this text: '55-year-old female with hypertension and diabetes.'\n"
                    "Return ONLY valid JSON: {\"age\": <int>, \"sex\": \"<m/f>\", "
                    "\"hypertension\": <bool>, \"diabetes\": <bool>}"
                )},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        elapsed = time.time() - t0
        raw = resp.choices[0].message.content.strip()
        # Try parsing JSON
        clean = raw
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean)
        _log("PASS", "LLM JSON extraction",
             f"{elapsed:.1f}s — parsed: {json.dumps(parsed)}")
    except json.JSONDecodeError:
        _log("FAIL", "LLM JSON extraction",
             f"Response not valid JSON: {raw[:100]}")
    except Exception as exc:
        _log("FAIL", "LLM JSON extraction", str(exc))

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  3.  FLASK APP API TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _req(method: str, path: str, **kwargs):
    """Helper to make HTTP requests."""
    import requests
    url = f"{BASE_URL}{path}"
    resp = getattr(requests, method)(url, timeout=30, **kwargs)
    return resp


def test_api_endpoints():
    """Test all Flask app API endpoints (app must be running)."""
    print(f"\n{'='*70}")
    print(f"  APP API TESTS — {BASE_URL}")
    print(f"{'='*70}\n")

    try:
        import requests as _req_lib
    except ImportError:
        _log("FAIL", "requests import", "pip install requests")
        return

    # ── Check app is running ──────────────────────────────────────────────
    try:
        r = _req("get", "/")
        _log("PASS", "GET /", f"HTTP {r.status_code}, {len(r.text)} bytes")
    except Exception as exc:
        _log("FAIL", "GET / (app running?)", str(exc))
        print(f"\n  {Fore.RED}App is not running! Start with: python app.py{Style.RESET_ALL}\n")
        return

    # ── Page routes ───────────────────────────────────────────────────────
    for route in ["/chat", "/workflows"]:
        try:
            r = _req("get", route)
            status = "PASS" if r.status_code == 200 else "FAIL"
            _log(status, f"GET {route}", f"HTTP {r.status_code}")
        except Exception as exc:
            _log("FAIL", f"GET {route}", str(exc))

    # ── GET /api/status ──────────────────────────────────────────────────
    try:
        r = _req("get", "/api/status")
        _log("PASS" if r.status_code == 200 else "FAIL",
             "GET /api/status", f"HTTP {r.status_code}")
    except Exception as exc:
        _log("FAIL", "GET /api/status", str(exc))

    # ── POST /api/reset ──────────────────────────────────────────────────
    try:
        r = _req("post", "/api/reset")
        _log("PASS" if r.status_code == 200 else "FAIL",
             "POST /api/reset", f"HTTP {r.status_code}")
    except Exception as exc:
        _log("FAIL", "POST /api/reset", str(exc))

    # ── POST /api/chat ───────────────────────────────────────────────────
    try:
        t0 = time.time()
        r = _req("post", "/api/chat",
                 json={"message": "I have a headache and mild fever for 2 days."})
        elapsed = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            reply = data.get("reply", "")[:80]
            _log("PASS", "POST /api/chat",
                 f"HTTP 200, {elapsed:.1f}s — \"{reply}...\"")
        else:
            _log("FAIL", "POST /api/chat", f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as exc:
        _log("FAIL", "POST /api/chat", str(exc))

    # ── GET /api/workflows ───────────────────────────────────────────────
    try:
        r = _req("get", "/api/workflows")
        if r.status_code == 200:
            cards = r.json()
            ids = [c["workflow_id"] for c in cards]
            _log("PASS", "GET /api/workflows",
                 f"{len(cards)} workflows: {', '.join(ids)}")
        else:
            _log("FAIL", "GET /api/workflows", f"HTTP {r.status_code}")
    except Exception as exc:
        _log("FAIL", "GET /api/workflows", str(exc))

    # ── GET /api/medasr/status ───────────────────────────────────────────
    try:
        r = _req("get", "/api/medasr/status")
        if r.status_code == 200:
            avail = r.json().get("available", False)
            _log("PASS", "GET /api/medasr/status",
                 f"available={avail}")
        else:
            _log("FAIL", "GET /api/medasr/status", f"HTTP {r.status_code}")
    except Exception as exc:
        _log("FAIL", "GET /api/medasr/status", str(exc))

    # ── POST /api/upload/csv (with sample data) ─────────────────────────
    try:
        csv_content = "test,value,unit\nHemoglobin,14.2,g/dL\nWBC,7.5,×10³/µL\n"
        files = {"file": ("test_labs.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        r = _req("post", "/api/upload/csv", files=files)
        if r.status_code == 200:
            data = r.json()
            _log("PASS", "POST /api/upload/csv",
                 f"HTTP 200, {data.get('rows', '?')} rows, {data.get('chars', '?')} chars")
        else:
            _log("FAIL", "POST /api/upload/csv", f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as exc:
        _log("FAIL", "POST /api/upload/csv", str(exc))

    # ── POST /api/upload/image (with tiny PNG) ──────────────────────────
    try:
        # 1×1 red PNG (67 bytes)
        import base64
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        files = {"file": ("test.png", io.BytesIO(tiny_png), "image/png")}
        r = _req("post", "/api/upload/image", files=files)
        if r.status_code == 200:
            data = r.json()
            has_uri = "image_url" in data and data["image_url"].startswith("data:")
            _log("PASS" if has_uri else "FAIL", "POST /api/upload/image",
                 f"HTTP 200, data-URI={'yes' if has_uri else 'no'}, {data.get('size', '?')} bytes")
        else:
            _log("FAIL", "POST /api/upload/image", f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as exc:
        _log("FAIL", "POST /api/upload/image", str(exc))

    # ── POST /api/upload/pdf (error case — no file) ─────────────────────
    try:
        r = _req("post", "/api/upload/pdf")
        _log("PASS" if r.status_code == 400 else "FAIL",
             "POST /api/upload/pdf (no file)",
             f"HTTP {r.status_code} (expected 400)")
    except Exception as exc:
        _log("FAIL", "POST /api/upload/pdf (no file)", str(exc))

    # ── POST /api/transcribe (error case — no audio) ────────────────────
    try:
        r = _req("post", "/api/transcribe", json={"audio_b64": ""})
        _log("PASS" if r.status_code == 400 else "FAIL",
             "POST /api/transcribe (no audio)",
             f"HTTP {r.status_code} (expected 400)")
    except Exception as exc:
        _log("FAIL", "POST /api/transcribe (no audio)", str(exc))

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  4.  WORKFLOW EXECUTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

WORKFLOW_TEST_PAYLOADS = {
    # ── Text/PDF workflows (LLM-backed) ──────────────────────────────────
    "stroke_risk": {
        "text": ("72-year-old male with atrial fibrillation, hypertension, "
                 "history of TIA 2 years ago, type 2 diabetes, no heart failure.")
    },
    "qsofa_calc": {
        "text": ("45-year-old female presents with suspected UTI. "
                 "BP 88/55, RR 24, confused and disoriented. Temp 39.2°C, HR 112.")
    },
    "wells_pe": {
        "text": ("62-year-old male with sudden onset dyspnea. HR 108 bpm. "
                 "Right calf tender and swollen. Recent hip replacement 2 weeks ago. "
                 "No hemoptysis. No cancer history. PE is the most likely diagnosis.")
    },
    "meld_calc": {
        "text": ("Cirrhosis patient labs: Bilirubin 3.8 mg/dL, INR 1.9, "
                 "Creatinine 1.5 mg/dL, Sodium 131 mEq/L. No dialysis. "
                 "Alcoholic liver disease.")
    },
    "gcs_calc": {
        "text": ("Head trauma patient: eyes open to pain (E2), "
                 "incomprehensible sounds (V2), flexion withdrawal to pain (M4). "
                 "Pupils: right reactive, left fixed and dilated. "
                 "Fell from 3-meter height.")
    },
    "triage_flag": {
        "text": ("CT Head without contrast: Large right-sided subdural hematoma "
                 "with 12mm midline shift. Uncal herniation noted. "
                 "Recommend urgent neurosurgical consultation.")
    },
    "growth_plotter": {
        "text": ("Well child visit: 8-year-old boy. Weight 28 kg, Height 130 cm. "
                 "No chronic conditions.")
    },
    "consult_scribe": {
        "text": ("Patient: 45F presents with 3-day history of productive cough, "
                 "fever 38.5C, and right-sided chest pain worse with inspiration. "
                 "PMH: asthma. O/E: decreased breath sounds right base, "
                 "dullness to percussion. SpO2 94% on room air. "
                 "Assessment: likely community acquired pneumonia. "
                 "Plan: CXR, CBC, CRP, start amoxicillin/clavulanate, follow up 48h.")
    },
    # ── CSV workflows (deterministic, no LLM) ────────────────────────────
    "lab_flagr": {
        "csv_text": (
            "test,value,unit\n"
            "Hemoglobin,8.2,g/dL\n"
            "WBC,15.8,×10³/µL\n"
            "Platelets,95,×10³/µL\n"
            "Sodium,128,mEq/L\n"
            "Potassium,6.2,mEq/L\n"
            "Creatinine,3.5,mg/dL\n"
            "Glucose,45,mg/dL\n"
            "BUN,42,mg/dL\n"
            "Calcium,8.1,mg/dL\n"
            "Albumin,2.8,g/dL\n"
            "ALT,89,U/L\n"
            "AST,112,U/L\n"
            "Bilirubin Total,2.1,mg/dL\n"
            "INR,1.8,\n"
            "Troponin,0.15,ng/mL\n"
            "Lactate,4.5,mmol/L\n"
        )
    },
    "rx_interact": {
        "csv_text": (
            "drug,dose,frequency\n"
            "Warfarin,5 mg,daily\n"
            "Aspirin,100 mg,daily\n"
            "Omeprazole,20 mg,daily\n"
            "Simvastatin,40 mg,at night\n"
            "Amlodipine,10 mg,daily\n"
            "Metformin,1000 mg,BID\n"
            "Furosemide,40 mg,morning\n"
            "Potassium,20 mEq,BID\n"
            "Lisinopril,10 mg,daily\n"
        )
    },
    "vitals_trend": {
        "csv_text": (
            "timestamp,hr,sbp,rr,temp,spo2,consciousness,on_o2\n"
            "2026-02-24 06:00,78,125,16,36.8,97,A,false\n"
            "2026-02-24 08:00,82,120,18,37.1,96,A,false\n"
            "2026-02-24 10:00,95,110,20,37.8,94,A,false\n"
            "2026-02-24 12:00,105,100,22,38.5,92,A,true\n"
            "2026-02-24 14:00,115,92,26,39.1,89,C,true\n"
            "2026-02-24 16:00,120,85,28,39.5,87,V,true\n"
        )
    },
}


def test_workflows():
    """Execute all 13 workflows with sample data via the API."""
    print(f"\n{'='*70}")
    print(f"  WORKFLOW EXECUTION TESTS — {BASE_URL}/api/workflows/run")
    print(f"{'='*70}\n")

    try:
        import requests as _req_lib
    except ImportError:
        _log("FAIL", "requests import", "pip install requests")
        return

    # Check app is running
    try:
        _req("get", "/")
    except Exception:
        _log("FAIL", "App connectivity",
             f"App not running at {BASE_URL}. Start with: python app.py")
        return

    for wf_id, payload in WORKFLOW_TEST_PAYLOADS.items():
        try:
            body = {"workflow_id": wf_id, **payload}
            t0 = time.time()
            r = _req("post", "/api/workflows/run", json=body)
            elapsed = time.time() - t0

            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "?")
                summary = (data.get("data", {}).get("summary", ""))[:100]

                if status == "success":
                    _log("PASS", f"Workflow: {wf_id}",
                         f"{elapsed:.1f}s — {summary}")
                else:
                    err = data.get("data", {}).get("summary", "Unknown error")
                    _log("FAIL", f"Workflow: {wf_id}",
                         f"{elapsed:.1f}s — status={status}: {err[:100]}")
            else:
                _log("FAIL", f"Workflow: {wf_id}",
                     f"HTTP {r.status_code}: {r.text[:100]}")
        except Exception as exc:
            _log("FAIL", f"Workflow: {wf_id}", str(exc))

    # Test workflows not in WORKFLOW_TEST_PAYLOADS (require image — skip)
    image_only = {"retina_counter", "medrec_guard"}
    for wf_id in image_only:
        _log("SKIP", f"Workflow: {wf_id}", "Requires image input — skipped in automated test")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  5.  SCENARIO TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_scenarios():
    """Test a couple of pre-built scenarios via the API."""
    print(f"\n{'='*70}")
    print(f"  SCENARIO TESTS — {BASE_URL}/api/scenario/<num>")
    print(f"{'='*70}\n")

    try:
        import requests as _req_lib
    except ImportError:
        _log("FAIL", "requests import", "pip install requests")
        return

    # Reset session first
    try:
        _req("post", "/api/reset")
    except Exception:
        pass

    # Test scenario 1 (headache triage) and scenario 8 (generic)
    for num in [1, 8]:
        try:
            t0 = time.time()
            r = _req("post", f"/api/scenario/{num}")
            elapsed = time.time() - t0

            if r.status_code == 200:
                data = r.json()
                conv = data.get("conversation", [])
                _log("PASS", f"Scenario {num}",
                     f"{elapsed:.1f}s — {len(conv)} exchange(s)")
            else:
                _log("FAIL", f"Scenario {num}",
                     f"HTTP {r.status_code}: {r.text[:100]}")
        except Exception as exc:
            _log("FAIL", f"Scenario {num}", str(exc))

    # Test invalid scenario
    try:
        r = _req("post", "/api/scenario/999")
        _log("PASS" if r.status_code == 404 else "FAIL",
             "Scenario 999 (invalid)",
             f"HTTP {r.status_code} (expected 404)")
    except Exception as exc:
        _log("FAIL", "Scenario 999 (invalid)", str(exc))

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")
    total = passed + failed + skipped

    print(f"{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped  ({total} total)")
    print(f"{'='*70}")

    if failed:
        print(f"\n  {Fore.RED}Failed tests:{Style.RESET_ALL}")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    ✗ {r['name']}  — {r['detail']}")
        print()

    return 0 if failed == 0 else 1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download model weights & test all API endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python download_and_test.py --download-only\n"
            "  python download_and_test.py --test-all\n"
            "  python download_and_test.py --test-llm --test-workflows\n"
        ),
    )
    parser.add_argument("--download-only", action="store_true",
                        help="Only download model weights (no tests)")
    parser.add_argument("--test-llm", action="store_true",
                        help="Test Groq LLM API via LiteLLM")
    parser.add_argument("--test-api", action="store_true",
                        help="Test all Flask API endpoints")
    parser.add_argument("--test-workflows", action="store_true",
                        help="Execute all 13 workflows with sample data")
    parser.add_argument("--test-scenarios", action="store_true",
                        help="Test pre-built clinical scenarios")
    parser.add_argument("--test-all", action="store_true",
                        help="Run everything: download + LLM + API + workflows + scenarios")
    parser.add_argument("--base-url", default=BASE_URL,
                        help=f"App base URL (default: {BASE_URL})")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    # If no flags, show help
    if not any([args.download_only, args.test_llm, args.test_api,
                args.test_workflows, args.test_scenarios, args.test_all]):
        parser.print_help()
        return 0

    print(f"\n{'#'*70}")
    print(f"  MedGemma Clinical Workflow Engine — Download & Test")
    print(f"  Base URL: {BASE_URL}")
    print(f"{'#'*70}")

    if args.download_only or args.test_all:
        download_models()

    if args.test_llm or args.test_all:
        test_llm_api()

    if args.test_api or args.test_all:
        test_api_endpoints()

    if args.test_workflows or args.test_all:
        test_workflows()

    if args.test_scenarios or args.test_all:
        test_scenarios()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
