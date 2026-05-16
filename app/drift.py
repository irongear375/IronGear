"""
WristAssist AI — Drift Detection Service
SCRUM-54: Implement drift detection with PSI computation

Compares the live prediction class distribution (from predictions.jsonl)
against the Sprint 2 training baseline (baseline_distribution.json) using
the Population Stability Index (PSI).

Thresholds (per HLD Section 3.2.8):
- PSI < 0.10  → OK      (no action)
- PSI 0.10–0.25 → WARN  (log entry only)
- PSI > 0.25  → ALERT   (auto-create GitHub issue + trigger retrain)

Features:
- Per-class PSI computation with configurable bin count
- Overall (weighted) PSI across all clinical classes
- Historical PSI trend stored in drift_history.jsonl
- GitHub issue auto-creation via repository_dispatch on ALERT
- 5-minute TTL cache to avoid re-computation

Usage:
    Called by app.py via `from drift import compute_drift`
"""

import json
import os
import time
import math
import logging
import urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("wristassist.drift")

# ── Configuration ──
LOG_DIR = Path(os.environ.get("LOG_DIR", "/data"))
PREDICTIONS_LOG = LOG_DIR / "predictions.jsonl"
DRIFT_HISTORY_LOG = LOG_DIR / "drift_history.jsonl"

# Baseline file — bundled in the Docker image at build time
APP_DIR = Path(__file__).parent
BASELINE_PATH = APP_DIR / "baseline_distribution.json"

MODEL_VERSION = os.environ.get("MODEL_VERSION", "sprint2_yolo11l_expB_v1")

# GitHub integration for alert dispatch (optional — set via HF Space secrets)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "irongear375/IronGear")

# Clinical classes (matches app.py and monitoring.py)
CLINICAL_CLASSES = ["fracture", "metal_implant", "periosteal_reaction", "pronator_sign"]

# PSI thresholds per HLD
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# ── Cache ──
_drift_cache: Optional[dict] = None
_cache_timestamp: float = 0.0
CACHE_TTL_SECONDS = 300  # 5 minutes


# ============================================================================
# Baseline loader
# ============================================================================

def _load_baseline() -> Optional[dict]:
    """Load the Sprint 2 training class distribution baseline."""
    if not BASELINE_PATH.exists():
        logger.warning(f"Baseline file not found at {BASELINE_PATH}")
        return None

    try:
        with open(BASELINE_PATH, "r") as f:
            baseline = json.load(f)
        logger.info(f"Loaded baseline distribution from {BASELINE_PATH}")
        return baseline
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return None


# ============================================================================
# Prediction log reader
# ============================================================================

def _read_predictions(hours: Optional[int] = None) -> list:
    """Read prediction log entries, optionally filtered by time window."""
    if not PREDICTIONS_LOG.exists():
        return []

    entries = []
    cutoff_iso = None
    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()

    try:
        with open(PREDICTIONS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if cutoff_iso is None or entry.get("timestamp", "") >= cutoff_iso:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to read predictions log: {e}")

    return entries


# ============================================================================
# Live distribution computation
# ============================================================================

def _compute_live_distribution(entries: list) -> dict:
    """
    Compute the live class prediction rate distribution from log entries.
    Returns {class_name: rate} where rate = fraction of predictions
    that contained that class.
    """
    if not entries:
        return {cls: 0.0 for cls in CLINICAL_CLASSES}

    total = len(entries)
    class_counts = {cls: 0 for cls in CLINICAL_CLASSES}

    for entry in entries:
        for cls in entry.get("classes_detected", []):
            if cls in class_counts:
                class_counts[cls] += 1

    return {cls: count / total for cls, count in class_counts.items()}


# ============================================================================
# PSI computation
# ============================================================================

def _compute_psi(baseline_rate: float, live_rate: float) -> float:
    """
    Compute Population Stability Index for a single class.

    PSI = (live - baseline) * ln(live / baseline)

    Small epsilon added to avoid log(0) and division by zero.
    """
    eps = 1e-6
    b = max(baseline_rate, eps)
    l = max(live_rate, eps)
    return (l - b) * math.log(l / b)


def _classify_psi(psi_value: float) -> str:
    """Classify PSI value into OK / WARN / ALERT."""
    if psi_value > PSI_ALERT_THRESHOLD:
        return "ALERT"
    elif psi_value > PSI_WARN_THRESHOLD:
        return "WARN"
    return "OK"


# ============================================================================
# GitHub alert dispatch
# ============================================================================

def _dispatch_github_alert(drift_result: dict):
    """
    Create a GitHub issue via repository_dispatch when PSI exceeds
    the ALERT threshold. Also fires a repository_dispatch event
    that can trigger the retrain.yml workflow.
    """
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — skipping alert dispatch")
        return

    # Fire repository_dispatch for retrain workflow
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
        payload = json.dumps({
            "event_type": "drift-alert",
            "client_payload": {
                "overall_psi": drift_result.get("overall_psi", 0),
                "per_class_psi": drift_result.get("per_class", {}),
                "model_version": MODEL_VERSION,
                "computed_at": drift_result.get("computed_at", ""),
                "alert_classes": [
                    cls for cls, info in drift_result.get("per_class", {}).items()
                    if info.get("status") == "ALERT"
                ],
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 204:
                logger.info("Drift alert dispatched to GitHub (repository_dispatch)")
            else:
                logger.warning(f"GitHub dispatch returned status {resp.status}")

    except Exception as e:
        logger.error(f"GitHub alert dispatch failed: {e}")


# ============================================================================
# Drift history logging
# ============================================================================

def _log_drift_result(result: dict):
    """Append drift computation result to drift_history.jsonl for trend tracking."""
    try:
        entry = {
            "computed_at": result["computed_at"],
            "overall_psi": result["overall_psi"],
            "overall_status": result["overall_status"],
            "per_class_psi": {
                cls: info["psi"] for cls, info in result["per_class"].items()
            },
            "prediction_count": result["prediction_count"],
            "window_hours": result["window_hours"],
        }
        with open(DRIFT_HISTORY_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log drift result: {e}")


def _read_drift_history(max_entries: int = 30) -> list:
    """Read the last N drift history entries for trend display."""
    if not DRIFT_HISTORY_LOG.exists():
        return []

    entries = []
    try:
        with open(DRIFT_HISTORY_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-max_entries:]
    except Exception as e:
        logger.error(f"Failed to read drift history: {e}")
        return []


# ============================================================================
# Public API — called by app.py
# ============================================================================

def compute_drift(window_hours: int = 168) -> dict:
    """
    Compute PSI drift for each clinical class comparing live predictions
    (over the specified window, default 7 days) against the Sprint 2
    training baseline.

    Returns a JSON-serialisable dict consumed by GET /drift.
    Results are cached for 5 minutes.
    """
    global _drift_cache, _cache_timestamp

    now = time.time()
    if _drift_cache is not None and (now - _cache_timestamp) < CACHE_TTL_SECONDS:
        return _drift_cache

    computed_at = datetime.now(timezone.utc).isoformat()

    # Load baseline
    baseline = _load_baseline()
    if baseline is None:
        result = {
            "status": "baseline_not_found",
            "detail": f"Baseline file not found at {BASELINE_PATH}. "
                      "Generate it from Sprint 2 evaluation results.",
            "computed_at": computed_at,
        }
        return result

    baseline_rates = baseline.get("class_rates", {})

    # Compute live distribution
    entries = _read_predictions(hours=window_hours)
    live_rates = _compute_live_distribution(entries)

    # Compute PSI per class
    per_class = {}
    psi_values = []

    for cls in CLINICAL_CLASSES:
        b_rate = baseline_rates.get(cls, 0.0)
        l_rate = live_rates.get(cls, 0.0)
        psi = _compute_psi(b_rate, l_rate)
        status = _classify_psi(psi)

        per_class[cls] = {
            "baseline_rate": round(b_rate, 4),
            "live_rate": round(l_rate, 4),
            "psi": round(psi, 6),
            "status": status,
        }
        psi_values.append(psi)

    # Overall PSI (average across classes)
    overall_psi = sum(psi_values) / len(psi_values) if psi_values else 0.0
    overall_status = _classify_psi(overall_psi)

    result = {
        "model_version": MODEL_VERSION,
        "computed_at": computed_at,
        "window_hours": window_hours,
        "prediction_count": len(entries),
        "overall_psi": round(overall_psi, 6),
        "overall_status": overall_status,
        "thresholds": {
            "warn": PSI_WARN_THRESHOLD,
            "alert": PSI_ALERT_THRESHOLD,
        },
        "per_class": per_class,
        "baseline_source": str(BASELINE_PATH),
        "history": _read_drift_history(),
    }

    # Log this computation to drift history
    _log_drift_result(result)

    # Dispatch GitHub alert if any class is in ALERT
    if overall_status == "ALERT" or any(
        info["status"] == "ALERT" for info in per_class.values()
    ):
        logger.warning(f"DRIFT ALERT: overall PSI={overall_psi:.4f}, "
                       f"classes={[c for c, i in per_class.items() if i['status'] == 'ALERT']}")
        _dispatch_github_alert(result)

    # Cache
    _drift_cache = result
    _cache_timestamp = now

    return result
