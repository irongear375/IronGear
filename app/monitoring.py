"""
WristAssist AI — Prediction and Performance Monitoring Service
SCRUM-53: Implement prediction and performance monitoring service

Reads predictions.jsonl from HF Spaces persistent storage and computes
aggregate statistics for the GET /metrics endpoint.

Features:
- Total prediction count, latency percentiles (p50, p95), average latency
- Per-class prediction rates over 24h and 7d rolling windows
- Laterality distribution breakdown
- Human review rate
- 60-second TTL cache to avoid re-computation on every request
- Log rotation at 100 MB to prevent unbounded growth on free-tier storage

Usage:
    Called by app.py via `from monitoring import compute_metrics`
"""

import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("wristassist.monitoring")

# ── Configuration ──
LOG_DIR = Path(os.environ.get("LOG_DIR", "/data"))
PREDICTIONS_LOG = LOG_DIR / "predictions.jsonl"
MAX_LOG_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB rotation threshold
MODEL_VERSION = os.environ.get("MODEL_VERSION", "sprint2_yolo11l_expB_v1")

# Clinical classes (excludes 'text' infrastructure class)
CLINICAL_CLASSES = ["fracture", "metal_implant", "periosteal_reaction", "pronator_sign"]

# ── Cache ──
_metrics_cache: Optional[dict] = None
_cache_timestamp: float = 0.0
CACHE_TTL_SECONDS = 60


# ============================================================================
# Log rotation
# ============================================================================

def _rotate_log_if_needed():
    """
    Rotate predictions.jsonl when it exceeds 100 MB.
    Keeps the most recent half of entries, archives the rest.
    """
    if not PREDICTIONS_LOG.exists():
        return

    try:
        file_size = PREDICTIONS_LOG.stat().st_size
        if file_size < MAX_LOG_SIZE_BYTES:
            return

        logger.info(
            f"Log rotation triggered: {file_size / 1024 / 1024:.1f} MB "
            f"exceeds {MAX_LOG_SIZE_BYTES / 1024 / 1024:.0f} MB limit"
        )

        with open(PREDICTIONS_LOG, "r") as f:
            lines = f.readlines()

        midpoint = len(lines) // 2
        archive_path = LOG_DIR / f"predictions_archived_{int(time.time())}.jsonl"
        with open(archive_path, "w") as f:
            f.writelines(lines[:midpoint])

        with open(PREDICTIONS_LOG, "w") as f:
            f.writelines(lines[midpoint:])

        logger.info(
            f"Log rotated: archived {midpoint} entries to {archive_path.name}, "
            f"kept {len(lines) - midpoint} recent entries"
        )
    except Exception as e:
        logger.error(f"Log rotation failed: {e}")


# ============================================================================
# Log reader
# ============================================================================

def _read_predictions() -> list:
    """Read all prediction log entries from predictions.jsonl."""
    if not PREDICTIONS_LOG.exists():
        return []

    entries = []
    try:
        with open(PREDICTIONS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Failed to read predictions log: {e}")

    return entries


def _filter_by_window(entries: list, hours: int) -> list:
    """Filter prediction entries to those within the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()

    return [e for e in entries if e.get("timestamp", "") >= cutoff_iso]


# ============================================================================
# Statistics computation
# ============================================================================

def _compute_latency_stats(entries: list) -> dict:
    """Compute latency percentiles and average from prediction entries."""
    latencies = [e["latency_ms"] for e in entries if "latency_ms" in e]

    if not latencies:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return {
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p50_latency_ms": round(latencies_sorted[n // 2], 1),
        "p95_latency_ms": round(latencies_sorted[min(int(n * 0.95), n - 1)], 1),
        "min_latency_ms": round(latencies_sorted[0], 1),
        "max_latency_ms": round(latencies_sorted[-1], 1),
    }


def _compute_class_rates(entries: list) -> dict:
    """Compute per-class prediction rates: fraction of predictions containing each class."""
    if not entries:
        return {cls: 0.0 for cls in CLINICAL_CLASSES}

    total = len(entries)
    class_counts = {cls: 0 for cls in CLINICAL_CLASSES}

    for entry in entries:
        for cls in entry.get("classes_detected", []):
            if cls in class_counts:
                class_counts[cls] += 1

    return {cls: round(count / total, 4) for cls, count in class_counts.items()}


def _compute_laterality_distribution(entries: list) -> dict:
    """Compute laterality breakdown across predictions."""
    dist = {"Left": 0, "Right": 0, "Unknown": 0}
    for entry in entries:
        lat = entry.get("laterality", "Unknown")
        dist[lat] = dist.get(lat, 0) + 1
    return dist


def _compute_review_rate(entries: list) -> float:
    """Compute fraction of predictions that flagged needs_human_review."""
    if not entries:
        return 0.0
    review_count = sum(1 for e in entries if e.get("needs_human_review", False))
    return round(review_count / len(entries), 4)


def _compute_confidence_stats(entries: list) -> dict:
    """Compute average confidence per class across all predictions."""
    class_confs = {cls: [] for cls in CLINICAL_CLASSES}

    for entry in entries:
        for cls, conf in entry.get("confidences", {}).items():
            if cls in class_confs:
                class_confs[cls].append(conf)

    return {
        cls: round(sum(vals) / len(vals), 4) if vals else None
        for cls, vals in class_confs.items()
    }


# ============================================================================
# Public API — called by app.py
# ============================================================================

def compute_metrics() -> dict:
    """
    Compute aggregate monitoring statistics.

    Returns a JSON-serialisable dict consumed by GET /metrics.
    Results are cached for 60 seconds to avoid re-computation on
    repeated requests.
    """
    global _metrics_cache, _cache_timestamp

    now = time.time()
    if _metrics_cache is not None and (now - _cache_timestamp) < CACHE_TTL_SECONDS:
        return _metrics_cache

    _rotate_log_if_needed()

    all_entries = _read_predictions()
    entries_24h = _filter_by_window(all_entries, hours=24)
    entries_7d = _filter_by_window(all_entries, hours=168)

    metrics = {
        "model_version": MODEL_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,

        "total_predictions": len(all_entries),
        "predictions_24h": len(entries_24h),
        "predictions_7d": len(entries_7d),

        "latency_all_time": _compute_latency_stats(all_entries),
        "latency_24h": _compute_latency_stats(entries_24h),

        "class_rates_24h": _compute_class_rates(entries_24h),
        "class_rates_7d": _compute_class_rates(entries_7d),
        "class_rates_all_time": _compute_class_rates(all_entries),

        "avg_confidence_per_class": _compute_confidence_stats(all_entries),

        "laterality_distribution": _compute_laterality_distribution(all_entries),
        "laterality_distribution_24h": _compute_laterality_distribution(entries_24h),

        "human_review_rate": _compute_review_rate(all_entries),
        "human_review_rate_24h": _compute_review_rate(entries_24h),

        "log_file_exists": PREDICTIONS_LOG.exists(),
        "log_file_size_mb": round(
            PREDICTIONS_LOG.stat().st_size / 1024 / 1024, 2
        ) if PREDICTIONS_LOG.exists() else 0.0,
    }

    _metrics_cache = metrics
    _cache_timestamp = now

    return metrics
