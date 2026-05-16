# ======================================================================
# Sprint 2 — Clinical Report Generation Module
# ======================================================================
# Project:  WristAssist AI | Team: Iron Gear
# Ticket:   SCRUM-33
# Location: Sprint2/utils/report.py
#
# Purpose:
#   Assembles a structured JSON clinical report from YOLO detection results,
#   EasyOCR laterality output, and inference metadata. Every inference call
#   produces one report. The report is the core clinical decision-support
#   output of WristAssist AI.
#
# Report Structure:
#   - request_id:               UUID4 unique identifier
#   - model_version:            e.g. "sprint2_yolo11x_v1"
#   - timestamp:                ISO 8601 UTC
#   - laterality:               "Left" | "Right" | "Unknown"
#   - raw_text_detected:        list of strings from EasyOCR
#   - findings_detected:        list of {class_name, confidence, box}
#   - no_findings:              clinical classes NOT detected (never includes "text")
#   - summary:                  boolean flags for quick triage
#   - confidence_threshold_used: float
#   - latency_ms:               end-to-end inference time
#
# Usage:
#   from Sprint2.utils.report import generate_report
#   report = generate_report(detections, laterality, raw_text,
#                            conf_threshold, latency_ms)
# ======================================================================

import uuid
from datetime import datetime, timezone
from typing import Any


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# The 4 clinical detection classes. "text" is an infrastructure class
# for laterality extraction — it is NEVER included in no_findings.
CLINICAL_CLASSES = {
    0: "fracture",
    1: "metal_implant",
    2: "periosteal_reaction",
    3: "pronator_sign",
}

# Class name → summary flag key mapping
_CLASS_TO_FLAG = {
    "fracture":             "fracture_suspected",
    "metal_implant":        "implant_present",
    "periosteal_reaction":  "periosteal_reaction_present",
    "pronator_sign":        "pronator_sign_present",
}

# Default model version string (updated each sprint)
DEFAULT_MODEL_VERSION = "sprint2_yolo11l_expB_v1"

# needs_human_review is triggered when:
# 1. Any detection has confidence between threshold and threshold + this margin
# 2. Fracture is detected but confidence is below this value
# 3. No clinical findings are detected at all
_REVIEW_MARGIN = 0.15
_FRACTURE_REVIEW_THRESHOLD = 0.70


def generate_report(
    detections: list[dict],
    laterality: str = "Unknown",
    raw_text_detected: list[str] | None = None,
    confidence_threshold: float = 0.42,
    latency_ms: float = 0.0,
    model_version: str = DEFAULT_MODEL_VERSION,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Generate a structured clinical report from inference results.

    Args:
        detections: List of detection dicts, each containing:
            - class_id (int): YOLO class ID (0-4)
            - class_name (str): human-readable class name
            - confidence (float): detection confidence score
            - box (list[float]): normalised [x1, y1, x2, y2] coordinates
        laterality: Wrist laterality from EasyOCR ("Left", "Right", "Unknown")
        raw_text_detected: Raw OCR output strings (for traceability)
        confidence_threshold: Confidence threshold used for filtering detections
        latency_ms: End-to-end inference latency in milliseconds
        model_version: Model version string for traceability
        request_id: Optional UUID string. Auto-generated if not provided.

    Returns:
        dict: Complete clinical report with all required fields.
    """
    # ── Generate metadata ──
    if request_id is None:
        request_id = str(uuid.uuid4())

    timestamp = datetime.now(timezone.utc).isoformat()

    if raw_text_detected is None:
        raw_text_detected = []

    # ── Filter detections: only clinical classes above threshold ──
    # Text class (ID 4) detections are used for OCR but NOT included
    # in the clinical findings list.
    findings_detected = []
    detected_class_names = set()

    for det in detections:
        class_id = det.get("class_id", -1)
        class_name = det.get("class_name", "")
        confidence = det.get("confidence", 0.0)
        box = det.get("box", [0, 0, 0, 0])

        # Skip non-clinical classes (text = class 4)
        if class_name not in CLINICAL_CLASSES.values():
            # Try lookup by class_id
            if class_id not in CLINICAL_CLASSES:
                continue
            class_name = CLINICAL_CLASSES[class_id]

        # Skip detections below confidence threshold
        if confidence < confidence_threshold:
            continue

        findings_detected.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "box": [round(b, 4) for b in box],
        })
        detected_class_names.add(class_name)

    # Sort findings by confidence descending (most confident first)
    findings_detected.sort(key=lambda d: d["confidence"], reverse=True)

    # ── Build no_findings list ──
    # Lists clinical classes NOT detected above threshold.
    # IMPORTANT: "text" is NEVER in this list — it's infrastructure, not clinical.
    no_findings = [
        name for name in CLINICAL_CLASSES.values()
        if name not in detected_class_names
    ]

    # ── Build summary flags ──
    summary = {
        "fracture_suspected":           "fracture" in detected_class_names,
        "implant_present":              "metal_implant" in detected_class_names,
        "periosteal_reaction_present":  "periosteal_reaction" in detected_class_names,
        "pronator_sign_present":        "pronator_sign" in detected_class_names,
        "needs_human_review":           _compute_review_flag(
            findings_detected, detected_class_names, confidence_threshold
        ),
    }

    # ── Assemble final report ──
    report = {
        "request_id":               request_id,
        "model_version":            model_version,
        "timestamp":                timestamp,
        "laterality":               laterality,
        "raw_text_detected":        raw_text_detected,
        "findings_detected":        findings_detected,
        "no_findings":              no_findings,
        "summary":                  summary,
        "confidence_threshold_used": confidence_threshold,
        "latency_ms":               round(latency_ms, 1),
    }

    return report


def _compute_review_flag(
    findings: list[dict],
    detected_names: set[str],
    threshold: float,
) -> bool:
    """Determine whether the needs_human_review flag should be True.

    The flag is set to True in three scenarios:

    1. BORDERLINE CONFIDENCE: Any clinical detection has confidence between
       the threshold and threshold + _REVIEW_MARGIN (0.15). This means the
       model is uncertain enough to warrant a second look.

    2. LOW-CONFIDENCE FRACTURE: Fracture is detected but its highest
       confidence is below _FRACTURE_REVIEW_THRESHOLD (0.70). Fractures
       are the primary clinical target — borderline ones must be reviewed.

    3. NO FINDINGS: No clinical findings detected at all. A completely
       negative scan may indicate the model missed something, especially
       if the X-ray was taken after a trauma presentation.

    Args:
        findings:       List of clinical finding dicts (above threshold)
        detected_names: Set of detected clinical class names
        threshold:      Confidence threshold used for detection filtering

    Returns:
        bool: True if human review is recommended
    """
    # Scenario 3: No clinical findings at all
    if not findings:
        return True

    for finding in findings:
        conf = finding["confidence"]

        # Scenario 1: Borderline confidence
        if conf < (threshold + _REVIEW_MARGIN):
            return True

        # Scenario 2: Low-confidence fracture
        if finding["class_name"] == "fracture" and conf < _FRACTURE_REVIEW_THRESHOLD:
            return True

    return False


def format_report_text(report: dict) -> str:
    """Format a clinical report as human-readable text for display.

    Used in the demo notebook to print reports in a readable format
    without requiring the user to parse raw JSON.

    Args:
        report: Clinical report dict from generate_report()

    Returns:
        str: Formatted multi-line text representation
    """
    lines = [
        "=" * 60,
        "  WRISTASSIST AI — CLINICAL REPORT",
        "=" * 60,
        f"  Request ID:    {report['request_id']}",
        f"  Model:         {report['model_version']}",
        f"  Timestamp:     {report['timestamp']}",
        f"  Laterality:    {report['laterality']}",
        f"  Latency:       {report['latency_ms']} ms",
        f"  Threshold:     {report['confidence_threshold_used']}",
        "",
        "  FINDINGS DETECTED:",
    ]

    if report["findings_detected"]:
        for f in report["findings_detected"]:
            lines.append(
                f"    • {f['class_name']:<25} "
                f"conf={f['confidence']:.3f}  "
                f"box={f['box']}"
            )
    else:
        lines.append("    (none)")

    lines.append("")
    lines.append("  NOT DETECTED (no findings):")
    if report["no_findings"]:
        for nf in report["no_findings"]:
            lines.append(f"    • {nf}")
    else:
        lines.append("    (all clinical classes detected)")

    lines.append("")
    lines.append("  SUMMARY FLAGS:")
    for key, val in report["summary"].items():
        icon = "🔴" if val else "🟢"
        lines.append(f"    {icon} {key}: {val}")

    if report.get("raw_text_detected"):
        lines.append("")
        lines.append(f"  Raw OCR text: {report['raw_text_detected']}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("Running report.py self-test...\n")

    # Test 1: Fracture case with laterality
    dets = [
        {"class_id": 0, "class_name": "fracture", "confidence": 0.91, "box": [0.18, 0.42, 0.33, 0.58]},
        {"class_id": 2, "class_name": "periosteal_reaction", "confidence": 0.62, "box": [0.28, 0.30, 0.42, 0.40]},
        {"class_id": 4, "class_name": "text", "confidence": 0.95, "box": [0.02, 0.02, 0.08, 0.06]},
    ]
    report = generate_report(dets, laterality="Left", raw_text_detected=["L"],
                             confidence_threshold=0.42, latency_ms=342.5)
    print("Test 1 — Fracture case:")
    print(format_report_text(report))
    assert report["laterality"] == "Left"
    assert report["summary"]["fracture_suspected"] is True
    assert "metal_implant" in report["no_findings"]
    assert "text" not in report["no_findings"]  # text is NEVER in no_findings
    print("✅ Test 1 passed\n")

    # Test 2: Negative case (no findings)
    dets_neg = [
        {"class_id": 4, "class_name": "text", "confidence": 0.88, "box": [0.01, 0.01, 0.05, 0.05]},
    ]
    report_neg = generate_report(dets_neg, laterality="Right", raw_text_detected=["R"],
                                 confidence_threshold=0.42, latency_ms=285.0)
    print("Test 2 — Negative case:")
    print(format_report_text(report_neg))
    assert len(report_neg["findings_detected"]) == 0
    assert len(report_neg["no_findings"]) == 4  # All 4 clinical classes
    assert report_neg["summary"]["needs_human_review"] is True  # No findings → review
    print("✅ Test 2 passed\n")

    # Test 3: JSON serialisation
    json_str = json.dumps(report, indent=2)
    print("Test 3 — JSON serialisation:")
    print(json_str[:200] + "...")
    assert json.loads(json_str)["request_id"] == report["request_id"]
    print("✅ Test 3 passed")

    print("\n✅ All self-tests passed")