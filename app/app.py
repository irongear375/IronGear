"""
WristAssist AI — Sprint 3 FastAPI Backend
SCRUM-47: Implement FastAPI backend with all PRD endpoints

Endpoints:
    GET  /          Serves index.html (web UI)
    GET  /health    Health check (CD smoke test target)
    POST /predict   Core inference — YOLO11l + EasyOCR laterality
    GET  /log       Recent prediction log entries
    POST /feedback  Clinician feedback capture
    GET  /metrics   Aggregate monitoring statistics (SCRUM-53)
    GET  /drift     PSI drift per class (SCRUM-54)

Production model: sprint2_yolo11l_expB_v1
Confidence threshold: 0.42 (calibrated from Sprint 2 F1 curve)
"""

import os
import io
import json
import time
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Configuration from environment ──
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "sprint2_yolo11l_expB_v1")
DEFAULT_CONF = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.42"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/data"))
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Class schema (locked from Sprint 2) ──
CLASS_NAMES = {0: "fracture", 1: "metal_implant", 2: "periosteal_reaction",
               3: "pronator_sign", 4: "text"}
# Clinical classes exclude 'text' (infrastructure class) from no_findings
CLINICAL_CLASSES = ["fracture", "metal_implant", "periosteal_reaction", "pronator_sign"]

# ── Bounding box colours (BGR for OpenCV) ──
CLASS_COLORS = {
    "fracture": (0, 0, 255),           # Red
    "metal_implant": (255, 165, 0),     # Orange
    "periosteal_reaction": (0, 215, 255),  # Gold
    "pronator_sign": (255, 0, 255),     # Magenta
    "text": (0, 255, 255),              # Yellow
}

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wristassist")

# ── Ensure log directory exists ──
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOG_DIR / "predictions.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

# ============================================================================
# JSON serialisation helpers (Sprint 2 lesson — float32 not serialisable)
# ============================================================================

def _json_default(obj):
    """Handle numpy/torch types in json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================================
# Model loading (once on startup, kept resident)
# ============================================================================

# These are populated in the lifespan handler
yolo_model = None
ocr_reader = None


# ============================================================================
# Image preprocessing (Sprint 2 patterns — do not deviate)
# ============================================================================

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to 8-bit BGR numpy array.
    Handles 16-bit GRAZPEDWRI-DX PNGs by normalising to 8-bit.
    All processing in-memory — never writes to disk.
    """
    # Decode from bytes
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Failed to decode image")

    # Handle 16-bit images (common in radiology)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ensure 3-channel BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ============================================================================
# Laterality extraction — CV-based (no EasyOCR in inference path)
# ============================================================================
#
# HISTORY OF THIS FUNCTION:
# Sprint 2 original:  EasyOCR × 16 calls → ~180s on CPU  (BROKEN)
# Sprint 3 attempt 1: EasyOCR × 3 calls  → ~177s on CPU  (STILL BROKEN)
# Sprint 3 attempt 2: EasyOCR × 1 call   → ~70s on CPU   (STILL TOO SLOW)
#
# ROOT CAUSE: EasyOCR is a deep-learning OCR engine. Even ONE readtext()
# call takes 60-70 seconds on HF Spaces free-tier CPU (8 vCPU, no GPU).
# This is not fixable with preprocessing — the model inference itself is slow.
#
# SOLUTION: Replace EasyOCR with lightweight computer-vision shape analysis.
# The X-ray text markers are lead letters ("L" or "R") with distinctive shapes:
#   - "L" has white pixels concentrated on the LEFT edge + BOTTOM edge
#   - "R" has white pixels concentrated on the LEFT edge + TOP-RIGHT area
# We threshold the crop, divide into quadrants, and compare pixel densities.
# Total time: <5 milliseconds (vs 70,000 milliseconds for EasyOCR).
#
# EasyOCR remains loaded for the /health warm-up but is NOT called during
# /predict. In a production deployment with GPU, EasyOCR would run in <1s
# and could be re-enabled.
# ============================================================================

def extract_laterality(img: np.ndarray, text_boxes: list) -> dict:
    """
    Extract wrist laterality from the detected text bounding box using
    shape analysis. No EasyOCR call — runs in <5ms on any CPU.

    Method: Binarise the text crop, then analyse the spatial distribution
    of foreground pixels to distinguish "L" from "R" letter shapes.
    """
    if not text_boxes:
        return {"laterality": "Unknown", "raw_text_detected": [], "method": "no_text_box"}

    # ── 1. Pick the best text box (highest confidence = first in list) ──
    box = text_boxes[0]
    x1, y1, x2, y2 = [int(c) for c in box]
    h, w = img.shape[:2]

    # Add padding to capture full marker
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * 0.3)
    pad_y = int(box_h * 0.3)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return {"laterality": "Unknown", "raw_text_detected": [], "method": "invalid_box"}

    crop = img[y1:y2, x1:x2]

    # ── 2. Preprocess to clean binary image (<5ms total) ──
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold to get clean binary
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure the letter is white-on-black (foreground = 255)
    # If more than half the pixels are white, invert (background was white)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # ── 3. Shape analysis to distinguish L from R ──
    bh, bw = binary.shape
    if bh < 5 or bw < 5:
        return {"laterality": "Unknown", "raw_text_detected": [], "method": "crop_too_small"}

    # Divide into quadrants
    mid_y = bh // 2
    mid_x = bw // 2

    top_left     = np.sum(binary[0:mid_y, 0:mid_x] > 0)
    top_right    = np.sum(binary[0:mid_y, mid_x:bw] > 0)
    bottom_left  = np.sum(binary[mid_y:bh, 0:mid_x] > 0)
    bottom_right = np.sum(binary[mid_y:bh, mid_x:bw] > 0)

    total_pixels = max(top_left + top_right + bottom_left + bottom_right, 1)

    # Compute ratios
    top_right_ratio    = top_right / total_pixels
    bottom_right_ratio = bottom_right / total_pixels
    bottom_left_ratio  = bottom_left / total_pixels

    # "L" shape signature: strong bottom-left and bottom-right, weak top-right
    #   The vertical stroke fills left column, horizontal stroke fills bottom row
    #   top_right is mostly empty
    #
    # "R" shape signature: strong top-right (the bump), moderate bottom-right (the leg)
    #   The vertical stroke fills left column, the curved part fills top-right
    #   bottom is more balanced

    # Score each letter hypothesis
    l_score = bottom_left_ratio + bottom_right_ratio - top_right_ratio
    r_score = top_right_ratio + bottom_right_ratio - bottom_left_ratio

    # Also check: "L" has very little ink in top-right quadrant
    # "R" has significant ink in top-right quadrant
    top_right_is_empty = top_right_ratio < 0.15

    # Decision with confidence threshold
    min_difference = 0.08  # Need at least 8% difference to be confident
    raw_info = [f"L_score={l_score:.2f}, R_score={r_score:.2f}, TR_ratio={top_right_ratio:.2f}"]

    if l_score > r_score + min_difference and top_right_is_empty:
        laterality = "Left"
        logger.info(f"Laterality → Left (shape: L_score={l_score:.2f}, R_score={r_score:.2f})")
    elif r_score > l_score + min_difference and not top_right_is_empty:
        laterality = "Right"
        logger.info(f"Laterality → Right (shape: L_score={l_score:.2f}, R_score={r_score:.2f})")
    else:
        laterality = "Unknown"
        logger.info(f"Laterality → Unknown (inconclusive: L={l_score:.2f}, R={r_score:.2f})")

    return {"laterality": laterality, "raw_text_detected": raw_info, "method": "shape_analysis"}


# ============================================================================
# Annotated image rendering
# ============================================================================

def render_annotated_image(img: np.ndarray, detections: list) -> str:
    """
    Draw bounding boxes and labels on the image.
    Returns base64-encoded PNG string.
    """
    annotated = img.copy()

    for det in detections:
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(c) for c in det["box"]]
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Encode to base64 PNG
    _, buffer = cv2.imencode(".png", annotated)
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================================
# Clinical report assembly
# ============================================================================

def assemble_clinical_report(
    detections: list,
    laterality_info: dict,
    conf_threshold: float,
    latency_ms: float,
) -> dict:
    """
    Build the structured clinical JSON report per PRD section 5.1.3.
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Separate clinical findings from text detections
    findings = [d for d in detections if d["class_name"] != "text"]

    # Determine which clinical classes were NOT detected
    detected_classes = {d["class_name"] for d in findings}
    no_findings = [c for c in CLINICAL_CLASSES if c not in detected_classes]

    # Summary flags
    summary = {
        "fracture_suspected": "fracture" in detected_classes,
        "implant_present": "metal_implant" in detected_classes,
        "periosteal_reaction_present": "periosteal_reaction" in detected_classes,
        "pronator_sign_present": "pronator_sign" in detected_classes,
        "needs_human_review": any(
            d["confidence"] < 0.6 for d in findings
        ) or len(findings) == 0,
    }

    return {
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "timestamp": timestamp,
        "latency_ms": round(latency_ms, 1),
        "laterality": laterality_info["laterality"],
        "raw_text_detected": laterality_info["raw_text_detected"],
        "findings_detected": findings,
        "no_findings": no_findings,
        "summary": summary,
        "confidence_threshold_used": conf_threshold,
    }


# ============================================================================
# Prediction logging
# ============================================================================

def log_prediction(report: dict):
    """Append one JSON line per prediction to predictions.jsonl."""
    log_entry = {
        "timestamp": report["timestamp"],
        "request_id": report["request_id"],
        "model_version": report["model_version"],
        "latency_ms": report["latency_ms"],
        "laterality": report["laterality"],
        "num_findings": len(report["findings_detected"]),
        "classes_detected": [d["class_name"] for d in report["findings_detected"]],
        "confidences": {d["class_name"]: d["confidence"] for d in report["findings_detected"]},
        "needs_human_review": report["summary"]["needs_human_review"],
        "confidence_threshold": report["confidence_threshold_used"],
    }
    try:
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(log_entry, default=_json_default) + "\n")
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


# ============================================================================
# FastAPI application
# ============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load models on startup, release on shutdown."""
    global yolo_model, ocr_reader

    logger.info(f"Loading YOLO model from {MODEL_PATH}...")
    from ultralytics import YOLO
    yolo_model = YOLO(MODEL_PATH)
    logger.info(f"YOLO model loaded: {MODEL_VERSION}")

    logger.info("Initialising EasyOCR reader (English, CPU)...")
    import easyocr
    ocr_reader = easyocr.Reader(["en"], gpu=False, download_enabled=False)
    logger.info("EasyOCR reader loaded (available but not used in inference — shape analysis is faster)")

    # Warm up YOLO only (OCR not in inference path — shape analysis used instead)
    logger.info("Warming up YOLO model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = yolo_model(dummy, verbose=False)
    logger.info("YOLO warmed up — ready for predictions")

    yield  # Application runs here

    logger.info("Shutting down — releasing models")
    yolo_model = None
    ocr_reader = None


app = FastAPI(
    title="WristAssist AI",
    description="Paediatric wrist trauma X-ray detection system. Decision support only.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Pydantic response models ──

class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    timestamp: str

class FeedbackRequest(BaseModel):
    request_id: str
    correct: bool
    comment: Optional[str] = ""

class FeedbackResponse(BaseModel):
    status: str = "recorded"
    request_id: str


# ============================================================================
# Endpoints
# ============================================================================

# ── GET / — Serve web UI ──
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page HTML UI."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>WristAssist AI</h1><p>UI not found. API is running at /docs</p>",
            status_code=200,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── GET /health — Health check (CD smoke test target) ──
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint. Returns model version and status."""
    return HealthResponse(
        status="ok" if yolo_model is not None else "model_not_loaded",
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── POST /predict — Core inference ──
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf_threshold: float = Form(default=DEFAULT_CONF),
):
    """
    Core inference endpoint.
    Accepts a paediatric wrist X-ray image (PNG/JPG, max 10 MB).
    Returns clinical JSON report with annotated image overlay.
    """
    start_time = time.time()

    # ── Input validation ──
    # MIME type check
    allowed_types = {"image/png", "image/jpeg", "image/jpg"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {image.content_type}. Expected PNG or JPEG.",
        )

    # File extension check
    ext = Path(image.filename or "").suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}. Expected .png, .jpg, or .jpeg.",
        )

    # Read file bytes (in memory — never written to disk)
    file_bytes = await image.read()

    # Size check
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB. Maximum is 10 MB.",
        )

    # ── Preprocess ──
    try:
        img = preprocess_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── YOLO inference ──
    results = yolo_model(img, conf=conf_threshold, verbose=False)
    result = results[0]

    # Parse detections
    detections = []
    text_boxes = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = round(float(box.conf[0]), 4)
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixels

        cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

        det = {
            "class_name": cls_name,
            "confidence": confidence,
            "box": [round(c, 1) for c in xyxy],
        }
        detections.append(det)

        # Collect text boxes for OCR
        if cls_name == "text":
            text_boxes.append(xyxy)

    # ── EasyOCR laterality extraction ──
    laterality_info = extract_laterality(img, text_boxes)

    # ── Assemble clinical report ──
    latency_ms = (time.time() - start_time) * 1000
    report = assemble_clinical_report(detections, laterality_info, conf_threshold, latency_ms)

    # ── Render annotated image ──
    annotated_b64 = render_annotated_image(img, detections)
    report["annotated_image"] = annotated_b64

    # ── Log prediction ──
    log_prediction(report)

    logger.info(
        f"Prediction {report['request_id']}: "
        f"{len(report['findings_detected'])} findings, "
        f"laterality={report['laterality']}, "
        f"latency={report['latency_ms']:.0f}ms"
    )

    return JSONResponse(content=json.loads(json.dumps(report, default=_json_default)))


# ── GET /log — Recent prediction log entries ──
@app.get("/log")
async def get_log(n: int = Query(default=10, ge=1, le=100)):
    """Return the last N prediction log entries."""
    if not PREDICTIONS_LOG.exists():
        return []

    try:
        with open(PREDICTIONS_LOG, "r") as f:
            lines = f.readlines()
        # Return last N entries, newest first
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        entries.reverse()
        return entries
    except Exception as e:
        logger.error(f"Failed to read prediction log: {e}")
        raise HTTPException(status_code=500, detail="Failed to read prediction log")


# ── POST /feedback — Clinician feedback capture ──
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Record clinician feedback (correct/incorrect + optional comment)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": feedback.request_id,
        "correct": feedback.correct,
        "comment": feedback.comment,
    }
    try:
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

    return FeedbackResponse(status="recorded", request_id=feedback.request_id)


# ── GET /metrics — Aggregate monitoring statistics ──
@app.get("/metrics")
async def get_metrics():
    """
    Aggregate monitoring statistics: total predictions, latency percentiles,
    per-class rates, laterality distribution, human review rate.
    """
    try:
        from monitoring import compute_metrics
        return compute_metrics()
    except ImportError:
        return {
            "status": "not_implemented",
            "message": "Monitoring available after SCRUM-53 implementation.",
            "total_predictions": 0,
        }
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {"status": "error", "detail": str(e)}


# ── GET /drift — PSI drift per class ──
@app.get("/drift")
async def get_drift():
    """
    Drift detection dashboard.
    Placeholder — full implementation in SCRUM-54 (drift.py).
    """
    try:
        from drift import compute_drift
        return compute_drift()
    except ImportError:
        return {
            "status": "not_implemented",
            "message": "Drift detection available after SCRUM-54 implementation.",
            "psi_per_class": {},
        }


# ============================================================================
# Main entry point (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
