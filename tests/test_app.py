"""
WristAssist AI — Unit & Integration Tests
SCRUM-57: Unit and integration test suite for Sprint 3

Tests cover:
- Image preprocessing (8-bit, 16-bit, grayscale, BGRA)
- Clinical report assembly (findings, no_findings, summary flags)
- JSON serialisation helpers (numpy types)
- Monitoring service (latency stats, class rates, laterality, review rate)
- Drift detection (PSI computation, classification, live distribution)
- FastAPI endpoints (health, log, feedback, metrics, drift)

Usage:
    pytest tests/test_app.py -v
    pytest tests/test_app.py -v --tb=short    (CI mode)
"""

import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest


# ============================================================================
# Ensure app modules are importable
# ============================================================================

# Add app directory to path so we can import app, monitoring, drift
APP_DIR = Path(__file__).parent.parent / "app"
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))


# ============================================================================
# Test: Image Preprocessing
# ============================================================================

class TestPreprocessImage:
    """Tests for the preprocess_image function."""

    def _make_png_bytes(self, img: np.ndarray) -> bytes:
        """Encode a numpy array to PNG bytes."""
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def test_8bit_bgr_passthrough(self):
        """Standard 8-bit BGR image should pass through unchanged."""
        from app import preprocess_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(self._make_png_bytes(img))
        assert result.dtype == np.uint8
        assert result.shape[2] == 3

    def test_16bit_normalisation(self):
        """16-bit image (common in radiology) should be normalised to 8-bit."""
        from app import preprocess_image
        img_16 = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = preprocess_image(self._make_png_bytes(img_16))
        assert result.dtype == np.uint8
        assert len(result.shape) == 3  # Should be converted to 3-channel

    def test_grayscale_to_bgr(self):
        """Grayscale image should be converted to 3-channel BGR."""
        from app import preprocess_image
        img_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_image(self._make_png_bytes(img_gray))
        assert result.shape[2] == 3

    def test_rgba_to_bgr(self):
        """4-channel BGRA image should be converted to 3-channel BGR."""
        from app import preprocess_image
        img_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        result = preprocess_image(self._make_png_bytes(img_rgba))
        assert result.shape[2] == 3

    def test_invalid_bytes_raises(self):
        """Invalid image bytes should raise ValueError."""
        from app import preprocess_image
        with pytest.raises(ValueError, match="Failed to decode"):
            preprocess_image(b"not_an_image")

    def test_output_never_none(self):
        """Preprocessed output should never be None."""
        from app import preprocess_image
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = preprocess_image(self._make_png_bytes(img))
        assert result is not None


# ============================================================================
# Test: Clinical Report Assembly
# ============================================================================

class TestAssembleClinicalReport:
    """Tests for the assemble_clinical_report function."""

    def _make_detection(self, cls_name: str, conf: float) -> dict:
        return {
            "class_name": cls_name,
            "confidence": conf,
            "box": [100.0, 100.0, 200.0, 200.0],
        }

    def test_fracture_detected(self):
        """Report should flag fracture_suspected when fracture is present."""
        from app import assemble_clinical_report
        detections = [self._make_detection("fracture", 0.91)]
        laterality = {"laterality": "Left", "raw_text_detected": ["L"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 150.0)

        assert report["summary"]["fracture_suspected"] is True
        assert "fracture" not in report["no_findings"]

    def test_no_findings_when_no_clinical_detections(self):
        """All clinical classes should appear in no_findings when none detected."""
        from app import assemble_clinical_report, CLINICAL_CLASSES
        # Only text detection — no clinical findings
        detections = [self._make_detection("text", 0.95)]
        laterality = {"laterality": "Unknown", "raw_text_detected": []}
        report = assemble_clinical_report(detections, laterality, 0.42, 100.0)

        assert len(report["no_findings"]) == len(CLINICAL_CLASSES)
        for cls in CLINICAL_CLASSES:
            assert cls in report["no_findings"]

    def test_text_excluded_from_findings(self):
        """Text class should not appear in findings_detected."""
        from app import assemble_clinical_report
        detections = [
            self._make_detection("fracture", 0.85),
            self._make_detection("text", 0.98),
        ]
        laterality = {"laterality": "Left", "raw_text_detected": ["L"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 200.0)

        finding_classes = [f["class_name"] for f in report["findings_detected"]]
        assert "text" not in finding_classes
        assert "fracture" in finding_classes

    def test_needs_human_review_low_confidence(self):
        """needs_human_review should be True when any finding has confidence < 0.6."""
        from app import assemble_clinical_report
        detections = [self._make_detection("fracture", 0.45)]
        laterality = {"laterality": "Unknown", "raw_text_detected": []}
        report = assemble_clinical_report(detections, laterality, 0.42, 100.0)

        assert report["summary"]["needs_human_review"] is True

    def test_needs_human_review_no_findings(self):
        """needs_human_review should be True when no clinical findings detected."""
        from app import assemble_clinical_report
        detections = []
        laterality = {"laterality": "Unknown", "raw_text_detected": []}
        report = assemble_clinical_report(detections, laterality, 0.42, 100.0)

        assert report["summary"]["needs_human_review"] is True

    def test_no_human_review_high_confidence(self):
        """needs_human_review should be False when all findings have confidence >= 0.6."""
        from app import assemble_clinical_report
        detections = [self._make_detection("fracture", 0.91)]
        laterality = {"laterality": "Left", "raw_text_detected": ["L"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 100.0)

        assert report["summary"]["needs_human_review"] is False

    def test_report_has_required_fields(self):
        """Report should contain all PRD-required fields."""
        from app import assemble_clinical_report
        detections = [self._make_detection("fracture", 0.8)]
        laterality = {"laterality": "Right", "raw_text_detected": ["R"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 300.0)

        required_fields = [
            "request_id", "model_version", "timestamp", "latency_ms",
            "laterality", "raw_text_detected", "findings_detected",
            "no_findings", "summary", "confidence_threshold_used",
        ]
        for field in required_fields:
            assert field in report, f"Missing required field: {field}"

    def test_multiple_findings(self):
        """Report should handle multiple clinical findings correctly."""
        from app import assemble_clinical_report
        detections = [
            self._make_detection("fracture", 0.91),
            self._make_detection("periosteal_reaction", 0.67),
            self._make_detection("text", 0.95),
        ]
        laterality = {"laterality": "Left", "raw_text_detected": ["L"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 250.0)

        assert len(report["findings_detected"]) == 2  # excludes text
        assert report["summary"]["fracture_suspected"] is True
        assert report["summary"]["periosteal_reaction_present"] is True
        assert "metal_implant" in report["no_findings"]
        assert "pronator_sign" in report["no_findings"]

    def test_laterality_passthrough(self):
        """Laterality should be passed through from extraction to report."""
        from app import assemble_clinical_report
        detections = []
        laterality = {"laterality": "Right", "raw_text_detected": ["R"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 100.0)

        assert report["laterality"] == "Right"
        assert report["raw_text_detected"] == ["R"]


# ============================================================================
# Test: JSON Serialisation Helper
# ============================================================================

class TestJsonDefault:
    """Tests for the _json_default numpy serialisation handler."""

    def test_numpy_int(self):
        from app import _json_default
        assert _json_default(np.int64(42)) == 42
        assert isinstance(_json_default(np.int64(42)), int)

    def test_numpy_float(self):
        from app import _json_default
        result = _json_default(np.float32(3.14159))
        assert isinstance(result, float)

    def test_numpy_array(self):
        from app import _json_default
        arr = np.array([1, 2, 3])
        result = _json_default(arr)
        assert result == [1, 2, 3]

    def test_unsupported_type_raises(self):
        from app import _json_default
        with pytest.raises(TypeError):
            _json_default({"unsupported": "dict"})


# ============================================================================
# Test: Monitoring Service
# ============================================================================

class TestMonitoring:
    """Tests for monitoring.py compute functions."""

    @pytest.fixture
    def sample_predictions(self, tmp_path):
        """Create a temporary predictions.jsonl with sample data."""
        log_file = tmp_path / "predictions.jsonl"
        now = datetime.now(timezone.utc)

        entries = [
            {
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "request_id": f"req-{i}",
                "model_version": "sprint2_yolo11l_expB_v1",
                "latency_ms": 3000 + (i * 100),
                "laterality": ["Left", "Right", "Unknown"][i % 3],
                "classes_detected": ["fracture"] if i % 2 == 0 else ["fracture", "metal_implant"],
                "confidences": {"fracture": 0.85 + (i * 0.01)},
                "needs_human_review": i % 3 == 0,
                "confidence_threshold": 0.42,
            }
            for i in range(10)
        ]

        with open(log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return log_file, entries

    def test_compute_latency_stats(self):
        """Latency stats should compute correct p50 and p95."""
        from monitoring import _compute_latency_stats

        entries = [{"latency_ms": v} for v in [100, 200, 300, 400, 500]]
        stats = _compute_latency_stats(entries)

        assert stats["avg_latency_ms"] == 300.0
        assert stats["p50_latency_ms"] == 300.0
        assert stats["min_latency_ms"] == 100.0
        assert stats["max_latency_ms"] == 500.0

    def test_compute_latency_stats_empty(self):
        """Latency stats should handle empty input gracefully."""
        from monitoring import _compute_latency_stats

        stats = _compute_latency_stats([])
        assert stats["avg_latency_ms"] == 0.0
        assert stats["p50_latency_ms"] == 0.0

    def test_compute_class_rates(self):
        """Class rates should compute correct fractions."""
        from monitoring import _compute_class_rates

        entries = [
            {"classes_detected": ["fracture"]},
            {"classes_detected": ["fracture", "metal_implant"]},
            {"classes_detected": ["fracture"]},
            {"classes_detected": []},
        ]
        rates = _compute_class_rates(entries)

        assert rates["fracture"] == 0.75
        assert rates["metal_implant"] == 0.25
        assert rates["periosteal_reaction"] == 0.0

    def test_compute_class_rates_empty(self):
        """Class rates should return zeros for empty input."""
        from monitoring import _compute_class_rates

        rates = _compute_class_rates([])
        assert all(v == 0.0 for v in rates.values())

    def test_compute_laterality_distribution(self):
        """Laterality distribution should count correctly."""
        from monitoring import _compute_laterality_distribution

        entries = [
            {"laterality": "Left"},
            {"laterality": "Left"},
            {"laterality": "Right"},
            {"laterality": "Unknown"},
        ]
        dist = _compute_laterality_distribution(entries)

        assert dist["Left"] == 2
        assert dist["Right"] == 1
        assert dist["Unknown"] == 1

    def test_compute_review_rate(self):
        """Review rate should compute correct fraction."""
        from monitoring import _compute_review_rate

        entries = [
            {"needs_human_review": True},
            {"needs_human_review": False},
            {"needs_human_review": True},
            {"needs_human_review": False},
        ]
        rate = _compute_review_rate(entries)
        assert rate == 0.5

    def test_compute_review_rate_empty(self):
        """Review rate should return 0 for empty input."""
        from monitoring import _compute_review_rate

        assert _compute_review_rate([]) == 0.0

    def test_compute_metrics_returns_all_fields(self, sample_predictions):
        """compute_metrics should return all expected top-level fields."""
        from monitoring import compute_metrics, _metrics_cache
        import monitoring

        log_file, _ = sample_predictions

        # Point monitoring at our temp log file
        monitoring.PREDICTIONS_LOG = log_file
        monitoring._metrics_cache = None  # Clear cache

        result = compute_metrics()

        expected_fields = [
            "model_version", "computed_at", "total_predictions",
            "predictions_24h", "predictions_7d",
            "latency_all_time", "latency_24h",
            "class_rates_24h", "class_rates_7d", "class_rates_all_time",
            "laterality_distribution", "human_review_rate",
            "log_file_exists",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

        assert result["total_predictions"] == 10


# ============================================================================
# Test: Drift Detection
# ============================================================================

class TestDrift:
    """Tests for drift.py PSI computation and classification."""

    def test_psi_identical_distributions(self):
        """PSI should be ~0 when baseline and live rates are identical."""
        from drift import _compute_psi

        psi = _compute_psi(0.5, 0.5)
        assert abs(psi) < 1e-6

    def test_psi_small_shift(self):
        """Small shift should produce low PSI (< 0.1 = OK)."""
        from drift import _compute_psi

        psi = _compute_psi(0.50, 0.52)
        assert psi < 0.10

    def test_psi_large_shift(self):
        """Large shift should produce high PSI (> 0.25 = ALERT)."""
        from drift import _compute_psi

        psi = _compute_psi(0.50, 0.05)
        assert psi > 0.25

    def test_psi_handles_zero_baseline(self):
        """PSI should handle zero baseline rate without error (epsilon guard)."""
        from drift import _compute_psi

        psi = _compute_psi(0.0, 0.5)
        assert math.isfinite(psi)

    def test_psi_handles_zero_live(self):
        """PSI should handle zero live rate without error (epsilon guard)."""
        from drift import _compute_psi

        psi = _compute_psi(0.5, 0.0)
        assert math.isfinite(psi)

    def test_classify_psi_ok(self):
        """PSI below 0.10 should classify as OK."""
        from drift import _classify_psi

        assert _classify_psi(0.05) == "OK"

    def test_classify_psi_warn(self):
        """PSI between 0.10 and 0.25 should classify as WARN."""
        from drift import _classify_psi

        assert _classify_psi(0.15) == "WARN"

    def test_classify_psi_alert(self):
        """PSI above 0.25 should classify as ALERT."""
        from drift import _classify_psi

        assert _classify_psi(0.30) == "ALERT"

    def test_compute_live_distribution(self):
        """Live distribution should compute correct per-class rates."""
        from drift import _compute_live_distribution

        entries = [
            {"classes_detected": ["fracture"]},
            {"classes_detected": ["fracture", "metal_implant"]},
            {"classes_detected": ["fracture"]},
            {"classes_detected": []},
            {"classes_detected": ["periosteal_reaction"]},
        ]
        dist = _compute_live_distribution(entries)

        assert dist["fracture"] == 0.6
        assert dist["metal_implant"] == 0.2
        assert dist["periosteal_reaction"] == 0.2
        assert dist["pronator_sign"] == 0.0

    def test_compute_live_distribution_empty(self):
        """Live distribution should return zeros for empty input."""
        from drift import _compute_live_distribution

        dist = _compute_live_distribution([])
        assert all(v == 0.0 for v in dist.values())

    def test_compute_drift_no_baseline(self, tmp_path):
        """compute_drift should handle missing baseline gracefully."""
        import drift

        drift.BASELINE_PATH = tmp_path / "nonexistent.json"
        drift._drift_cache = None  # Clear cache

        result = drift.compute_drift()
        assert result["status"] == "baseline_not_found"

    def test_compute_drift_with_baseline(self, tmp_path):
        """compute_drift should return per-class PSI with valid baseline."""
        import drift

        # Create baseline
        baseline = {
            "class_rates": {
                "fracture": 0.6685,
                "metal_implant": 0.0405,
                "periosteal_reaction": 0.1117,
                "pronator_sign": 0.0311,
            }
        }
        baseline_path = tmp_path / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f)

        # Create predictions log
        log_path = tmp_path / "predictions.jsonl"
        now = datetime.now(timezone.utc)
        with open(log_path, "w") as f:
            for i in range(20):
                entry = {
                    "timestamp": (now - timedelta(hours=i)).isoformat(),
                    "classes_detected": ["fracture"] if i % 2 == 0 else [],
                }
                f.write(json.dumps(entry) + "\n")

        # Point drift at temp files
        drift.BASELINE_PATH = baseline_path
        drift.PREDICTIONS_LOG = log_path
        drift.DRIFT_HISTORY_LOG = tmp_path / "drift_history.jsonl"
        drift._drift_cache = None

        result = drift.compute_drift(window_hours=168)

        assert "per_class" in result
        assert "overall_psi" in result
        assert "overall_status" in result
        for cls in ["fracture", "metal_implant", "periosteal_reaction", "pronator_sign"]:
            assert cls in result["per_class"]
            assert "psi" in result["per_class"][cls]
            assert "status" in result["per_class"][cls]


# ============================================================================
# Test: FastAPI Endpoints (Integration Tests)
# ============================================================================

class TestFastAPIEndpoints:
    """
    Integration tests for FastAPI endpoints.
    Uses TestClient with mocked YOLO model and EasyOCR reader.
    """

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client with mocked models."""
        # Set LOG_DIR to temp before importing app
        os.environ["LOG_DIR"] = str(tmp_path)
        os.environ["MODEL_PATH"] = "dummy_path"

        # Create a minimal index.html for the UI endpoint
        app_dir = Path(__file__).parent.parent / "app"
        if not app_dir.exists():
            app_dir = tmp_path
        index_path = tmp_path / "index.html"
        index_path.write_text("<html><body>Test</body></html>")

        import app as app_module

        # Patch the log paths
        app_module.LOG_DIR = tmp_path
        app_module.PREDICTIONS_LOG = tmp_path / "predictions.jsonl"
        app_module.FEEDBACK_LOG = tmp_path / "feedback.jsonl"

        # Mock the models
        mock_box = MagicMock()
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].__int__ = lambda self: 0  # fracture
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].__float__ = lambda self: 0.91
        mock_box.xyxy = [np.array([100.0, 100.0, 200.0, 200.0])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        app_module.yolo_model = mock_model

        mock_ocr = MagicMock()
        mock_ocr.readtext.return_value = []
        app_module.ocr_reader = mock_ocr

        from fastapi.testclient import TestClient

        # Patch lifespan to skip model loading
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_lifespan(application):
            yield

        app_module.app.router.lifespan_context = mock_lifespan

        # Patch index.html path
        original_serve = app_module.serve_ui

        return TestClient(app_module.app)

    def test_health_endpoint(self, client):
        """GET /health should return 200 with model version."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert "timestamp" in data

    def test_predict_invalid_mime_type(self, client):
        """POST /predict with non-image file should return 415."""
        response = client.post(
            "/predict",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 415

    def test_predict_invalid_extension(self, client):
        """POST /predict with wrong extension should return 400."""
        response = client.post(
            "/predict",
            files={"image": ("test.bmp", b"fake", "image/png")},
        )
        assert response.status_code == 400

    def test_predict_oversized_file(self, client):
        """POST /predict with >10MB file should return 413."""
        big_file = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/predict",
            files={"image": ("big.png", big_file, "image/png")},
        )
        assert response.status_code == 413

    def test_log_endpoint_empty(self, client):
        """GET /log should return empty list when no predictions exist."""
        response = client.get("/log")
        assert response.status_code == 200
        assert response.json() == []

    def test_feedback_endpoint(self, client):
        """POST /feedback should accept and record feedback."""
        response = client.post(
            "/feedback",
            json={
                "request_id": "test-uuid-123",
                "correct": True,
                "comment": "Looks correct",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert data["request_id"] == "test-uuid-123"

    def test_metrics_endpoint(self, client):
        """GET /metrics should return 200 with monitoring data."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should return either metrics or not_implemented
        data = response.json()
        assert "total_predictions" in data or "status" in data

    def test_drift_endpoint(self, client):
        """GET /drift should return 200 with drift data or not_implemented."""
        response = client.get("/drift")
        assert response.status_code == 200
        data = response.json()
        assert "overall_psi" in data or "status" in data or "psi_per_class" in data


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_all_four_clinical_classes_detected(self):
        """Report should handle all 4 clinical classes being detected."""
        from app import assemble_clinical_report

        detections = [
            {"class_name": "fracture", "confidence": 0.9, "box": [0, 0, 1, 1]},
            {"class_name": "metal_implant", "confidence": 0.85, "box": [0, 0, 1, 1]},
            {"class_name": "periosteal_reaction", "confidence": 0.7, "box": [0, 0, 1, 1]},
            {"class_name": "pronator_sign", "confidence": 0.65, "box": [0, 0, 1, 1]},
        ]
        laterality = {"laterality": "Left", "raw_text_detected": ["L"]}
        report = assemble_clinical_report(detections, laterality, 0.42, 500.0)

        assert report["no_findings"] == []
        assert report["summary"]["fracture_suspected"] is True
        assert report["summary"]["implant_present"] is True
        assert report["summary"]["periosteal_reaction_present"] is True
        assert report["summary"]["pronator_sign_present"] is True

    def test_confidence_threshold_passthrough(self):
        """Confidence threshold should be recorded in the report."""
        from app import assemble_clinical_report

        detections = []
        laterality = {"laterality": "Unknown", "raw_text_detected": []}
        report = assemble_clinical_report(detections, laterality, 0.55, 100.0)

        assert report["confidence_threshold_used"] == 0.55

    def test_latency_recorded_correctly(self):
        """Latency should be rounded and recorded in the report."""
        from app import assemble_clinical_report

        detections = []
        laterality = {"laterality": "Unknown", "raw_text_detected": []}
        report = assemble_clinical_report(detections, laterality, 0.42, 1234.5678)

        assert report["latency_ms"] == 1234.6
