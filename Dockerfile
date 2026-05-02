# ============================================================================
# WristAssist AI — Sprint 3 Dockerfile
# SCRUM-46: Containerise FastAPI backend with Docker
# ============================================================================
# Target: Hugging Face Spaces (Docker SDK, free tier)
#   - Port 7860 (required by HF Spaces)
#   - CPU-only inference (no GPU on free tier)
#   - EasyOCR weights pre-cached at build time
#   - best.pt bundled at build time
#   - Image size target: < 3 GB
# ============================================================================

FROM python:3.11-slim AS base

# ── System dependencies ──
# libgl1 + libglib2 required by opencv-python-headless
# libgomp1 required by PyTorch for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces best practice) ──
RUN useradd -m -u 1000 appuser

# ── Install Python dependencies ──
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch CPU-only first to avoid pulling CUDA (~2 GB savings)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt



# ── Application code ──
WORKDIR /app

# Copy model weights (Experiment B best.pt, ~48 MB)
COPY Sprint2/models/best.pt /app/models/best.pt

# Copy Sprint 2 utility modules (ocr.py, report.py)
COPY Sprint2/utils/ /app/utils/

# Copy Sprint 3 application code
COPY Sprint3/app/ /app/

# Copy drift baseline (produced from Sprint 2 test set evaluation)
# This file may not exist yet during initial scaffold — build won't fail
COPY Sprint3/app/baseline_distribution.jso[n] /app/baseline_distribution.json

# ── HF Spaces persistent storage ──
# /data/ is HF persistent storage — survives container restarts
# predictions.jsonl and feedback.jsonl are written here at runtime
RUN mkdir -p /data && chown appuser:appuser /data

# ── Set ownership and switch to non-root ──
RUN chown -R appuser:appuser /app
USER appuser

# ── Pre-cache EasyOCR model weights at build time ──
# This avoids ~500 MB download on first request at runtime.
# Downloads English detection + recognition models to /home/appuser/.EasyOCR/
RUN python -c "\
import easyocr; \
reader = easyocr.Reader(['en'], gpu=False, download_enabled=True); \
print('EasyOCR weights cached successfully')"

# ── Environment variables ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best.pt \
    MODEL_VERSION=sprint2_yolo11l_expB_v1 \
    CONFIDENCE_THRESHOLD=0.42 \
    LOG_DIR=/data \
    PORT=7860

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Expose port and run ──
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
