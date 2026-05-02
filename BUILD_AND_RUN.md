# WristAssist AI — Sprint 3 Build & Run Guide

## Quick Start (Local Development)

```bash
# 1. Ensure best.pt exists at Sprint2/models/best.pt
#    If not, pull from Git LFS: git lfs pull

# 2. Build the Docker image
docker build -t wristassist:dev .

# 3. Check image size (must be < 3 GB for HF Spaces)
docker images wristassist:dev

# 4. Run locally
docker run -p 7860:7860 -v $(pwd)/data:/data wristassist:dev

# 5. Test health check
curl http://localhost:7860/health

# 6. Open in browser
open http://localhost:7860
```

## Expected /health Response

```json
{
  "status": "ok",
  "model_version": "sprint2_yolo11l_expB_v1",
  "timestamp": "2026-04-29T10:00:00+00:00"
}
```

## File Layout (What Goes Into the Container)

```
/ (build context)
├── Dockerfile              ← This file controls the build
├── .dockerignore           ← Excludes docs, notebooks, data
├── requirements.txt        ← Pinned Python dependencies
├── Sprint2/
│   ├── models/best.pt      ← Production model (~48 MB, via Git LFS)
│   └── utils/
│       ├── ocr.py          ← EasyOCR laterality extraction
│       └── report.py       ← Clinical JSON report generation
└── Sprint3/
    └── app/
        ├── app.py           ← FastAPI backend (7 endpoints)
        ├── index.html       ← Web UI (single file)
        ├── monitoring.py    ← Praveer builds this (SCRUM-53)
        └── drift.py         ← Praveer builds this (SCRUM-54)
```

## Inside the Container

```
/app/
├── app.py               ← Entrypoint
├── index.html            ← Served at GET /
├── models/best.pt        ← YOLO11l weights
├── utils/
│   ├── ocr.py
│   └── report.py
├── monitoring.py         ← When available
├── drift.py              ← When available
└── baseline_distribution.json  ← When available

/data/                    ← HF persistent storage (mounted at runtime)
├── predictions.jsonl     ← Written by app.py on every /predict
└── feedback.jsonl        ← Written by app.py on POST /feedback
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Path to YOLO11l weights inside container |
| `MODEL_VERSION` | `sprint2_yolo11l_expB_v1` | Version string logged with every prediction |
| `CONFIDENCE_THRESHOLD` | `0.42` | Default confidence threshold |
| `LOG_DIR` | `/data` | Directory for prediction and feedback logs |
| `PORT` | `7860` | Server port (must be 7860 for HF Spaces) |

## Troubleshooting

### Image size > 3 GB
- Ensure `.dockerignore` excludes `data/`, `Sprint1/`, notebooks, docs
- Verify PyTorch CPU-only install: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Check EasyOCR only caches English weights

### /health returns "model_not_loaded"
- Verify `Sprint2/models/best.pt` exists and is not a Git LFS pointer file
- Run `git lfs pull` before building

### EasyOCR downloads weights at runtime
- The Dockerfile pre-caches weights during build
- If you see downloads at runtime, rebuild with the current Dockerfile

### Port 7860 not accessible
- HF Spaces requires exactly port 7860
- Verify: `docker run -p 7860:7860` not some other port

## SCRUM-46 Acceptance Criteria Checklist

- [ ] `Dockerfile` at repo root using `python:3.11-slim` base
- [ ] Multi-stage build with EasyOCR weights pre-downloaded during build
- [ ] `requirements.txt` pinning all dependencies
- [ ] Image size under 3 GB
- [ ] `docker build` completes in under 10 minutes on clean machine
- [ ] `docker run -p 7860:7860` starts, `/health` returns 200
- [ ] Container includes best.pt at build time
- [ ] Screenshot of build + run + health check + `docker images` size
