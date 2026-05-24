# WristAssist AI

**Paediatric Wrist Trauma X-Ray Detection System**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://irongear375-wristassist.hf.space)
[![CI](https://img.shields.io/github/actions/workflow/status/irongear375/IronGear/ci.yml?style=for-the-badge&label=CI&logo=github)](https://github.com/irongear375/IronGear/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/irongear375/IronGear/cd.yml?style=for-the-badge&label=CD&logo=github)](https://github.com/irongear375/IronGear/actions/workflows/cd.yml)
[![Tests](https://img.shields.io/badge/Tests-42%20passing-brightgreen?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)

WristAssist AI is a clinical decision-support tool that detects and localises trauma findings in paediatric wrist X-rays. Upload an X-ray, get annotated results with bounding boxes, confidence scores, wrist laterality, and a structured clinical report — all within seconds.

> **Decision support only.** Not a diagnostic tool. Clinicians retain final responsibility. Education/research use only.

---

## Quick Links

| Resource | URL |
|----------|-----|
| Live Application | https://irongear375-wristassist.hf.space |
| API Documentation | https://irongear375-wristassist.hf.space/docs |
| Health Check | https://irongear375-wristassist.hf.space/health |
| Monitoring Metrics | https://irongear375-wristassist.hf.space/metrics |
| Drift Detection | https://irongear375-wristassist.hf.space/drift |
| ClearML Project | app.clear.ml → IronGear/WristAssist |

---

## What It Does

A clinician uploads a paediatric wrist X-ray. The system:

1. **Detects** four clinical findings with bounding boxes and confidence scores
2. **Extracts** wrist laterality (Left / Right) from the text marker position
3. **Returns** an annotated image overlay and a structured JSON clinical report
4. **Reports** explicit no-findings for undetected classes (not implied by absence)
5. **Flags** cases that need human review based on confidence thresholds
6. **Logs** every prediction for monitoring and drift detection

### Detection Classes

| Class | AP@0.5 | Role |
|-------|--------|------|
| Fracture | 94.1% | Primary clinical target |
| Metal implant | 90.6% | Surgical hardware detection |
| Periosteal reaction | 69.5% | Bone surface change indicator |
| Pronator sign | 67.8% | Occult fracture indicator |
| Text | 99.1% | Laterality extraction (L/R) |

**Overall mAP@0.5: 0.858** | **Fracture recall: 91.6%** | **CPU latency: 3-5 seconds**

---

## Architecture

Monolithic FastAPI application deployed as a Docker container on Hugging Face Spaces.

```
Browser (clinician / assessor)
    │
    ▼
Web UI (index.html) ── Upload page → Results page → Feedback panel
    │
    ▼
FastAPI Backend (app.py) ── 7 endpoints │ Validation │ Preprocessing │ Logging
    │
    ▼
ML Inference ── YOLO11l detector (best.pt, 48 MB) + Position-based laterality
    │
    ▼
HF Persistent Storage ── predictions.jsonl │ feedback.jsonl │ baseline_distribution.json
```

### Key Design Decisions

- **In-memory only**: Uploaded images are decoded, processed, and discarded in RAM. Never written to disk.
- **Model loaded on startup**: Eliminates 30s per-request loading overhead. Kept resident for container lifetime.
- **CPU inference**: Meets ≤6s latency target on HF Spaces free tier (8 vCPU, 16 GB RAM).
- **No EasyOCR at runtime**: Replaced with position-based laterality extraction (<1ms vs 60-70s per OCR call).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve web UI |
| `GET` | `/health` | Health check — returns model version and status |
| `POST` | `/predict` | Core inference — upload X-ray, get clinical report + annotated image |
| `GET` | `/log` | Recent prediction log entries (default 10, max 100) |
| `POST` | `/feedback` | Record clinician feedback (correct/incorrect + comment) |
| `GET` | `/metrics` | Monitoring dashboard — latency percentiles, per-class rates |
| `GET` | `/drift` | Drift detection — PSI per class vs Sprint 2 baseline |

### POST /predict Response

```json
{
  "request_id": "a1b2c3d4-...",
  "model_version": "sprint2_yolo11l_expB_v1",
  "timestamp": "2026-05-24T10:30:15Z",
  "latency_ms": 3240,
  "laterality": "Left",
  "findings_detected": [
    {"class_name": "fracture", "confidence": 0.91, "box": [120, 80, 200, 130]}
  ],
  "no_findings": ["metal_implant", "periosteal_reaction", "pronator_sign"],
  "summary": {
    "fracture_suspected": true,
    "implant_present": false,
    "periosteal_reaction_present": false,
    "pronator_sign_present": false,
    "needs_human_review": true
  },
  "confidence_threshold_used": 0.42,
  "annotated_image": "data:image/png;base64,..."
}
```

---

## Repository Structure

```
IronGear/
├── app/                          # Deployed application
│   ├── app.py                    #   FastAPI backend (7 endpoints)
│   ├── index.html                #   Web UI (single file, no framework)
│   ├── monitoring.py             #   Prediction logging + GET /metrics
│   ├── drift.py                  #   PSI drift detection + GET /drift
│   └── baseline_distribution.json#   Sprint 2 class distribution baseline
├── models/
│   ├── best.pt                   #   YOLO11l production model (Git LFS, 48 MB)
│   └── model_card.json           #   Model metadata and training lineage
├── pipeline/
│   ├── pipeline_controller.py    #   6-stage ClearML pipeline orchestrator
│   ├── data_processing.py        #   Stages 1-2: data extraction + processing
│   ├── hpo_stage.py              #   Stage 4: hyperparameter optimisation
│   └── final_model_stage.py      #   Stage 5: final model training
├── scripts/
│   ├── register_model.py         #   CLI: register model in ClearML
│   └── promote_model.py          #   CLI: promote model to production
├── tests/
│   └── test_app.py               #   42 pytest tests (6 categories)
├── .github/workflows/
│   ├── ci.yml                    #   CI: lint → test → validate → Docker build
│   ├── cd.yml                    #   CD: push to HF Spaces → smoke test
│   └── retrain.yml               #   Retrain: drift-triggered via repository_dispatch
├── Dockerfile                    #   Docker build recipe for HF Spaces
├── requirements.txt              #   Python dependencies
├── experiment_config.yaml        #   Central config (hyperparams, gate thresholds, HPO)
├── MODEL_REGISTRY.md             #   Model versioning strategy and lifecycle
└── .hfignore                     #   Files excluded from HF Spaces deployment
```

---

## Model

**Production model:** `sprint2_yolo11l_expB_v1`

| Property | Value |
|----------|-------|
| Architecture | YOLO11l (25M parameters) |
| Image size | 640 × 640 |
| Training platform | Kaggle 2×T4 (30 GB VRAM) |
| Epochs | 35 (best at epoch 33) |
| Augmentation | mixup=0.15, copy_paste=0.3 |
| Confidence threshold | 0.42 (calibrated from F1 curve) |
| File size | ~48 MB |

### Experiment Comparison

| Config | Sprint 1 (POC) | Exp A | **Exp B (Selected)** | Exp C |
|--------|----------------|-------|----------------------|-------|
| Model | YOLO11x | YOLO11l | **YOLO11l** | YOLO11l |
| Image size | 640 | 1280 | **640** | 1280 |
| Augmentation | Standard | Heavy | **Heavy** | Light |
| mAP@0.5 | 0.685 | 0.846 | **0.858** | 0.845 |
| Fracture recall | — | 91.1% | **91.6%** | 89.7% |

**Key finding:** Heavy augmentation (mixup + copy_paste) was more effective than doubling image resolution. GRAZPEDWRI-DX images are not intrinsically high-resolution, so upscaling added compute without improving accuracy.

### Approval Gate

Both conditions must pass for any model to be deployed:

- [x] Overall mAP@0.5 > 0.685 → achieved **0.858**
- [x] Fracture AP@0.5 ≥ 0.90 → achieved **0.941**

---

## Dataset

**GRAZPEDWRI-DX** (Nagy et al., 2022) — 20,327 paediatric wrist X-ray images across 6,091 patients.

| Split | Images | Ratio |
|-------|--------|-------|
| Train (base) | 14,229 | 70% |
| Train (oversampled) | 35,734 | — |
| Validation | 2,969 | 15% |
| Test (locked) | 3,115 | 15% |

- **Patient-level split** prevents data leakage (one patient's images never appear in multiple splits)
- **Class oversampling** on training set only: metal_implant ×5, periosteal_reaction ×2, pronator_sign ×5
- **9 → 5 class remapping**: dropped boneanomaly, bonelesion, foreignbody (too rare), soft_tissue (AP=0.307)

---

## CI/CD Pipeline

```
Code push → ci.yml (lint → 42 tests → pipeline validate → Docker build)
                │
                ▼ (pass)
            cd.yml (push to HF Spaces → auto-rebuild → smoke test /health)
                │
                ▼ (deployed)
         Live at irongear375-wristassist.hf.space
                │
                ▼ (monitoring)
         /metrics + /drift (PSI per class vs baseline)
                │
                ▼ (PSI > 0.25)
         retrain.yml (GitHub issue → ClearML task → operator runs on GPU)
```

---

## Monitoring & Drift Detection

Every prediction is logged to `predictions.jsonl` with: timestamp, request_id, latency, per-class confidences, laterality, and needs_human_review flag.

**GET /metrics** returns: total predictions, p50/p95 latency, per-class prediction rates (24h/7d), laterality distribution, human review rate.

**GET /drift** computes PSI (Population Stability Index) per class against the Sprint 2 training baseline:

| PSI Range | Status | Action |
|-----------|--------|--------|
| < 0.10 | OK | No action |
| 0.10 - 0.25 | WARNING | Logged for monitoring |
| > 0.25 | ALERT | Auto-creates GitHub issue + triggers retrain workflow |

---

## Testing

42 tests across 6 categories, run in CI on every push:

| Category | Tests | Coverage |
|----------|-------|----------|
| API Endpoints | 12 | Health, predict, log, feedback, metrics, drift |
| Input Validation | 8 | MIME type, extension, file size, 16-bit PNG |
| Clinical Report | 6 | findings, no_findings, laterality, summary flags |
| Monitoring | 5 | Logging, latency, metric aggregation |
| Drift Detection | 5 | PSI computation, thresholds, alerts |
| Pipeline Validation | 6 | Config schema, class mapping, approval gate |

```bash
pytest tests/test_app.py -v
```

---

## ClearML Pipeline

6-stage automated ML pipeline:

```
data → process → train → hpo → final_model → evaluate
```

| Stage | What It Does |
|-------|-------------|
| data | Scan dataset, build master index |
| process | Remap classes (9→5), patient-level split, oversample |
| train | Train YOLO11l with baseline hyperparameters |
| hpo | Grid search over lr, mixup, copy_paste, cls weight |
| final_model | Train with best HPO configuration |
| evaluate | Evaluate on locked test set + enforce approval gate |

All settings are read from `experiment_config.yaml`. Run modes: `validate` (CI), `ingest` (register existing model), `full` (complete training).

---

## Local Development

```bash
# Clone with LFS
git clone https://github.com/irongear375/IronGear.git
cd IronGear
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run locally
cd app
uvicorn app:app --host 0.0.0.0 --port 7860

# Open http://localhost:7860
```

### Docker

```bash
docker build -t wristassist .
docker run -p 7860:7860 wristassist
```

### Run Tests

```bash
pytest tests/test_app.py -v
```

---

## Sprint History

| Sprint | Dates | MLOps Level | Key Achievement |
|--------|-------|-------------|-----------------|
| Sprint 1 | 29 Mar 2026 | Level 0 (Manual) | YOLO11x POC, mAP@0.5 = 0.685 |
| Sprint 2 | 26 Apr 2026 | Level 1 (Pipeline) | YOLO11l Exp B, mAP@0.5 = 0.858 (+25%) |
| Sprint 3 | 24 May 2026 | Level 2 (CI/CD) | Public deployment, CI/CD, monitoring, 42 tests |

---

## Team

**Iron Gear** — UTS 42174 AI Studio, Autumn 2026

| Member | Student ID | Sprint 1 | Sprint 2 | Sprint 3 |
|--------|-----------|----------|----------|----------|
| Muhammad Hashim | 25744970 | Sprint Manager | Data Scientist | Data Engineer |
| Vaibhav Bairathi | 25534645 | Data Scientist | Data Engineer | Sprint Manager |
| Praveer Jain | 25947209 | Data Engineer | Sprint Manager | Data Scientist |

---

## References

- Nagy, E. et al. (2022). A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX). *Scientific Data*, 9, 222.
- Ju, R.-Y. & Cai, W. (2023). Fracture detection in pediatric wrist trauma X-ray images using YOLOv8. *Scientific Reports*, 13(1), 20077.
- Google Cloud (2024). MLOps: Continuous delivery and automation pipelines in machine learning.

---

## License

This project is developed for academic assessment purposes as part of UTS 42174 AI Studio (Autumn 2026). The GRAZPEDWRI-DX dataset is used under its original research license.
