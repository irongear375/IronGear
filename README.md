# WristAssist AI — Sprint 1 & Sprint 2 Development Code

> **This is the `legacy-sprints` branch.** It contains the Sprint 1 (POC) and Sprint 2 (pipeline) development notebooks and artifacts. For the production application and Sprint 3 code, switch to the [`main`](https://github.com/irongear375/IronGear/tree/main) branch.

[![Live Demo](https://img.shields.io/badge/Live%20App-main%20branch-blue?style=for-the-badge)](https://irongear375-wristassist.hf.space)

---

## What's In This Branch

This branch preserves the Sprint 1 and Sprint 2 development history: data exploration, data processing, model training notebooks, the ClearML pipeline, and utility modules. The `main` branch was restructured for Sprint 3 deployment, so this branch keeps the original notebook-based workflow accessible.

```
legacy-sprints/
├── Sprint1-POC/                    # Sprint 1 — Proof of Concept (MLOps Level 0)
│   ├── 01_data_processing.ipynb    #   Patient-level split, 9→5 class remap, oversampling
│   ├── 02_train_evaluate.ipynb     #   YOLO11x training + locked test set evaluation
│   └── 03_demo.ipynb              #   Inference demo with ipywidgets upload panel
│
├── Sprint2/                        # Sprint 2 — Automated Pipeline (MLOps Level 1)
│   ├── notebooks/
│   │   ├── Sprint2_data_processing.ipynb   # Updated data processing with 5-class schema
│   │   ├── 02_train_evaluate.ipynb         # Thin notebook wrapper for pipeline.py
│   │   └── 03_demo.ipynb                  # Demo notebook with EasyOCR laterality + JSON report
│   ├── clearml_pipeline/
│   │   ├── pipeline_from_tasks.py          # 4-stage ClearML PipelineController DAG
│   │   ├── s1_data.py                      # Stage 1: environment check + dataset scan
│   │   ├── s2_process.py                   # Stage 2: validate schema, symlink images, dataset.yaml
│   │   ├── s3_train.py                     # Stage 3: YOLO11l training (ingest/smoke/full modes)
│   │   └── s4_evaluate.py                  # Stage 4: evaluation + approval gate
│   ├── utils/
│   │   ├── pipeline.py                     # Core pipeline logic (~480 lines)
│   │   └── report.py                       # Structured clinical JSON report generation
│   └── models/
│       └── best.pt                         # Experiment B production model (Git LFS, 48 MB)
│
├── data/                           # Dataset index files
│   ├── master_index.csv            #   20,327 images indexed with patient IDs
│   └── dataset_paths.json          #   Patient-level 70/15/15 split assignments
│
├── data_exploration.ipynb          # EDA: class distributions, image counts, annotation stats
└── data_processing.ipynb           # Original Sprint 1 data processing (superseded by Sprint1-POC/)
```

---

## Sprint 1 — Proof of Concept (MLOps Level 0)

**Goal:** Train a baseline model and prove the concept works.

| Detail | Value |
|--------|-------|
| Model | YOLO11x (57M params) |
| Platform | AWS SageMaker ml.g4dn.xlarge (1×T4) |
| Image size | 640 × 640 |
| Batch size | 2 (VRAM constrained) |
| Epochs | 50 |
| mAP@0.5 | 0.685 (became the baseline for all future models) |
| Fracture AP@0.5 | 0.939 |

**Key decisions made in Sprint 1:**
- Patient-level 70/15/15 split to prevent data leakage across 6,091 patients
- 9 → 5 class remapping (dropped boneanomaly, bonelesion, foreignbody, soft_tissue)
- Class oversampling: metal_implant ×5, periosteal_reaction ×2, pronator_sign ×5
- Approval gate defined: mAP@0.5 > 0.685 AND fracture AP@0.5 ≥ 0.90

---

## Sprint 2 — Automated Pipeline (MLOps Level 1)

**Goal:** Run collaborative experiments, select a production model, build an automated pipeline.

### Three Experiments

| Config | Exp A (Hashim) | **Exp B (Vaibhav)** | Exp C (Praveer) |
|--------|----------------|---------------------|-----------------|
| Image size | 1280 | **640** | 1280 |
| Batch size | 8 | **16** | 8 |
| Augmentation | Heavy | **Heavy** | Light |
| Epochs | 88 | **35** | 56 |
| mAP@0.5 | 0.846 | **0.858** | 0.845 |
| Fracture recall | 91.1% | **91.6%** | 89.7% |
| Platform | Kaggle 2×T4 | Kaggle 2×T4 | Kaggle 2×T4 |

**Experiment B selected** — highest mAP, highest fracture recall, fastest training.

**Key finding:** Heavy augmentation (mixup=0.15, copy_paste=0.3) mattered more than doubling image resolution from 640 to 1280.

### ClearML Pipeline (4 stages)

```
s1_data → s2_process → s3_train → s4_evaluate
```

### New in Sprint 2
- Switched from YOLO11x to YOLO11l (fits on T4 with room to spare)
- Migrated training from SageMaker to Kaggle 2×T4 (30 GB free VRAM)
- Added EasyOCR laterality extraction (YOLO detects text bbox → crop → OCR reads L/R)
- Added structured clinical JSON report with findings, no-findings, laterality, summary flags
- Confidence threshold calibrated at 0.42 from F1 curve

---

## Dataset

**GRAZPEDWRI-DX** (Nagy et al., 2022) — available on Kaggle.

- 20,327 PNG images across 6,091 patients
- Patient-level 70/15/15 split (train: 14,229 base / 35,734 oversampled, val: 2,969, test: 3,115)
- 5-class schema: fracture (0), metal_implant (1), periosteal_reaction (2), pronator_sign (3), text (4)

---

## How to Run These Notebooks

### Sprint 1 (SageMaker)

```bash
# On SageMaker ml.g4dn.xlarge or any machine with a T4 GPU
cd Sprint1-POC
# 1. Run data processing first
jupyter lab 01_data_processing.ipynb
# 2. Train the model
jupyter lab 02_train_evaluate.ipynb
# 3. Run the demo
jupyter lab 03_demo.ipynb
```

### Sprint 2 (Kaggle)

```bash
# On Kaggle with 2×T4 GPU accelerator enabled
cd Sprint2/notebooks
# Upload the notebook to Kaggle, attach GRAZPEDWRI-DX dataset
# Set EXPERIMENT = "B" in the notebook for the production configuration
jupyter lab 02_train_evaluate.ipynb
```

### ClearML Pipeline

```bash
cd Sprint2/clearml_pipeline
# Requires ClearML credentials configured
python pipeline_from_tasks.py
```

---

## Production Model

The production model (`best.pt`) in this branch is **sprint2_yolo11l_expB_v1** — the same model deployed on the live application. It is tracked with Git LFS.

```bash
git lfs pull  # Download the actual model weights
```

---

## Where to Find Sprint 3

Sprint 3 (public deployment, CI/CD, monitoring, 42 tests) is on the **main branch**:

→ [`main` branch](https://github.com/irongear375/IronGear/tree/main)
→ [Live application](https://irongear375-wristassist.hf.space)

---

## Team

**Iron Gear** — UTS 42174 AI Studio, Autumn 2026

| Member | Sprint 1 Role | Sprint 2 Role |
|--------|--------------|--------------|
| Muhammad Hashim (25744970) | Sprint Manager | Data Scientist (Exp A) |
| Vaibhav Bairathi (25534645) | Data Scientist | Data Engineer (Exp B) |
| Praveer Jain (25947209) | Data Engineer | Sprint Manager (Exp C) |

---

## References

- Nagy, E. et al. (2022). A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX). *Scientific Data*, 9, 222.
- Ju, R.-Y. & Cai, W. (2023). Fracture detection in pediatric wrist trauma X-ray images using YOLOv8. *Scientific Reports*, 13(1), 20077.
