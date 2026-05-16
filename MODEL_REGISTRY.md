# Model Registry and Versioning — WristAssist AI

This document describes how WristAssist AI manages model versions, promotion workflows, and deployment traceability using ClearML as the model registry and Git LFS as the delivery mechanism.

## Registry Architecture

WristAssist uses a two-layer versioning system:

- **ClearML Model Registry** is the source of truth for model metadata, evaluation metrics, approval gate status, and promotion tags. Every model trained through the pipeline is registered here with full lineage.
- **Git LFS** is the delivery mechanism. The production-tagged `best.pt` is stored in `models/best.pt` via Git LFS, and the CD workflow deploys it to Hugging Face Spaces on every push to `main`.

This separation means ClearML answers "which model is approved and why" while Git answers "which model is deployed right now." Both point to the same artifact via SHA checksum.

## Current Production Model

| Field | Value |
|-------|-------|
| Version string | `sprint2_yolo11l_expB_v1` |
| Architecture | YOLO11l (25M parameters) |
| Training platform | Kaggle 2×T4 GPUs (DDP) |
| Image size | 640 × 640 |
| Augmentation | Heavy (mixup=0.15, copy_paste=0.3, cls=1.5) |
| File size | ~48 MB |
| Confidence threshold | 0.42 (calibrated from Sprint 2 F1 curve) |
| Artifact path | `models/best.pt` (Git LFS) |
| ClearML tags | `production`, `sprint2_expB`, `yolo11l`, `gate-passed` |

### Test Set Performance (3,115 images, locked split)

| Class | Images | Instances | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|--------|-------------|
| all | 3,115 | 7,107 | 0.811 | 0.837 | 0.842 | 0.551 |
| fracture | 2,082 | 2,752 | 0.934 | 0.866 | 0.941 | 0.551 |
| metal_implant | 126 | 142 | 0.920 | 0.892 | 0.906 | 0.776 |
| periosteal_reaction | 348 | 518 | 0.714 | 0.641 | 0.695 | 0.337 |
| pronator_sign | 97 | 97 | 0.517 | 0.805 | 0.678 | 0.368 |
| text | 3,099 | 3,598 | 0.971 | 0.982 | 0.991 | 0.722 |

## Version Naming Convention

Model versions follow the format:

```
sprint{N}_yolo11{size}_exp{letter}_v{version}
```

Examples:
- `sprint2_yolo11l_expB_v1` — Sprint 2, YOLO11l, Experiment B, first version
- `sprint3_yolo11l_hpo_v1` — Sprint 3, YOLO11l, HPO-optimised, first version

The version string is embedded in `experiment_config.yaml` under `production.model_version` and returned by the `/health` endpoint at runtime.

## Model Lifecycle

Every model passes through four stages before reaching production:

```
CANDIDATE → EVALUATED → APPROVED → PRODUCTION
```

### Stage 1 — Candidate

A new model is trained via the pipeline (either manually on Kaggle or triggered by the HPO stage). The `final_model_stage.py` script:

1. Reads the best hyperparameters from the HPO stage artifact
2. Trains YOLO11l on the full training split
3. Registers the resulting `best.pt` as a ClearML `OutputModel` with metadata

At this point the model is tagged `candidate` and `hpo_optimised` in ClearML.

### Stage 2 — Evaluated

The same stage evaluates the model on the locked test split (3,115 images, patient-level split, never seen during training). Metrics logged to ClearML:

- mAP@0.5 (overall)
- mAP@0.5:0.95 (overall)
- Per-class AP@0.5 (fracture, metal_implant, periosteal_reaction, pronator_sign, text)
- Fracture recall

### Stage 3 — Approved (Gate Passed)

The approval gate checks two conditions defined in `experiment_config.yaml`:

```yaml
approval_gate:
  mAP50_min: 0.685          # Sprint 1 baseline
  fracture_AP50_min: 0.90   # Clinical safety threshold
  fracture_recall_min: 0.85  # Minimum fracture sensitivity
```

Both conditions must pass. On pass, the model is tagged `gate-passed` and `production-candidate`. On fail, it is tagged `gate-failed` and the pipeline exits non-zero, blocking deployment.

### Stage 4 — Production

A model is promoted to production by:

1. Running `scripts/register_model.py --promote` (updates ClearML tags)
2. Copying `best.pt` to `models/best.pt` in the repo
3. Updating `production.model_version` in `experiment_config.yaml`
4. Pushing to `main` — the CD workflow deploys it automatically

Only one model carries the `production` tag at any time. The previous production model is re-tagged to `archived`.

## ClearML Integration

### Project Structure

All experiments, pipelines, and models live under the ClearML project `IronGear/WristAssist`.

| ClearML Entity | Purpose |
|----------------|---------|
| Experiments | Individual training runs (Exp A, B, C, HPO trials) |
| Pipelines | 6-stage DAG: data → process → train → hpo → final_model → evaluate |
| Models | Registered OutputModel artifacts with tags and metrics |

### How Models Are Registered

Registration happens in `pipeline/final_model_stage.py` (lines 187–202):

```python
output_model = OutputModel(
    task=task,
    name="final_model",
    framework="PyTorch",
)
output_model.update_weights(
    weights_filename=str(best_pt),
    auto_delete_file=False,
)

if gate_passed:
    output_model.tags = ["production", "hpo_optimised", "gate-passed"]
else:
    output_model.tags = ["candidate", "hpo_optimised", "gate-failed"]
```

The model artifact is stored in ClearML's file server with the task ID as lineage. The same `best.pt` file is then pushed to Git LFS for deployment.

### Querying the Registry

To find the current production model:

```python
from clearml import Model

models = Model.query_models(
    project_name="IronGear/WristAssist",
    tags=["production"],
    only_published=False,
)
production_model = models[0] if models else None
```

To find all gate-passed candidates:

```python
candidates = Model.query_models(
    project_name="IronGear/WristAssist",
    tags=["gate-passed"],
)
```

## Promotion Workflow

### Automated (via pipeline)

When the 6-stage pipeline completes with a gate-passed model, `final_model_stage.py` automatically tags it as `production`. The operator then:

1. Pulls the `best.pt` artifact from ClearML or the pipeline output directory
2. Copies it to `models/best.pt` in the repo
3. Updates `experiment_config.yaml` with the new version string
4. Pushes to `main`

### Manual (via register_model.py)

For ad-hoc model promotion outside the pipeline (e.g., promoting a manually trained model):

```bash
# Register a new model in ClearML with metadata
python scripts/register_model.py register \
    --weights models/best.pt \
    --version sprint3_yolo11l_hpo_v1 \
    --metrics '{"mAP50": 0.858, "fracture_AP50": 0.941}'

# Promote a model to production (demotes the current production model)
python scripts/register_model.py promote \
    --model-id <CLEARML_MODEL_ID>

# List all registered models
python scripts/register_model.py list
```

### Drift-Triggered Retraining

When the drift detection module (`app/drift.py`) detects PSI > 0.25 on any clinical class:

1. `drift.py` fires a `repository_dispatch` event to GitHub
2. `retrain.yml` creates a GitHub issue documenting the drift
3. `retrain.yml` submits a ClearML retraining task to the `default` queue
4. An operator runs `clearml-agent` on a GPU machine to execute training
5. The new model goes through the same lifecycle: candidate → evaluated → approved → production

## Deployment Flow

```
ClearML Registry                    Git LFS                    HF Spaces
┌─────────────────┐          ┌──────────────────┐       ┌──────────────────┐
│ model tagged     │   copy   │ models/best.pt   │  CD   │ Docker container  │
│ "production"     │ ───────→ │ pushed to main   │ ────→ │ loads best.pt on  │
│ + gate-passed    │          │ via Git LFS      │       │ startup           │
└─────────────────┘          └──────────────────┘       └──────────────────┘
```

The CD workflow (`cd.yml`) does not read from ClearML directly — it deploys whatever `models/best.pt` is in the `main` branch. ClearML's role is governance (approving which model gets pushed), not transport.

## Model Card

A structured model card is maintained at `models/model_card.json`. It contains:

- Model architecture and version
- Training configuration and hyperparameters
- Dataset details (GRAZPEDWRI-DX, patient-level split)
- Evaluation metrics on the locked test split
- Approval gate results
- Intended use and limitations
- Lineage (source experiment, pipeline run, ClearML task IDs)

The model card is updated each time a new model is promoted to production.

## Rollback Procedure

If a deployed model causes issues in production:

1. Identify the previous production model's commit SHA via `git log models/best.pt`
2. Revert: `git revert <bad-commit-sha>` on `main`
3. Push — CD workflow redeploys the previous model
4. In ClearML, re-tag the reverted model as `production` and the bad model as `rolled-back`
5. Investigate and document the issue in a GitHub issue

## Files Reference

| File | Purpose |
|------|---------|
| `models/best.pt` | Production model weights (Git LFS) |
| `models/model_card.json` | Structured model card with lineage |
| `scripts/register_model.py` | CLI utility for model registration and promotion |
| `pipeline/final_model_stage.py` | Pipeline stage that trains, evaluates, and registers models |
| `experiment_config.yaml` | Central config including production model version and approval gate |
| `app/baseline_distribution.json` | Sprint 2 class distribution baseline for drift detection |
