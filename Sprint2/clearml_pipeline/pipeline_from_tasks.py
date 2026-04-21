"""
Sprint 2 ClearML Pipeline — Controller
=======================================
Wires the 4 registered tasks into a DAG:

    stage_data  →  stage_process  →  stage_train  →  stage_evaluate

Maps the monolithic Sprint2_pipeline.py's 8 stages onto 4 DAG nodes:

    s1_data.py       ← (monolith stage 1: env check) + (stage 2: data extraction)
    s2_process.py    ← (stage 3: validation)          + (stage 4: preparation)
    s3_train.py      ← (stage 5: training)            [modes: ingest/smoke/full]
    s4_evaluate.py   ← (stage 6: eval) + (stage 7: gate) + (stage 8: export)

DEFAULT CONFIGURATION — DOES NOT AFFECT YOUR EXISTING KAGGLE WORK
  - stage_train runs in mode="ingest" — it does NOT re-train
  - It consumes your existing Kaggle-produced best.pt as input
  - Your 19 existing ClearML tasks remain untouched
  - Your ongoing Exp A/B/C runs keep their IDs, tags, and metrics

Prerequisites:
  1. Each sN_*.py must have been run once to register its Task in ClearML.
     Look for these 4 new task names under IronGear/WristAssist:
         Pipeline step 1 data extraction
         Pipeline step 2 data processing
         Pipeline step 3 train model
         Pipeline step 4 evaluate model

  2. Path to your Kaggle-produced best.pt must be set below in the
     "pretrained_weights_path" parameter.

Team: Iron Gear | Subject: 42174 AI Studio (Autumn 2026)
"""

from clearml import PipelineController

PROJECT = "IronGear/WristAssist"

# ═════════════════════════════════════════════════════════════════════════════
#  EDIT THIS  — path to your Kaggle-produced best.pt (Experiment A at mAP=0.841)
# ═════════════════════════════════════════════════════════════════════════════
KAGGLE_BEST_PT = "/home/sagemaker-user/user-default-efs/IronGear/Sprint2/models/best.pt"

# ═════════════════════════════════════════════════════════════════════════════
#  EDIT THIS  — match your SageMaker EFS layout for the raw dataset
# ═════════════════════════════════════════════════════════════════════════════
IMAGES_ROOT = "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset/images"
LABELS_ROOT = "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset/labels"
YOLO_DIR    = "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset"


# -----------------------------------------------------------------------------
# Status callbacks — print to console during pipeline execution
# -----------------------------------------------------------------------------
def pre_execute(pipeline, node, parameters):
    print(f"\n[pipeline] ⏳ Starting stage: {node.name}")
    if parameters:
        for k, v in parameters.items():
            print(f"[pipeline]     {k} = {v}")
    return True


def post_execute(pipeline, node):
    print(f"[pipeline] ✅ Finished stage: {node.name} (task_id={node.executed})")


# -----------------------------------------------------------------------------
# Build the DAG
# -----------------------------------------------------------------------------
pipe = PipelineController(
    name="WristAssist Sprint 2 Pipeline",
    project=PROJECT,
    version="1.0",
    add_pipeline_tags=True,
)
pipe.set_default_execution_queue("default")  # harmless in local mode

# ─── Stage 1: data ──────────────────────────────────────────────────────────
pipe.add_step(
    name="stage_data",
    base_task_project=PROJECT,
    base_task_name="Pipeline step 1 data extraction",
    parameter_override={
        "General/images_root": IMAGES_ROOT,
        "General/labels_root": LABELS_ROOT,
    },
    pre_execute_callback=pre_execute,
    post_execute_callback=post_execute,
)

# ─── Stage 2: process ───────────────────────────────────────────────────────
pipe.add_step(
    name="stage_process",
    parents=["stage_data"],
    base_task_project=PROJECT,
    base_task_name="Pipeline step 2 data processing",
    parameter_override={
        "General/dataset_task_id": "${stage_data.id}",
        "General/yolo_dir": YOLO_DIR,
        # skip_prep=False — stage 2 will:
        #   (1) validate labels, (2) symlink images (idempotent),
        #   (3) skip label copy safely if labels_src == yolo_dir/labels,
        #   (4) symlink images for existing _os* labels, (5) write dataset.yaml
        # It does NOT create new oversampled labels — it uses what's already there.
        "General/skip_prep": False,
    },
    pre_execute_callback=pre_execute,
    post_execute_callback=post_execute,
)

# ─── Stage 3: train ─────────────────────────────────────────────────────────
# DEFAULT mode=ingest — uses your Kaggle best.pt, does NOT re-train.
# To actually train (smoke or full), change "mode" below.
pipe.add_step(
    name="stage_train",
    parents=["stage_process"],
    base_task_project=PROJECT,
    base_task_name="Pipeline step 3 train model",
    parameter_override={
        "General/process_task_id": "${stage_process.id}",
        "General/mode": "ingest",  # "ingest" | "smoke" | "full"
        "General/pretrained_weights_path": KAGGLE_BEST_PT,
        # The below are only used if mode != "ingest"
        "General/experiment": "A",
        "General/smoke_epochs": 2,
        "General/smoke_imgsz": 640,
        "General/smoke_batch": 4,
    },
    pre_execute_callback=pre_execute,
    post_execute_callback=post_execute,
)

# ─── Stage 4: evaluate ──────────────────────────────────────────────────────
pipe.add_step(
    name="stage_evaluate",
    parents=["stage_train"],
    base_task_project=PROJECT,
    base_task_name="Pipeline step 4 evaluate model",
    parameter_override={
        "General/train_task_id": "${stage_train.id}",
        "General/imgsz": 1280,            # match Exp A training resolution
        "General/split": "test",
        "General/min_map50": 0.685,
        "General/min_fracture_ap": 0.90,
    },
    pre_execute_callback=pre_execute,
    post_execute_callback=post_execute,
)


# -----------------------------------------------------------------------------
# Run locally — no clearml-agent daemon required
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("  WristAssist Sprint 2 — ClearML Pipeline (4-stage DAG)")
    print("  Project: IronGear/WristAssist")
    print(f"  Ingesting Kaggle best.pt from: {KAGGLE_BEST_PT}")
    print("=" * 72)

    pipe.start_locally(run_pipeline_steps_locally=True)

    print("\n[pipeline] Pipeline finished. View DAG at:")
    print("  https://app.clear.ml → IronGear → WristAssist → Pipelines")
