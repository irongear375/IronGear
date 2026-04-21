"""
Sprint 2 ClearML Pipeline — Stage 4: Evaluate
==============================================
Combines the monolithic pipeline's Stage 6 (Model Evaluation),
Stage 7 (Approval Gate), and Stage 8 (Artifact Export) into one DAG node.

CRITICAL FIX — the monolithic Sprint2_pipeline.py had this bug:
    lg.report_scalar("evaluation", "mAP@0.5", 0, self.eval_results["map50"])
The arguments are wrong: `0` is passed as `value` (int) and the mAP float is
passed as `iteration`. ClearML's type checker rejects the float-as-iteration
with "Expected iter of type int, got float". THIS killed Exp A at stage 6.

In this file, ALL ClearML scalar calls go through _safe_scalar / _safe_single
which cast to native types and swallow logging errors, so evaluation never
dies because of a metrics-logging quirk.

Responsibilities:
  - Pull best.pt + dataset.yaml from stage_train by task ID
  - Run YOLO val on the locked test split
  - Log per-class AP@0.5, mAP@0.5, mAP@0.5:0.95, precision, recall
  - Evaluate Sprint 2 approval gate: mAP > 0.685 AND fracture_AP >= 0.90
  - Write eval_summary.json + export figures + register as artifacts
  - Tag the task with 'gate-passed' or 'gate-failed'

Team: Iron Gear | Subject: 42174 AI Studio (Autumn 2026)
"""

from pathlib import Path
import json
import shutil
from clearml import Task

# -----------------------------------------------------------------------------
# Safe ClearML logging helpers — fixes the TypeError that killed Exp A
# -----------------------------------------------------------------------------
def _safe_scalar(logger, title, series, value, iteration=0):
    """
    Correct ClearML signature: report_scalar(title, series, value, iteration)
    This wrapper enforces value=float and iteration=int (what ClearML requires),
    and swallows any logging error so evaluation can always complete.
    """
    try:
        logger.report_scalar(
            title=str(title),
            series=str(series),
            value=float(value) if value is not None else 0.0,
            iteration=int(iteration) if iteration is not None else 0,
        )
    except Exception as e:
        print(f"  [ClearML scalar skip] {title}/{series}: {e}")


def _safe_single(logger, name, value):
    try:
        logger.report_single_value(name=str(name), value=float(value))
    except Exception as e:
        print(f"  [ClearML single skip] {name}: {e}")


# -----------------------------------------------------------------------------
# 1. Initialise ClearML Task
# -----------------------------------------------------------------------------
task = Task.init(
    project_name="IronGear/WristAssist",
    task_name="Pipeline step 4 evaluate model",
    task_type=Task.TaskTypes.testing,
    reuse_last_task_id=False,
    tags=["sprint2", "pipeline", "stage_evaluate"],
)

# -----------------------------------------------------------------------------
# 2. Parameters
# -----------------------------------------------------------------------------
args = {
    "train_task_id": "5ce4088c7260413b9e867f6d195f3729",     # injected from stage_train.id
    "imgsz": 1280,           # match training imgsz for fairest eval (Exp A used 1280)
    "batch": 8,
    "device": "0",
    "conf_threshold": 0.001, # low threshold for mAP calculation
    "iou_threshold": 0.6,
    "split": "test",
    "min_map50": 0.685,      # Sprint 2 approval gate: > Sprint 1 baseline
    "min_fracture_ap": 0.90, # Sprint 2 approval gate: fracture must be >= 0.90
    "output_dir": "/home/sagemaker-user/user-default-efs/IronGear/Sprint2/eval",
}
task.connect(args)

print("[stage_evaluate] Parameters:")
for k, v in args.items():
    print(f"  {k}: {v}")

if not args["train_task_id"]:
    raise ValueError("train_task_id required — controller must inject stage_train.id")

# -----------------------------------------------------------------------------
# 3. Fetch best.pt + dataset.yaml from stage_train
# -----------------------------------------------------------------------------
upstream = Task.get_task(task_id=args["train_task_id"])
print(f"[stage_evaluate] Upstream task: {upstream.id} — {upstream.name}")

best_pt_local = upstream.artifacts["best_pt"].get_local_copy()
dataset_yaml_local = upstream.artifacts["dataset_yaml"].get_local_copy()

print(f"[stage_evaluate] best.pt:      {best_pt_local}")
print(f"[stage_evaluate] dataset.yaml: {dataset_yaml_local}")

# -----------------------------------------------------------------------------
# 4. Run YOLO validation
# -----------------------------------------------------------------------------
from ultralytics import YOLO

model = YOLO(best_pt_local)
print(f"[stage_evaluate] Running YOLO val on split='{args['split']}'")

val_results = model.val(
    data=dataset_yaml_local,
    imgsz=int(args["imgsz"]),
    batch=int(args["batch"]),
    device=args["device"],
    conf=float(args["conf_threshold"]),
    iou=float(args["iou_threshold"]),
    split=args["split"],
    verbose=True,
)

# -----------------------------------------------------------------------------
# 5. Extract metrics (cast every numpy scalar to native float)
# -----------------------------------------------------------------------------
eval_results = {
    "map50": round(float(val_results.box.map50), 4),
    "map50_95": round(float(val_results.box.map), 4),
    "precision": round(float(val_results.box.mp), 4),
    "recall": round(float(val_results.box.mr), 4),
}

CLASS_NAMES = ["fracture", "metal_implant", "periosteal_reaction", "pronator_sign", "text"]
per_class_ap50 = {}
for i, name in enumerate(CLASS_NAMES):
    if i < len(val_results.box.ap50):
        per_class_ap50[name] = round(float(val_results.box.ap50[i]), 4)
eval_results["per_class_ap50"] = per_class_ap50

# Sprint 1 baseline for diff reporting
SPRINT1_AP = {
    "fracture": 0.939,
    "metal_implant": 0.865,
    "periosteal_reaction": 0.649,
    "pronator_sign": 0.665,
}

print(f"\n[stage_evaluate] === RESULTS ===")
print(f"  mAP@0.5:       {eval_results['map50']:.4f}")
print(f"  mAP@0.5:0.95:  {eval_results['map50_95']:.4f}")
print(f"  Precision:     {eval_results['precision']:.4f}")
print(f"  Recall:        {eval_results['recall']:.4f}")
print(f"\n  Per-class AP@0.5 (vs Sprint 1):")
for name, ap in per_class_ap50.items():
    s1 = SPRINT1_AP.get(name)
    diff = f"({'+' if ap > s1 else ''}{ap - s1:+.3f})" if s1 else "(NEW)"
    print(f"    {name:<22} {ap:.4f}  {diff}")

# -----------------------------------------------------------------------------
# 6. Log to ClearML — using SAFE wrappers (fixes the Exp A TypeError)
# -----------------------------------------------------------------------------
logger = task.get_logger()

_safe_single(logger, "test_mAP50",    eval_results["map50"])
_safe_single(logger, "test_mAP50_95", eval_results["map50_95"])
_safe_single(logger, "test_precision", eval_results["precision"])
_safe_single(logger, "test_recall",   eval_results["recall"])

for name, ap in per_class_ap50.items():
    _safe_single(logger, f"AP50_{name}", ap)
    # Correct arg order: (title, series, value, iteration)
    _safe_scalar(logger, "Per-Class AP@0.5", name, ap, iteration=0)

# -----------------------------------------------------------------------------
# 7. Approval gate
# -----------------------------------------------------------------------------
fracture_ap = per_class_ap50.get("fracture", 0.0)
gate_map = eval_results["map50"] > float(args["min_map50"])
gate_fracture = fracture_ap >= float(args["min_fracture_ap"])
approved = bool(gate_map and gate_fracture)

print(f"\n[stage_evaluate] === APPROVAL GATE ===")
print(f"  mAP@0.5 > {args['min_map50']}:        "
      f"{eval_results['map50']:.4f}  [{'PASS' if gate_map else 'FAIL'}]")
print(f"  fracture_AP >= {args['min_fracture_ap']}:   "
      f"{fracture_ap:.4f}  [{'PASS' if gate_fracture else 'FAIL'}]")
print(f"\n  Overall: {'APPROVED' if approved else 'NOT APPROVED'}")

task.add_tags(["gate-passed"] if approved else ["gate-failed"])
_safe_single(logger, "gate_passed", 1.0 if approved else 0.0)

# -----------------------------------------------------------------------------
# 8. Artifact export — eval_summary.json + figures
# -----------------------------------------------------------------------------
out_dir = Path(args["output_dir"])
(out_dir / "figures").mkdir(parents=True, exist_ok=True)

summary = {
    "sprint": 2,
    "approved": approved,
    "criteria": {
        f"mAP50 > {args['min_map50']}": gate_map,
        f"fracture_AP >= {args['min_fracture_ap']}": gate_fracture,
    },
    **eval_results,
    "sprint1_baseline_per_class": SPRINT1_AP,
}

summary_path = out_dir / "eval_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[stage_evaluate] eval_summary.json → {summary_path}")

# Copy figures from the training run if the path is inferable
try:
    train_dir = Path(best_pt_local).parent.parent
    for fig in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                "F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png"]:
        fp = train_dir / fig
        if fp.exists():
            shutil.copy2(fp, out_dir / "figures" / fig)
except Exception as e:
    print(f"[stage_evaluate] Figure copy skipped: {e}")

# Register as artifacts
task.upload_artifact(name="eval_summary", artifact_object=str(summary_path))
task.upload_artifact(name="approval_gate", artifact_object=summary)
task.upload_artifact(name="best_pt", artifact_object=str(best_pt_local))

print(f"[stage_evaluate] Task ID: {task.id}")
print("[stage_evaluate] Done.")
