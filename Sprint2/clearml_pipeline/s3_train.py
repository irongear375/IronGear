"""
Sprint 2 ClearML Pipeline — Stage 3: Train
==========================================
The monolithic pipeline's Stage 5 (Model Training).

THREE MODES — selected via the `mode` parameter:

  1. "ingest"  (DEFAULT for SageMaker)
     Skips training entirely. Ingests an existing best.pt (your Kaggle-trained
     Exp A/B/C model) and registers it as this stage's output. Use this when
     you've already trained on Kaggle — no re-training, no risk to existing work.

  2. "smoke"
     Short training run (2 epochs, imgsz=640, batch=4) purely to demonstrate
     the pipeline wiring. SageMaker-compatible. Not clinically useful.

  3. "full"
     Full training using the same hyperparams as the monolith's Experiment A/B/C
     configs. Requires a suitable GPU — DO NOT run on SageMaker (~2GB usable VRAM).
     Intended for Kaggle or a cloud VM with clearml-agent.

Team: Iron Gear | Subject: 42174 AI Studio (Autumn 2026)
"""

from pathlib import Path
import time
import shutil
from clearml import Task, OutputModel

# Experiment configs — mirrors Sprint2_pipeline.py EXPERIMENT_CONFIGS
EXPERIMENT_CONFIGS = {
    "A": {"imgsz": 1280, "batch": 8,  "epochs": 150, "mixup": 0.15, "copy_paste": 0.3, "cls": 1.5},
    "B": {"imgsz":  640, "batch": 16, "epochs": 100, "mixup": 0.15, "copy_paste": 0.3, "cls": 1.5},
    "C": {"imgsz": 1280, "batch": 8,  "epochs": 150, "mixup": 0.0,  "copy_paste": 0.0, "cls": 0.5},
}

# -----------------------------------------------------------------------------
# 1. Initialise ClearML Task
# -----------------------------------------------------------------------------
task = Task.init(
    project_name="IronGear/WristAssist",
    task_name="Pipeline step 3 train model",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
    tags=["sprint2", "pipeline", "stage_train"],
)

# -----------------------------------------------------------------------------
# 2. Parameters — overridden by PipelineController
# -----------------------------------------------------------------------------
args = {
    "process_task_id": "bc93b375108843fbb4476812043e7f09",          # injected from stage_process.id

    # MODE (pick one) — default is "ingest" to protect your existing Kaggle work
    "mode": "ingest",
    "pretrained_weights_path": "/home/sagemaker-user/user-default-efs/IronGear/Sprint2/models/best.pt",  # required for mode=ingest — path to your Kaggle best.pt
    "experiment": "B",              # which Exp config to use for mode=full ('A'/'B'/'C')

    # Smoke / full mode params (used only when NOT in ingest mode)
    "model_name": "yolo11l.pt",
    "device": "0",
    "workers": 4,
    "seed": 42,
    "project_dir": "/home/sagemaker-user/user-default-efs/IronGear/Sprint2/runs",
    "run_name": "pipeline_run",

    # Smoke-test overrides
    "smoke_epochs": 2,
    "smoke_imgsz": 640,
    "smoke_batch": 4,
}
task.connect(args)

print("[stage_train] Parameters:")
for k, v in args.items():
    print(f"  {k}: {v}")

if not args["process_task_id"]:
    raise ValueError("process_task_id required — controller must inject stage_process.id")

# -----------------------------------------------------------------------------
# 3. Fetch dataset.yaml from stage_process
# -----------------------------------------------------------------------------
upstream = Task.get_task(task_id=args["process_task_id"])
print(f"[stage_train] Upstream task: {upstream.id} — {upstream.name}")
dataset_yaml_local = upstream.artifacts["dataset_yaml"].get_local_copy()
print(f"[stage_train] dataset.yaml: {dataset_yaml_local}")

# -----------------------------------------------------------------------------
# 4. MODE SWITCH
# -----------------------------------------------------------------------------
mode = args["mode"].lower()
training_minutes = 0.0
run_dir = None

if mode == "ingest":
    # -------------------------------------------------------------------------
    # INGEST MODE — use existing best.pt (your Kaggle-trained Exp A)
    # -------------------------------------------------------------------------
    pretrained = Path(args["pretrained_weights_path"])
    if not pretrained.exists():
        raise FileNotFoundError(
            f"mode=ingest requires a valid pretrained_weights_path. "
            f"Got: {pretrained}. Pass your Kaggle-produced best.pt here."
        )
    print(f"[stage_train] INGEST MODE — using existing weights: {pretrained}")
    best_pt_path = pretrained

elif mode in ("smoke", "full"):
    # -------------------------------------------------------------------------
    # TRAINING MODES — actually train
    # -------------------------------------------------------------------------
    from ultralytics import YOLO

    if mode == "full":
        exp_cfg = EXPERIMENT_CONFIGS[args["experiment"]]
        epochs, imgsz, batch = exp_cfg["epochs"], exp_cfg["imgsz"], exp_cfg["batch"]
        mixup, copy_paste, cls = exp_cfg["mixup"], exp_cfg["copy_paste"], exp_cfg["cls"]
        print(f"[stage_train] FULL MODE — Experiment {args['experiment']}")
    else:
        epochs, imgsz, batch = args["smoke_epochs"], args["smoke_imgsz"], args["smoke_batch"]
        mixup = copy_paste = 0.0
        cls = 0.5
        print(f"[stage_train] SMOKE MODE — minimal demo run")

    print(f"[stage_train]   model={args['model_name']} epochs={epochs} imgsz={imgsz} batch={batch}")

    project_dir = Path(args["project_dir"])
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args["model_name"])
    t0 = time.time()
    model.train(
        data=dataset_yaml_local,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=args["device"],
        workers=int(args["workers"]),
        seed=int(args["seed"]),
        mixup=float(mixup),
        copy_paste=float(copy_paste),
        cls=float(cls),
        project=str(project_dir),
        name=args["run_name"],
        exist_ok=True,
        verbose=True,
    )
    training_minutes = (time.time() - t0) / 60
    run_dir = project_dir / args["run_name"]
    best_pt_path = run_dir / "weights" / "best.pt"
    if not best_pt_path.exists():
        best_pt_path = run_dir / "weights" / "last.pt"
    if not best_pt_path.exists():
        raise FileNotFoundError(f"No best.pt or last.pt in {run_dir / 'weights'}")
    print(f"[stage_train] Training complete in {training_minutes:.1f} min")
    print(f"[stage_train] Weights: {best_pt_path}")

else:
    raise ValueError(f"Unknown mode: {mode!r}. Use 'ingest', 'smoke', or 'full'.")

# -----------------------------------------------------------------------------
# 5. Register best.pt + dataset.yaml as artifacts for stage_evaluate
# -----------------------------------------------------------------------------
task.upload_artifact(name="best_pt", artifact_object=str(best_pt_path))
task.upload_artifact(name="dataset_yaml", artifact_object=dataset_yaml_local)
task.upload_artifact(
    name="train_summary",
    artifact_object={
        "mode": mode,
        "weights_path": str(best_pt_path),
        "training_minutes": round(training_minutes, 2),
        "experiment": args["experiment"] if mode == "full" else None,
    },
)

# Register as ClearML model (shows in Models tab)
output_model = OutputModel(task=task, name="WristAssist_YOLO11l")
output_model.update_weights(weights_filename=str(best_pt_path))

logger = task.get_logger()
logger.report_single_value(name="training_minutes", value=float(training_minutes))

print(f"[stage_train] Task ID: {task.id}")
print("[stage_train] Done.")
