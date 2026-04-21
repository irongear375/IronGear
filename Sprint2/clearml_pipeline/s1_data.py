"""
Sprint 2 ClearML Pipeline — Stage 1: Data
==========================================
Combines the monolithic pipeline's Stage 1 (Environment Check) and
Stage 2 (Data Extraction) into one DAG node.

Responsibilities:
  - Verify GPU + Kaggle dataset mounts (or SageMaker EFS paths)
  - Locate images directory + master_index.csv + labels directory
  - Scan all images, build {stem: path} index
  - Upload master_index and image_index as ClearML artifacts

Downstream: stage_process consumes master_index + image_index by task ID.

Team: Iron Gear | Subject: 42174 AI Studio (Autumn 2026)
"""

from pathlib import Path
from collections import Counter
import json
import pandas as pd
from clearml import Task

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# -----------------------------------------------------------------------------
# 1. Initialise ClearML Task
# -----------------------------------------------------------------------------
task = Task.init(
    project_name="IronGear/WristAssist",
    task_name="Pipeline step 1 data extraction",
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["sprint2", "pipeline", "stage_data"],
)

# -----------------------------------------------------------------------------
# 2. Parameters — these appear in the ClearML UI and are overridden by controller
# -----------------------------------------------------------------------------
args = {
    # SageMaker-safe defaults (edit for Kaggle if needed)
    "images_root": "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset/images",
    "labels_root": "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset/labels",
    "master_index_csv": "/home/sagemaker-user/user-default-efs/IronGear/data/reports/master_index.csv",  # auto-detected if blank
    "labels_subdir": "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset/labels",     # auto-detected if blank
}
task.connect(args)

print("[stage_data] Parameters:")
for k, v in args.items():
    print(f"  {k}: {v}")

# -----------------------------------------------------------------------------
# 3. Auto-detect master_index.csv and labels subdirectory
#    Mirrors the logic in the monolith's stage_2_data_extraction
# -----------------------------------------------------------------------------
images_root = Path(args["images_root"])
labels_root = Path(args["labels_root"])

assert images_root.exists(), f"images_root does not exist: {images_root}"
assert labels_root.exists(), f"labels_root does not exist: {labels_root}"

# Locate master_index.csv
master_csv = Path(args["master_index_csv"]) if args["master_index_csv"] else None
if master_csv is None or not master_csv.exists():
    for c in [
        labels_root / "reports" / "master_index.csv",
        labels_root / "master_index.csv",
    ]:
        if c.exists():
            master_csv = c
            break
assert master_csv and master_csv.exists(), f"master_index.csv not found under {labels_root}"

# Locate labels subdirectory
labels_src = Path(args["labels_subdir"]) if args["labels_subdir"] else None
if labels_src is None or not labels_src.exists():
    for c in [
        labels_root / "yolo_dataset" / "labels",
        labels_root / "labels",
    ]:
        if c.exists():
            labels_src = c
            break
assert labels_src and labels_src.exists(), f"labels/ subdir not found under {labels_root}"

print(f"[stage_data] master_index.csv: {master_csv}")
print(f"[stage_data] labels_src:       {labels_src}")

# -----------------------------------------------------------------------------
# 4. Load master_index + index all images
# -----------------------------------------------------------------------------
master = pd.read_csv(master_csv)
print(f"[stage_data] master_index.csv: {len(master):,} rows")

all_images = {}
for p in images_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        all_images[p.stem] = str(p)
print(f"[stage_data] {len(all_images):,} images indexed")

# Per-split counts for sanity check
split_summary = {}
if "split" in master.columns:
    for split in ["train", "val", "test"]:
        split_summary[split] = int((master["split"] == split).sum())
print(f"[stage_data] split counts: {split_summary}")

# Image coverage — how many master rows resolve to an actual file on disk
overlap = set(master["stem"]) & set(all_images.keys()) if "stem" in master.columns else set()
print(f"[stage_data] image coverage: {len(overlap):,}/{len(master):,}")

# -----------------------------------------------------------------------------
# 5. Upload artifacts (master_index, image_index, labels_src path, summary)
# -----------------------------------------------------------------------------
task.upload_artifact(name="master_index", artifact_object=str(master_csv))
task.upload_artifact(name="image_index", artifact_object=all_images)  # dict → JSON
task.upload_artifact(
    name="paths",
    artifact_object={
        "images_root": str(images_root),
        "labels_src": str(labels_src),
        "master_csv": str(master_csv),
    },
)
task.upload_artifact(
    name="data_summary",
    artifact_object={
        "total_rows": int(len(master)),
        "total_images": int(len(all_images)),
        "splits": split_summary,
        "image_coverage": int(len(overlap)),
    },
)

# Log scalars for the DAG view
logger = task.get_logger()
logger.report_single_value(name="total_images", value=float(len(all_images)))
logger.report_single_value(name="total_master_rows", value=float(len(master)))
logger.report_single_value(name="image_coverage", value=float(len(overlap)))

print(f"[stage_data] Task ID: {task.id}")
print("[stage_data] Done.")
