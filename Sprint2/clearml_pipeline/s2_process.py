"""
Sprint 2 ClearML Pipeline — Stage 2: Process
=============================================
Combines the monolithic pipeline's Stage 3 (Data Validation) and
Stage 4 (Data Preparation) into one DAG node.

Responsibilities:
  - Pull artifacts from stage_data by task ID
  - Validate Sprint 2 5-class label schema across train/val/test
  - Symlink images + copy labels into yolo_dataset/{images,labels}/{split}/
  - Handle oversampled (*_os*) files (Sprint 2 has these in train split)
  - Write dataset.yaml and upload as artifact for stage_train

Downstream: stage_train consumes dataset_yaml + (optional) yolo_dataset_root.

Team: Iron Gear | Subject: 42174 AI Studio (Autumn 2026)
"""

from pathlib import Path
from collections import Counter
import os, re, json, shutil
import yaml
import pandas as pd
from clearml import Task

# -----------------------------------------------------------------------------
# 1. Initialise ClearML Task
# -----------------------------------------------------------------------------
task = Task.init(
    project_name="IronGear/WristAssist",
    task_name="Pipeline step 2 data processing",
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["sprint2", "pipeline", "stage_process"],
)

# -----------------------------------------------------------------------------
# 2. Parameters — overridden by PipelineController
# -----------------------------------------------------------------------------
args = {
    "dataset_task_id": "fd313f49cf4b482785245d53761f2c09",   # injected by PipelineController from stage_data.id
    "yolo_dir": "/home/sagemaker-user/user-default-efs/IronGear/data/yolo_dataset",
    "skip_prep": False,      # set True to only validate (faster re-runs)
}
task.connect(args)

print("[stage_process] Parameters:")
for k, v in args.items():
    print(f"  {k}: {v}")

if not args["dataset_task_id"]:
    raise ValueError("dataset_task_id is required — controller must inject stage_data.id")

# -----------------------------------------------------------------------------
# 3. Sprint 2 class schema — FIXED, do not edit
# -----------------------------------------------------------------------------
CLASS_NAMES = {
    0: "fracture",
    1: "metal_implant",
    2: "periosteal_reaction",
    3: "pronator_sign",
    4: "text",
}
EXPECTED_IDS = set(CLASS_NAMES.keys())
NC = len(CLASS_NAMES)

# -----------------------------------------------------------------------------
# 4. Fetch artifacts from stage_data
# -----------------------------------------------------------------------------
upstream = Task.get_task(task_id=args["dataset_task_id"])
print(f"[stage_process] Upstream task: {upstream.id} — {upstream.name}")

master_local = upstream.artifacts["master_index"].get_local_copy()
image_index = upstream.artifacts["image_index"].get()  # dict: stem → path
paths = upstream.artifacts["paths"].get()

master = pd.read_csv(master_local)
labels_src = Path(paths["labels_src"])
print(f"[stage_process] labels_src: {labels_src}")
print(f"[stage_process] master rows: {len(master):,}")
print(f"[stage_process] image_index entries: {len(image_index):,}")

# -----------------------------------------------------------------------------
# 5. Validation — per-split class counts + unexpected ID detection
# -----------------------------------------------------------------------------
print("[stage_process] === VALIDATION ===")
validation_report = {}
for split in ["train", "val", "test"]:
    d = labels_src / split
    if not d.exists():
        print(f"  MISSING split: {split}")
        continue
    label_files = list(d.glob("*.txt"))
    counts = Counter()
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counts[int(parts[0])] += 1
    unexpected = set(counts) - EXPECTED_IDS
    print(f"  {split}: {len(label_files):,} label files")
    for cid in sorted(counts):
        name = CLASS_NAMES.get(cid, "UNKNOWN")
        marker = "OK" if cid in EXPECTED_IDS else "!!"
        print(f"    [{marker}] class {cid} {name:<22} {counts[cid]:>7,}")
    if unexpected:
        print(f"    UNEXPECTED class IDs: {unexpected}")
    validation_report[split] = {
        "label_files": len(label_files),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "unexpected_ids": [int(x) for x in unexpected],
    }

# -----------------------------------------------------------------------------
# 6. Preparation — symlink images + copy labels into yolo_dir
# -----------------------------------------------------------------------------
yolo_dir = Path(args["yolo_dir"])

if args["skip_prep"]:
    print("[stage_process] skip_prep=True — only validating, not building yolo_dir")
else:
    print("[stage_process] === PREPARATION ===")

    dst_labels = yolo_dir / "labels"

    # SAFETY CHECK: if labels_src IS dst_labels (i.e. the dataset is already
    # assembled at yolo_dir), DO NOT delete-and-recopy — that would destroy
    # the very labels we're trying to use. Just skip to validation of images.
    same_path = labels_src.resolve() == dst_labels.resolve()

    if same_path:
        print(f"[stage_process] labels_src == yolo_dir/labels ({dst_labels})")
        print(f"[stage_process] Dataset already assembled — skipping label copy (safe mode)")
    else:
        # Different paths — safe to do the delete-and-copy dance.
        for split in ["train", "val", "test"]:
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        if dst_labels.exists():
            shutil.rmtree(dst_labels)
        shutil.copytree(labels_src, dst_labels)
        print(f"[stage_process] labels copied to {dst_labels}")

    # Ensure image directories exist (idempotent)
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    # Symlink images from master_index rows (idempotent — skips existing)
    # Also safe when images are already linked from a previous run.
    linked = already_there = missing = 0
    for _, row in master.iterrows():
        dst = yolo_dir / "images" / row["split"] / f"{row['stem']}{row['ext']}"
        if dst.exists() or dst.is_symlink():
            already_there += 1
            continue
        src = image_index.get(row["stem"])
        if src and Path(src).exists():
            try:
                os.symlink(src, dst)
                linked += 1
            except FileExistsError:
                already_there += 1
        else:
            missing += 1
    print(f"[stage_process] Images — newly linked: {linked:,} | already present: {already_there:,} | missing: {missing}")

    # Handle oversampled files in train split (*_os1, *_os2, ...)
    print("[stage_process] Handling oversampled images...")
    ti = yolo_dir / "images" / "train"
    tl = yolo_dir / "labels" / "train"
    os_fixed = 0
    for lf in tl.glob("*_os*.txt"):
        stem = lf.stem
        if list(ti.glob(f"{stem}.*")):
            continue
        orig_stem = re.sub(r"_os\d+$", "", stem)
        orig = list(ti.glob(f"{orig_stem}.*"))
        if not orig:
            raw = image_index.get(orig_stem)
            if raw and Path(raw).exists():
                orig = [Path(raw)]
        if orig:
            src = orig[0].resolve() if orig[0].is_symlink() else orig[0]
            dst = ti / f"{stem}{orig[0].suffix}"
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                    os_fixed += 1
                except FileExistsError:
                    pass
    print(f"[stage_process] Oversampled linked: {os_fixed:,}")

    # Final per-split counts
    print("[stage_process] Final yolo_dataset counts:")
    for split in ["train", "val", "test"]:
        ni = len(list((yolo_dir / "images" / split).glob("*")))
        nl = len(list((yolo_dir / "labels" / split).glob("*.txt")))
        status = "OK" if ni == nl else "MISMATCH"
        print(f"    {split}: {ni:,} images / {nl:,} labels  [{status}]")

# -----------------------------------------------------------------------------
# 7. Write dataset.yaml
# -----------------------------------------------------------------------------
yolo_dir.mkdir(parents=True, exist_ok=True)
cfg = {
    "path": str(yolo_dir),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": NC,
    "names": CLASS_NAMES,
}
yaml_path = yolo_dir / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"[stage_process] dataset.yaml written → {yaml_path}")

# -----------------------------------------------------------------------------
# 8. Register artifacts for stage_train
# -----------------------------------------------------------------------------
task.upload_artifact(name="dataset_yaml", artifact_object=str(yaml_path))
task.upload_artifact(name="validation_report", artifact_object=validation_report)
task.upload_artifact(name="class_names", artifact_object=CLASS_NAMES)
task.upload_artifact(name="yolo_dir", artifact_object={"path": str(yolo_dir)})

logger = task.get_logger()
for split, rpt in validation_report.items():
    logger.report_single_value(name=f"{split}_label_files", value=float(rpt["label_files"]))

print(f"[stage_process] Task ID: {task.id}")
print("[stage_process] Done.")
