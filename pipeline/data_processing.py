"""
WristAssist AI — Data Processing Pipeline Stage
SCRUM-63: Parameterised data processing script for CI/CD integration

Converts raw GRAZPEDWRI-DX labels into a clean, split, class-balanced
YOLO-format dataset. All configuration is read from experiment_config.yaml
(SCRUM-64), eliminating hardcoded values and enabling config-driven experiments.

Pipeline stages:
    1. Load configuration from YAML
    2. Load master_index.csv (patient-level split from Sprint 1)
    3. Remap raw 9-class labels to project 5-class schema
    4. Apply oversampling to training split only
    5. Write dataset.yaml for YOLO training
    6. Validate output and emit summary

Usage:
    # From CLI (standalone)
    python data_processing.py --config experiment_config.yaml --output ./dataset

    # From ClearML pipeline (stage_0)
    python data_processing.py --config experiment_config.yaml --output ./dataset --ci

    # Dry run (validate config without writing files)
    python data_processing.py --config experiment_config.yaml --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("wristassist.data_processing")


# ============================================================================
# Configuration loader
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate experiment configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["dataset", "class_schema", "oversampling"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate class schema
    schema = config["class_schema"]
    if schema["num_classes"] != len(schema["names"]):
        raise ValueError(
            f"num_classes ({schema['num_classes']}) doesn't match "
            f"names count ({len(schema['names'])})"
        )

    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"  Dataset: {config['dataset']['name']}")
    logger.info(f"  Classes: {schema['num_classes']} — {list(schema['names'].values())}")
    logger.info(f"  Oversampling: {'enabled' if config['oversampling']['enabled'] else 'disabled'}")

    return config


# ============================================================================
# Master index loader
# ============================================================================

def load_master_index(index_path: str) -> dict:
    """
    Load master_index.csv which contains patient-level split assignments.
    Returns {image_stem: split_name} mapping.

    The master_index was created in Sprint 1 and preserves patient-level
    splitting to prevent data leakage across train/val/test.
    """
    import csv

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Master index not found: {index_path}")

    index = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: image_id (or filename), split
            stem = row.get("image_id", row.get("filename", ""))
            split = row.get("split", "")
            if stem and split:
                # Normalise stem (remove extension if present)
                stem = Path(stem).stem
                index[stem] = split

    logger.info(f"Master index loaded: {len(index)} images across splits")

    # Count per split
    split_counts = defaultdict(int)
    for split in index.values():
        split_counts[split] += 1
    for split, count in sorted(split_counts.items()):
        logger.info(f"  {split}: {count} images")

    return index


# ============================================================================
# Label remapping
# ============================================================================

def remap_label_file(
    src_path: Path,
    dst_path: Path,
    raw_to_project: dict,
) -> dict:
    """
    Read a raw YOLO label file, remap class IDs per the project schema,
    and write the remapped labels to dst_path.

    Lines with class IDs mapping to null (dropped classes) are removed.
    Returns stats: {kept: int, dropped: int, class_counts: dict}
    """
    stats = {"kept": 0, "dropped": 0, "class_counts": defaultdict(int)}
    output_lines = []

    if not src_path.exists():
        return stats

    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            raw_id = int(parts[0])
            project_id = raw_to_project.get(raw_id)

            if project_id is None:
                stats["dropped"] += 1
                continue

            # Replace class ID, keep bbox coordinates unchanged
            parts[0] = str(project_id)
            output_lines.append(" ".join(parts))
            stats["kept"] += 1
            stats["class_counts"][project_id] += 1

    # Write output (even if empty — YOLO handles empty label files)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("\n".join(output_lines))
        if output_lines:
            f.write("\n")

    return stats


def process_labels(
    raw_labels_dir: Path,
    output_dir: Path,
    master_index: dict,
    config: dict,
) -> dict:
    """
    Process all raw label files: remap classes and write to split directories.

    Output structure:
        output_dir/labels/train/*.txt
        output_dir/labels/val/*.txt
        output_dir/labels/test/*.txt
    """
    raw_to_project = {}
    for raw_id_str, project_id in config["class_schema"]["raw_to_project"].items():
        raw_to_project[int(raw_id_str)] = project_id

    total_stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "annotations_kept": 0,
        "annotations_dropped": 0,
        "per_split": defaultdict(lambda: {"files": 0, "annotations": 0}),
        "per_class": defaultdict(int),
    }

    # Find all raw label files
    label_files = sorted(raw_labels_dir.rglob("*.txt"))
    logger.info(f"Found {len(label_files)} raw label files in {raw_labels_dir}")

    for src_path in label_files:
        stem = src_path.stem

        # Look up split assignment
        split = master_index.get(stem)
        if split is None:
            total_stats["files_skipped"] += 1
            continue

        # Determine output path
        dst_path = output_dir / "labels" / split / f"{stem}.txt"

        # Remap
        stats = remap_label_file(src_path, dst_path, raw_to_project)

        total_stats["files_processed"] += 1
        total_stats["annotations_kept"] += stats["kept"]
        total_stats["annotations_dropped"] += stats["dropped"]
        total_stats["per_split"][split]["files"] += 1
        total_stats["per_split"][split]["annotations"] += stats["kept"]
        for cls_id, count in stats["class_counts"].items():
            total_stats["per_class"][cls_id] += count

    # Log summary
    logger.info(f"Label processing complete:")
    logger.info(f"  Files processed: {total_stats['files_processed']}")
    logger.info(f"  Files skipped (not in index): {total_stats['files_skipped']}")
    logger.info(f"  Annotations kept: {total_stats['annotations_kept']}")
    logger.info(f"  Annotations dropped: {total_stats['annotations_dropped']}")

    class_names = config["class_schema"]["names"]
    for cls_id, count in sorted(total_stats["per_class"].items()):
        name = class_names.get(cls_id, f"unknown_{cls_id}")
        logger.info(f"  Class {cls_id} ({name}): {count} boxes")

    for split, info in sorted(total_stats["per_split"].items()):
        logger.info(f"  Split {split}: {info['files']} files, {info['annotations']} annotations")

    return total_stats


# ============================================================================
# Image symlinking
# ============================================================================

def symlink_images(
    images_dir: Path,
    output_dir: Path,
    master_index: dict,
) -> dict:
    """
    Create symlinks from the raw images to the split output directories.

    Output structure:
        output_dir/images/train/*.png
        output_dir/images/val/*.png
        output_dir/images/test/*.png
    """
    stats = {"linked": 0, "skipped": 0, "per_split": defaultdict(int)}

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_files)} images in {images_dir}")

    for img_path in image_files:
        stem = img_path.stem
        split = master_index.get(stem)
        if split is None:
            stats["skipped"] += 1
            continue

        dst_dir = output_dir / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / img_path.name

        if dst_path.exists():
            continue

        try:
            # Use symlink if on same filesystem, else copy
            dst_path.symlink_to(img_path.resolve())
            stats["linked"] += 1
            stats["per_split"][split] += 1
        except OSError:
            # Fallback to copy (e.g., cross-device on Kaggle)
            shutil.copy2(img_path, dst_path)
            stats["linked"] += 1
            stats["per_split"][split] += 1

    logger.info(f"Image linking complete: {stats['linked']} linked, {stats['skipped']} skipped")
    for split, count in sorted(stats["per_split"].items()):
        logger.info(f"  {split}: {count} images")

    return stats


# ============================================================================
# Oversampling
# ============================================================================

def apply_oversampling(
    output_dir: Path,
    config: dict,
) -> dict:
    """
    Apply class-aware oversampling to the training split ONLY.

    For each class with a multiplier > 1, find all training images
    containing that class and create N-1 additional copies with
    '_os{i}' suffix (matching Sprint 2 convention).
    """
    if not config["oversampling"]["enabled"]:
        logger.info("Oversampling disabled in config — skipping")
        return {"total_copies": 0}

    multipliers = config["oversampling"]["multipliers"]
    class_names = config["class_schema"]["names"]

    # Build reverse lookup: class_name → class_id
    name_to_id = {v: k for k, v in class_names.items()}

    train_labels_dir = output_dir / "labels" / "train"
    train_images_dir = output_dir / "images" / "train"

    stats = {"total_copies": 0, "per_class": {}}

    for class_name, multiplier in multipliers.items():
        if multiplier <= 1:
            continue

        class_id = name_to_id.get(class_name)
        if class_id is None:
            logger.warning(f"Oversampling class '{class_name}' not in schema — skipping")
            continue

        copies_made = 0
        extra_copies = multiplier - 1

        # Find training labels containing this class
        for label_path in sorted(train_labels_dir.glob("*.txt")):
            # Skip already-oversampled files
            if "_os" in label_path.stem:
                continue

            # Check if this label contains the target class
            with open(label_path, "r") as f:
                lines = f.readlines()

            has_class = any(
                line.strip().split()[0] == str(class_id)
                for line in lines if line.strip()
            )

            if not has_class:
                continue

            # Create copies
            stem = label_path.stem
            for i in range(1, extra_copies + 1):
                os_stem = f"{stem}_os{i}"

                # Copy label
                os_label = train_labels_dir / f"{os_stem}.txt"
                if not os_label.exists():
                    shutil.copy2(label_path, os_label)

                # Copy/symlink image
                for ext in [".png", ".jpg", ".jpeg"]:
                    src_img = train_images_dir / f"{stem}{ext}"
                    if src_img.exists():
                        dst_img = train_images_dir / f"{os_stem}{ext}"
                        if not dst_img.exists():
                            shutil.copy2(src_img, dst_img)
                        break

                copies_made += 1

        stats["per_class"][class_name] = {
            "multiplier": multiplier,
            "copies_created": copies_made,
        }
        logger.info(f"  Oversampled {class_name} (×{multiplier}): {copies_made} copies created")

    stats["total_copies"] = sum(
        v["copies_created"] for v in stats["per_class"].values()
    )
    logger.info(f"Oversampling complete: {stats['total_copies']} total copies")

    return stats


# ============================================================================
# dataset.yaml generation
# ============================================================================

def write_dataset_yaml(output_dir: Path, config: dict):
    """
    Write dataset.yaml for YOLO training with auto-detected paths.
    Supports Kaggle, SageMaker, and local environments.
    """
    class_names = config["class_schema"]["names"]

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": config["class_schema"]["num_classes"],
        "names": class_names,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"dataset.yaml written to {yaml_path}")
    return yaml_path


# ============================================================================
# Validation
# ============================================================================

def validate_output(output_dir: Path, config: dict) -> dict:
    """
    Validate the processed dataset: check file counts, class coverage,
    and directory structure.
    """
    errors = []
    warnings = []
    summary = {}

    # Check directory structure
    for split in ["train", "val", "test"]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"Missing directory: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"Missing directory: {lbl_dir}")
            continue

        img_count = len(list(img_dir.glob("*")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))

        summary[split] = {"images": img_count, "labels": lbl_count}

        if img_count == 0:
            errors.append(f"No images in {split} split")
        if lbl_count == 0:
            errors.append(f"No labels in {split} split")

    # Check dataset.yaml exists
    if not (output_dir / "dataset.yaml").exists():
        errors.append("dataset.yaml not found")

    # Check class coverage in training labels
    train_labels = output_dir / "labels" / "train"
    if train_labels.exists():
        class_ids_found = set()
        for lbl_path in train_labels.glob("*.txt"):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids_found.add(int(parts[0]))

        expected_ids = set(config["class_schema"]["names"].keys())
        missing = expected_ids - class_ids_found
        if missing:
            warnings.append(
                f"Classes not found in training labels: "
                f"{[config['class_schema']['names'][i] for i in missing]}"
            )

    validation = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }

    if errors:
        logger.error(f"Validation FAILED with {len(errors)} errors:")
        for e in errors:
            logger.error(f"  ✗ {e}")
    else:
        logger.info("Validation PASSED")

    if warnings:
        for w in warnings:
            logger.warning(f"  ⚠ {w}")

    for split, counts in summary.items():
        logger.info(f"  {split}: {counts['images']} images, {counts['labels']} labels")

    return validation


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(
    config_path: str,
    raw_labels_dir: str,
    raw_images_dir: str,
    output_dir: str,
    master_index_path: Optional[str] = None,
    dry_run: bool = False,
    ci_mode: bool = False,
) -> dict:
    """
    Execute the full data processing pipeline.

    Args:
        config_path: Path to experiment_config.yaml
        raw_labels_dir: Path to raw GRAZPEDWRI-DX label files
        raw_images_dir: Path to raw GRAZPEDWRI-DX images
        output_dir: Output directory for processed dataset
        master_index_path: Override path for master_index.csv
        dry_run: Validate config without writing files
        ci_mode: Exit with non-zero code on validation failure
    """
    logger.info("=" * 60)
    logger.info("WristAssist AI — Data Processing Pipeline")
    logger.info("=" * 60)

    # Step 1: Load config
    config = load_config(config_path)

    if dry_run:
        logger.info("DRY RUN — config validated, no files will be written")
        return {"status": "dry_run", "config_valid": True}

    # Step 2: Load master index
    index_path = master_index_path or config["dataset"]["master_index"]
    master_index = load_master_index(index_path)

    # Step 3: Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 4: Remap labels
    logger.info("Step 4/7: Remapping labels...")
    label_stats = process_labels(
        Path(raw_labels_dir), out, master_index, config
    )

    # Step 5: Symlink images
    logger.info("Step 5/7: Linking images...")
    image_stats = symlink_images(
        Path(raw_images_dir), out, master_index
    )

    # Step 6: Apply oversampling
    logger.info("Step 6/7: Applying oversampling...")
    os_stats = apply_oversampling(out, config)

    # Step 7: Write dataset.yaml
    logger.info("Step 7/7: Writing dataset.yaml...")
    yaml_path = write_dataset_yaml(out, config)

    # Validate
    logger.info("Validating output...")
    validation = validate_output(out, config)

    # Emit summary
    result = {
        "status": "success" if validation["valid"] else "failed",
        "config_path": config_path,
        "output_dir": str(out),
        "dataset_yaml": str(yaml_path),
        "label_stats": {
            "files_processed": label_stats["files_processed"],
            "annotations_kept": label_stats["annotations_kept"],
            "annotations_dropped": label_stats["annotations_dropped"],
        },
        "image_stats": {
            "linked": image_stats["linked"],
        },
        "oversampling": os_stats,
        "validation": validation,
    }

    # Write summary JSON
    summary_path = out / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Summary written to {summary_path}")

    # CI mode: exit non-zero on failure
    if ci_mode and not validation["valid"]:
        logger.error("CI MODE: Validation failed — exiting with code 1")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete: {result['status']}")
    logger.info("=" * 60)

    return result


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WristAssist AI — Data Processing Pipeline (SCRUM-63)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment_config.yaml",
    )
    parser.add_argument(
        "--raw-labels", required=False, default="./raw_labels",
        help="Path to raw GRAZPEDWRI-DX label files",
    )
    parser.add_argument(
        "--raw-images", required=False, default="./raw_images",
        help="Path to raw GRAZPEDWRI-DX image files",
    )
    parser.add_argument(
        "--output", required=False, default="./dataset",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--master-index", required=False, default=None,
        help="Override path for master_index.csv",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config without writing files",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: exit non-zero on validation failure",
    )

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        raw_labels_dir=args.raw_labels,
        raw_images_dir=args.raw_images,
        output_dir=args.output,
        master_index_path=args.master_index,
        dry_run=args.dry_run,
        ci_mode=args.ci,
    )
