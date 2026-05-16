"""
WristAssist AI — Model Registry CLI
SCRUM-56: Model registration, promotion, and listing via ClearML.

Provides three commands:
    register  — Register a local best.pt in ClearML with metadata and tags
    promote   — Promote a registered model to production (demotes current)
    list      — List all registered models with tags and metrics

Usage:
    # Register a new model
    python scripts/register_model.py register \
        --weights models/best.pt \
        --version sprint3_yolo11l_hpo_v1 \
        --metrics '{"mAP50": 0.870, "fracture_AP50": 0.945}'

    # Promote a model to production
    python scripts/register_model.py promote \
        --model-id <CLEARML_MODEL_ID>

    # List all models
    python scripts/register_model.py list

Requires:
    pip install clearml pyyaml
    ClearML credentials configured (clearml-init or env vars)
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("wristassist.registry")

CONFIG_PATH = Path(__file__).parent.parent / "experiment_config.yaml"


def load_config() -> dict:
    """Load experiment configuration."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def sha256_checksum(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Command: register
# ============================================================================


def cmd_register(args):
    """Register a model artifact in ClearML with metadata."""
    from clearml import OutputModel, Task

    config = load_config()
    project = config.get("infrastructure", {}).get(
        "clearml_project", "IronGear/WristAssist"
    )

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    # Parse metrics if provided
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for --metrics: {e}")
            sys.exit(1)

    # Compute checksum
    checksum = sha256_checksum(str(weights_path))
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)

    logger.info(f"Registering model: {args.version}")
    logger.info(f"  Weights: {weights_path} ({file_size_mb:.1f} MB)")
    logger.info(f"  SHA-256: {checksum[:16]}...")

    # Create a ClearML task for the registration
    task = Task.init(
        project_name=project,
        task_name=f"Model Registration — {args.version}",
        task_type=Task.TaskTypes.custom,
    )
    task.add_tags(["model_registration", args.version])

    # Log metadata
    task.connect(
        {
            "model_version": args.version,
            "file_size_mb": round(file_size_mb, 1),
            "sha256": checksum,
            **metrics,
        }
    )

    # Log metrics as scalars for ClearML UI
    for key, value in metrics.items():
        task.get_logger().report_scalar(
            "registration_metrics", key, value=float(value), iteration=0
        )

    # Register the model
    output_model = OutputModel(
        task=task,
        name=args.version,
        framework="PyTorch",
    )
    output_model.update_weights(
        weights_filename=str(weights_path),
        auto_delete_file=False,
    )

    # Apply tags
    tags = ["registered", args.version]
    if args.tags:
        tags.extend(args.tags.split(","))
    output_model.tags = tags

    # Upload model card as artifact
    model_card = {
        "model_version": args.version,
        "architecture": "YOLO11l",
        "framework": "PyTorch",
        "file_size_mb": round(file_size_mb, 1),
        "sha256_checksum": checksum,
        "metrics": metrics,
        "tags": tags,
        "clearml_model_id": output_model.id,
        "clearml_task_id": task.id,
    }
    task.upload_artifact("model_card", artifact_object=model_card)

    task.close()

    logger.info(f"Model registered successfully")
    logger.info(f"  ClearML Model ID: {output_model.id}")
    logger.info(f"  ClearML Task ID:  {task.id}")
    logger.info(f"  Tags: {tags}")

    return output_model.id


# ============================================================================
# Command: promote
# ============================================================================


def cmd_promote(args):
    """Promote a model to production, demoting the current production model."""
    from clearml import Model

    config = load_config()
    project = config.get("infrastructure", {}).get(
        "clearml_project", "IronGear/WristAssist"
    )

    logger.info(f"Promoting model {args.model_id} to production")

    # Find and demote current production model
    current_production = Model.query_models(
        project_name=project,
        tags=["production"],
        only_published=False,
    )

    for model in current_production:
        if model.id == args.model_id:
            continue
        old_tags = list(model.tags) if model.tags else []
        new_tags = [t for t in old_tags if t != "production"]
        new_tags.append("archived")
        model.tags = new_tags
        logger.info(
            f"  Demoted previous production model: {model.id} "
            f"(now tagged: {new_tags})"
        )

    # Promote the target model
    target = Model(model_id=args.model_id)
    current_tags = list(target.tags) if target.tags else []
    new_tags = [t for t in current_tags if t not in ("candidate", "archived")]
    if "production" not in new_tags:
        new_tags.append("production")
    if "gate-passed" not in new_tags:
        new_tags.append("gate-passed")
    target.tags = new_tags

    logger.info(f"  Promoted model {args.model_id} to production")
    logger.info(f"  Tags: {new_tags}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Copy the model weights to models/best.pt")
    logger.info("  2. Update production.model_version in experiment_config.yaml")
    logger.info("  3. Push to main — CD will deploy automatically")


# ============================================================================
# Command: list
# ============================================================================


def cmd_list(args):
    """List all registered models in the ClearML project."""
    from clearml import Model

    config = load_config()
    project = config.get("infrastructure", {}).get(
        "clearml_project", "IronGear/WristAssist"
    )

    models = Model.query_models(
        project_name=project,
        only_published=False,
        max_results=args.limit,
    )

    if not models:
        logger.info("No models found in the registry.")
        return

    logger.info(f"Found {len(models)} model(s) in {project}:\n")

    header = f"{'Model ID':<40} {'Name':<35} {'Tags'}"
    logger.info(header)
    logger.info("-" * len(header))

    for model in models:
        name = model.name or "(unnamed)"
        tags = ", ".join(model.tags) if model.tags else "(no tags)"
        is_prod = " ← PRODUCTION" if model.tags and "production" in model.tags else ""
        logger.info(f"{model.id:<40} {name:<35} {tags}{is_prod}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="WristAssist AI — Model Registry CLI (SCRUM-56)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register
    reg = subparsers.add_parser(
        "register", help="Register a model in ClearML"
    )
    reg.add_argument(
        "--weights", required=True, help="Path to best.pt file"
    )
    reg.add_argument(
        "--version", required=True,
        help="Version string (e.g. sprint3_yolo11l_hpo_v1)",
    )
    reg.add_argument(
        "--metrics", default=None,
        help='JSON string of metrics (e.g. \'{"mAP50": 0.858}\')',
    )
    reg.add_argument(
        "--tags", default=None,
        help="Comma-separated additional tags",
    )

    # promote
    promo = subparsers.add_parser(
        "promote", help="Promote a model to production"
    )
    promo.add_argument(
        "--model-id", required=True, help="ClearML model ID to promote"
    )

    # list
    lst = subparsers.add_parser(
        "list", help="List all registered models"
    )
    lst.add_argument(
        "--limit", type=int, default=20, help="Max models to list"
    )

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "promote":
        cmd_promote(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
