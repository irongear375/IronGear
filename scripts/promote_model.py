"""
WristAssist AI -- Model Promotion Script
Downloads a gate-passed model from ClearML and pushes to GitHub,
which auto-triggers the CD workflow to deploy to HF Spaces.

This closes the loop:
    Kaggle training -> ClearML registry -> THIS SCRIPT -> GitHub -> HF Spaces

Usage:
    # Promote the latest gate-passed model
    python scripts/promote_model.py

    # Promote a specific model by ClearML ID
    python scripts/promote_model.py --model-id <CLEARML_MODEL_ID>

    # Dry run (download only, don't push)
    python scripts/promote_model.py --dry-run

What this script does:
    1. Queries ClearML for the latest gate-passed model
    2. Downloads the weights to models/best.pt
    3. Updates experiment_config.yaml with the new version
    4. Commits and pushes to main
    5. GitHub CD workflow auto-deploys to HF Spaces
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wristassist.promote")

PROJECT = "IronGear/WristAssist"
MODELS_DIR = Path("models")
CONFIG_PATH = Path("experiment_config.yaml")


def run_cmd(cmd, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def find_model(model_id=None):
    """Find the model to promote from ClearML."""
    from clearml import Model

    if model_id:
        logger.info(f"Looking up model: {model_id}")
        model = Model(model_id=model_id)
        return model

    # Find latest gate-passed model
    logger.info("Searching for latest gate-passed model...")
    models = Model.query_models(
        project_name=PROJECT,
        tags=["gate-passed"],
        only_published=False,
        max_results=10,
    )

    if not models:
        logger.error("No gate-passed models found in ClearML")
        sys.exit(1)

    # Pick the most recent one
    model = models[0]
    logger.info(f"Found model: {model.id}")
    logger.info(f"  Name: {model.name}")
    logger.info(f"  Tags: {model.tags}")
    return model


def download_model(model):
    """Download model weights from ClearML."""
    logger.info("Downloading model weights...")
    local_path = model.get_local_copy()

    if local_path is None:
        logger.error("Failed to download model from ClearML")
        sys.exit(1)

    dest = MODELS_DIR / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)

    # Copy to models/best.pt
    import shutil
    shutil.copy2(local_path, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved to: {dest} ({size_mb:.1f} MB)")
    return dest


def update_config(version_string):
    """Update experiment_config.yaml with new model version."""
    import yaml

    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found at {CONFIG_PATH}, skipping update")
        return

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    old_version = config.get("production", {}).get("model_version", "unknown")
    config["production"]["model_version"] = version_string

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Updated config: {old_version} -> {version_string}")


def update_model_card(model, eval_metrics=None):
    """Update model_card.json with new model info."""
    card_path = MODELS_DIR / "model_card.json"
    if not card_path.exists():
        logger.warning("model_card.json not found, skipping update")
        return

    with open(card_path, "r") as f:
        card = json.load(f)

    card["model_version"] = model.name or model.id
    card["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    card["lineage"]["clearml_model_id"] = model.id
    card["lineage"]["clearml_tags"] = list(model.tags) if model.tags else []

    if eval_metrics:
        card["evaluation"]["overall_metrics"]["mAP50"] = eval_metrics.get("mAP50", 0)
        card["approval_gate"]["results"]["mAP50"] = eval_metrics.get("mAP50", 0)
        card["approval_gate"]["results"]["fracture_AP50"] = eval_metrics.get("fracture_AP50", 0)

    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    logger.info(f"  Updated model_card.json")


def git_push(version_string):
    """Commit and push to main, triggering CD."""
    logger.info("Committing and pushing to GitHub...")

    run_cmd("git add models/best.pt models/model_card.json experiment_config.yaml")
    run_cmd(f'git commit -m "Promote model: {version_string} (gate-passed, auto-deploy)"')
    run_cmd("git push origin main")

    logger.info("  Pushed to main. CD workflow will deploy to HF Spaces automatically.")


def main():
    parser = argparse.ArgumentParser(description="WristAssist AI -- Model Promotion")
    parser.add_argument("--model-id", default=None, help="ClearML model ID to promote")
    parser.add_argument("--version", default=None, help="Version string for the new model")
    parser.add_argument("--dry-run", action="store_true", help="Download only, don't push")
    args = parser.parse_args()

    print("=" * 60)
    print("  WristAssist AI -- Model Promotion")
    print("  ClearML -> GitHub -> HF Spaces (automated)")
    print("=" * 60)

    # Step 1: Find the model
    model = find_model(args.model_id)

    # Step 2: Download weights
    dest = download_model(model)

    # Step 3: Generate version string
    version = args.version or f"sprint3_yolo11l_{datetime.now().strftime('%Y%m%d')}_v1"
    logger.info(f"Version: {version}")

    # Step 4: Update config and model card
    update_config(version)
    update_model_card(model)

    # Step 5: Push to GitHub (triggers CD)
    if args.dry_run:
        logger.info("DRY RUN -- skipping git push")
        logger.info("Files updated locally. Review and push manually.")
    else:
        git_push(version)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  DRY RUN COMPLETE -- files updated locally")
    else:
        print("  PROMOTION COMPLETE")
        print(f"  Model {model.id} -> GitHub main -> HF Spaces")
        print(f"  CD workflow will deploy automatically in ~5 minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
