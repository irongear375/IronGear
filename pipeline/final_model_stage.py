"""
WristAssist AI — Final Model Training Stage
Takes the best hyperparameters from stage_hpo, trains a final model,
evaluates against the approval gate, and registers the artifact.

Pipeline position: stage_data → stage_process → stage_train → stage_hpo → stage_final_model

The final model stage:
1. Reads best_parameters artifact from stage_hpo
2. Trains YOLO11l with those parameters on the full training set
3. Evaluates on the locked test split
4. Checks the approval gate (mAP50 > 0.685, fracture AP50 >= 0.90)
5. Registers the model artifact in ClearML with 'production' tag

Usage:
    # Standalone
    python final_model_stage.py --hpo-task-id <HPO_TASK_ID> --config experiment_config.yaml

    # Called by pipeline_controller.py as stage_final_model
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("wristassist.final_model")


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_final_model(
    hpo_task_id: str,
    config: dict,
    dataset_yaml: str = "dataset.yaml",
):
    """
    Train the final model using best HPO parameters.

    Args:
        hpo_task_id: ClearML task ID of the completed HPO stage
        config: Loaded experiment config dict
        dataset_yaml: Path to the YOLO dataset.yaml file
    """
    from clearml import Task, OutputModel
    from ultralytics import YOLO

    project = config.get("infrastructure", {}).get(
        "clearml_project", "IronGear/WristAssist"
    )
    gate = config.get("approval_gate", {})

    # ── Initialise task ──
    task = Task.init(
        project_name=project,
        task_name="WristAssist Final Model (HPO-optimised)",
        task_type=Task.TaskTypes.training,
    )
    task.add_tags(["final_model", "hpo_optimised", "sprint3"])

    # ── Retrieve best parameters from HPO ──
    logger.info(f"Loading best parameters from HPO task: {hpo_task_id}")

    hpo_task = Task.get_task(task_id=hpo_task_id)
    best_params_artifact = hpo_task.artifacts.get("best_parameters")

    if best_params_artifact is None:
        logger.error("No best_parameters artifact found in HPO task")
        task.mark_failed(status_reason="HPO artifact missing")
        return None

    best_params = best_params_artifact.get()
    if isinstance(best_params, str):
        best_params = json.loads(best_params)

    params = best_params.get("best_parameters", {})
    logger.info("Best HPO parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")

    # Connect parameters for ClearML tracking
    task.connect(params)

    # ── Extract training hyperparameters ──
    model_name = params.get("model", "yolo11l.pt")
    imgsz = int(params.get("imgsz", 640))
    batch = int(params.get("batch", 16))
    epochs = int(params.get("epochs", 100))
    lr0 = float(params.get("lr0", 0.01))
    mixup = float(params.get("mixup", 0.15))
    copy_paste = float(params.get("copy_paste", 0.3))
    cls_weight = float(params.get("cls", 1.5))
    patience = int(params.get("patience", 15))

    # ── Train ──
    logger.info("=" * 60)
    logger.info("FINAL MODEL TRAINING")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  imgsz: {imgsz}, batch: {batch}, epochs: {epochs}")
    logger.info(f"  lr0: {lr0}, mixup: {mixup}, copy_paste: {copy_paste}")
    logger.info("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        patience=patience,
        lr0=lr0,
        mixup=mixup,
        copy_paste=copy_paste,
        cls=cls_weight,
        optimizer="auto",
        project="runs/final_model",
        name="hpo_best",
        exist_ok=True,
        verbose=True,
    )

    # ── Evaluate on test split ──
    logger.info("Evaluating final model on locked test split...")
    best_pt = Path("runs/final_model/hpo_best/weights/best.pt")

    if not best_pt.exists():
        logger.error(f"best.pt not found at {best_pt}")
        task.mark_failed(status_reason="Training did not produce best.pt")
        return None

    eval_model = YOLO(str(best_pt))
    eval_results = eval_model.val(
        data=dataset_yaml,
        split="test",
        imgsz=imgsz,
    )

    # ── Extract metrics ──
    map50 = float(eval_results.results_dict.get("metrics/mAP50(B)", 0))
    map50_95 = float(eval_results.results_dict.get("metrics/mAP50-95(B)", 0))

    # Per-class AP (class 0 = fracture)
    fracture_ap50 = 0.0
    if hasattr(eval_results, "ap_class_index") and 0 in eval_results.ap_class_index:
        idx = list(eval_results.ap_class_index).index(0)
        fracture_ap50 = float(eval_results.ap50[idx])

    # Log metrics to ClearML
    task.get_logger().report_scalar("eval", "mAP50", value=map50, iteration=0)
    task.get_logger().report_scalar("eval", "mAP50-95", value=map50_95, iteration=0)
    task.get_logger().report_scalar("eval", "fracture_AP50", value=fracture_ap50, iteration=0)

    logger.info(f"Final model metrics:")
    logger.info(f"  mAP@0.5:     {map50:.4f}")
    logger.info(f"  mAP@0.5:0.95: {map50_95:.4f}")
    logger.info(f"  Fracture AP:  {fracture_ap50:.4f}")

    # ── Approval gate ──
    min_map50 = gate.get("mAP50_min", 0.685)
    min_fracture_ap = gate.get("fracture_AP50_min", 0.90)

    gate_passed = map50 >= min_map50 and fracture_ap50 >= min_fracture_ap

    logger.info("=" * 60)
    if gate_passed:
        logger.info("APPROVAL GATE: PASSED")
        task.add_tags(["gate-passed", "production-candidate"])
    else:
        logger.warning("APPROVAL GATE: FAILED")
        logger.warning(f"  mAP50: {map50:.4f} (need >= {min_map50})")
        logger.warning(f"  Fracture AP: {fracture_ap50:.4f} (need >= {min_fracture_ap})")
        task.add_tags(["gate-failed"])
    logger.info("=" * 60)

    # ── Register model artifact ──
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
        logger.info(f"Model registered with 'production' tag")
    else:
        output_model.tags = ["candidate", "hpo_optimised", "gate-failed"]

    # ── Save result summary ──
    result = {
        "status": "gate_passed" if gate_passed else "gate_failed",
        "best_pt_path": str(best_pt),
        "metrics": {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "fracture_AP50": fracture_ap50,
        },
        "gate": {
            "mAP50_min": min_map50,
            "fracture_AP50_min": min_fracture_ap,
            "passed": gate_passed,
        },
        "hpo_parameters": params,
        "model_id": output_model.id,
    }

    task.upload_artifact(
        name="final_model_result",
        artifact_object=result,
    )

    summary_path = Path("runs/final_model/hpo_best/final_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Model artifact ID: {output_model.id}")

    return result


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WristAssist AI — Final Model Stage"
    )
    parser.add_argument(
        "--hpo-task-id", required=True,
        help="ClearML task ID of the completed HPO stage",
    )
    parser.add_argument(
        "--config", default="experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dataset-yaml", default="dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    result = run_final_model(
        hpo_task_id=args.hpo_task_id,
        config=config,
        dataset_yaml=args.dataset_yaml,
    )

    if result is None or not result.get("gate", {}).get("passed", False):
        logger.warning("Final model did not pass approval gate")
        sys.exit(1)
