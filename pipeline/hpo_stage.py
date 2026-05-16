"""
WristAssist AI — Hyperparameter Optimization Stage
Extends the ClearML pipeline with automated HPO using ClearML's
HyperParameterOptimizer (UniformIntegerParameterRange / UniformParameterRange).

Pipeline position: stage_data → stage_process → stage_train → stage_hpo → stage_final_model

The HPO stage:
1. Takes the base training task ID (from stage_train) as a template
2. Defines search ranges for key hyperparameters
3. Launches parallel training jobs via ClearML agent queue
4. Tracks validation mAP@0.5 as the optimization objective
5. Saves the best parameters as a ClearML artifact

Requirements:
    - ClearML agent(s) running on GPU machine(s) in the 'default' queue
    - At least 2 workers: 1 for the HPO controller + 1 for spawned training tasks
    - Base training task must be registered in ClearML (Draft status)

Usage:
    # Standalone
    python hpo_stage.py --base-task-id <TASK_ID> --config experiment_config.yaml

    # Called by pipeline_controller.py as stage_hpo
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
logger = logging.getLogger("wristassist.hpo")


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_hpo(
    base_task_id: str,
    config: dict,
    experiment: str = "B",
    max_jobs: int = 5,
    time_limit_minutes: int = 120,
    queue_name: str = "default",
):
    """
    Run ClearML HyperParameterOptimizer against a base training task.

    Args:
        base_task_id: ClearML task ID of the base training task (template)
        config: Loaded experiment config dict
        experiment: Experiment variant to use as base
        max_jobs: Maximum total HPO trials to run
        time_limit_minutes: Maximum wall-clock time for HPO
        queue_name: ClearML queue to submit training jobs to
    """
    from clearml import Task
    from clearml.automation import (
        HyperParameterOptimizer,
        UniformIntegerParameterRange,
        UniformParameterRange,
    )

    # Get project info from config
    project = config.get("infrastructure", {}).get(
        "clearml_project", "IronGear/WristAssist"
    )
    exp_params = config.get("experiments", {}).get(experiment, {})

    # ── Initialise HPO controller task ──
    task = Task.init(
        project_name=project,
        task_name=f"WristAssist HPO — Exp {experiment}",
        task_type=Task.TaskTypes.optimizer,
    )
    task.add_tags(["hpo", f"exp_{experiment}", "sprint3"])

    logger.info(f"HPO controller task: {task.id}")
    logger.info(f"Base training task: {base_task_id}")
    logger.info(f"Queue: {queue_name}")
    logger.info(f"Max jobs: {max_jobs}, time limit: {time_limit_minutes} min")

    # ── Define hyperparameter search space ──
    # Search ranges are centred around the production Experiment B values
    hpo_search_space = [
        UniformIntegerParameterRange(
            name="General/batch",
            min_value=8,
            max_value=32,
            step_size=8,
        ),
        UniformIntegerParameterRange(
            name="General/epochs",
            min_value=30,
            max_value=100,
            step_size=10,
        ),
        UniformParameterRange(
            name="General/lr0",
            min_value=0.0001,
            max_value=0.01,
            step_size=0.001,
        ),
        UniformParameterRange(
            name="General/mixup",
            min_value=0.0,
            max_value=0.3,
            step_size=0.05,
        ),
        UniformParameterRange(
            name="General/copy_paste",
            min_value=0.0,
            max_value=0.5,
            step_size=0.1,
        ),
        UniformParameterRange(
            name="General/cls",
            min_value=0.5,
            max_value=2.0,
            step_size=0.25,
        ),
    ]

    # ── Configure the optimizer ──
    optimizer = HyperParameterOptimizer(
        # Base task to clone and modify for each trial
        base_task_id=base_task_id,

        # Hyperparameter search space
        hyper_parameters=hpo_search_space,

        # Optimization objective: maximize mAP@0.5
        objective_metric_title="val",
        objective_metric_series="mAP50",
        objective_metric_sign="max",

        # Execution settings
        max_number_of_concurrent_tasks=2,
        optimizer_class=None,  # Uses default RandomSearch
        execution_queue=queue_name,

        # Budget
        total_max_jobs=max_jobs,
        max_iteration_per_job=0,  # 0 = no limit, run to completion
        time_limit_per_job=time_limit_minutes / max_jobs,  # Per-job limit
    )

    # ── Run HPO ──
    logger.info("Starting HPO search...")
    logger.info(f"Search space:")
    for param in hpo_search_space:
        logger.info(f"  {param.name}: [{param.min_value} — {param.max_value}]")

    # Start and wait for completion
    optimizer.start(
        job_complete_callback=None,
    )
    optimizer.set_time_limit(
        in_minutes=time_limit_minutes,
    )
    optimizer.wait()

    # ── Extract best parameters ──
    best_experiment = optimizer.get_top_experiments(top_k=1)

    if not best_experiment:
        logger.error("HPO completed but no experiments succeeded")
        task.upload_artifact(
            name="hpo_result",
            artifact_object={"status": "no_results"},
        )
        return None

    best_task = best_experiment[0]
    best_params = best_task.get_parameters()
    best_metrics = best_task.get_last_scalar_metrics()

    # Extract the mAP@0.5 value
    best_map50 = None
    try:
        best_map50 = best_metrics.get("val", {}).get("mAP50", {}).get("last", 0)
    except (AttributeError, KeyError):
        pass

    hpo_result = {
        "status": "completed",
        "best_task_id": best_task.id,
        "best_mAP50": best_map50,
        "best_parameters": {
            k.replace("General/", ""): v
            for k, v in best_params.items()
            if k.startswith("General/")
        },
        "total_jobs": max_jobs,
        "time_limit_minutes": time_limit_minutes,
        "experiment_base": experiment,
    }

    logger.info("=" * 60)
    logger.info("HPO COMPLETE")
    logger.info(f"  Best task: {best_task.id}")
    logger.info(f"  Best mAP@0.5: {best_map50}")
    logger.info(f"  Best parameters:")
    for k, v in hpo_result["best_parameters"].items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 60)

    # Save best parameters as artifact for stage_final_model
    task.upload_artifact(
        name="best_parameters",
        artifact_object=hpo_result,
    )

    optimizer.stop()

    return hpo_result


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WristAssist AI — HPO Stage"
    )
    parser.add_argument(
        "--base-task-id", required=True,
        help="ClearML task ID of the base training task",
    )
    parser.add_argument(
        "--config", default="experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--experiment", default="B",
        help="Base experiment variant (default: B)",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=5,
        help="Maximum HPO trials (default: 5)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=120,
        help="Time limit in minutes (default: 120)",
    )
    parser.add_argument(
        "--queue", default="default",
        help="ClearML queue name (default: default)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    result = run_hpo(
        base_task_id=args.base_task_id,
        config=config,
        experiment=args.experiment,
        max_jobs=args.max_jobs,
        time_limit_minutes=args.time_limit,
        queue_name=args.queue,
    )

    if result is None:
        sys.exit(1)
