"""
======================================================================
Sprint 2 — MLOps Level 1: Automated ML Pipeline
======================================================================
Project:  WristAssist AI | Team: Iron Gear
Subject:  42174 AI Studio — Autumn 2026 (UTS)

Pipeline Stages (Google Cloud MLOps Level 1 reference):
  1. Environment Check    — GPU, credentials, datasets
  2. Data Extraction      — Locate images + labels from Kaggle inputs
  3. Data Validation      — Verify splits, classes, image-label matching
  4. Data Preparation     — Assemble YOLO dataset (symlink + copy)
  5. Model Training       — YOLO11x, resume-safe, ClearML tracked
  6. Model Evaluation     — Per-class AP on locked test set
  7. Model Validation     — Approval gate: mAP>0.685 AND fracture AP>=0.90
  8. Artifact Export      — best.pt, figures, eval_summary.json

Collaborative Experiments (3 team members, 1 ClearML project):
  Experiment A (Hashim)  — High-res + heavy augmentation
  Experiment B (Vaibhav) — Base-res baseline (isolates resolution)
  Experiment C (Praveer) — High-res + light augmentation

Usage:
  from pipeline import Pipeline
  pipe = Pipeline(experiment="A")
  pipe.run()
======================================================================
"""

import os, sys, time, json, yaml, shutil, re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════

_SHARED = {
    "images_dataset": "grazpedwri-dx",
    "labels_dataset": "irongear-s2-labels",
    "nc": 5,
    "class_names": {0:"fracture", 1:"metal_implant", 2:"periosteal_reaction", 3:"pronator_sign", 4:"text"},
    "model_name": "yolo11l.pt",  # YOLO11l (large) — YOLO11x OOMs on T4 at 1280
    "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "cos_lr": True,
    "warmup_epochs": 5, "workers": 4, "cache": "disk", "seed": 42,
    "fliplr": 0.5, "flipud": 0.0, "degrees": 8, "scale": 0.8, "mosaic": 1.0,
    "min_map50": 0.685, "min_fracture_ap": 0.90,
    "sprint1_results": {
        "map50": 0.685,
        "per_class_ap50": {"fracture": 0.939, "metal_implant": 0.865,
                           "periosteal_reaction": 0.649, "pronator_sign": 0.665}
    },
    "yolo_dir": "/kaggle/working/yolo_dataset",
    "output_dir": "/kaggle/working/Sprint2",
}

EXPERIMENT_CONFIGS = {
    "A": {**_SHARED,
        "experiment_name": "A_high_res_heavy_aug",
        "experiment_owner": "Muhammad Hashim",
        "hypothesis": "High resolution (1280) + heavy augmentation gives best overall mAP",
        "imgsz": 1280, "batch": 8, "epochs": 150, "patience": 30, "device": "0,1",
        "mixup": 0.15, "copy_paste": 0.3, "cls": 1.5, "box": 8.0,
    },
    "B": {**_SHARED,
        "experiment_name": "B_base_res_baseline",
        "experiment_owner": "Vaibhav Bairathi",
        "hypothesis": "Isolate resolution effect — same aug as A but imgsz=640",
        "imgsz": 640, "batch": 16, "epochs": 100, "patience": 25, "device": "0,1",
        "mixup": 0.15, "copy_paste": 0.3, "cls": 1.5, "box": 8.0,
    },
    "C": {**_SHARED,
        "experiment_name": "C_high_res_light_aug",
        "experiment_owner": "Praveer Jain",
        "hypothesis": "Does heavy augmentation help or hurt rare class detection?",
        "imgsz": 1280, "batch": 8, "epochs": 150, "patience": 30, "device": "0,1",
        "mixup": 0.0, "copy_paste": 0.0, "cls": 0.5, "box": 7.5,
    },
}


class Pipeline:
    def __init__(self, experiment="A", config_overrides=None):
        assert experiment in EXPERIMENT_CONFIGS, f"Use 'A', 'B', or 'C', not '{experiment}'"
        self.experiment = experiment
        self.config = {**EXPERIMENT_CONFIGS[experiment], **(config_overrides or {})}
        self.yolo_dir = Path(self.config["yolo_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.yaml_path = self.best_pt = self.eval_results = self.gate_passed = None
        self.train_duration = None
        self.task = None
        self.clearml_ok = False
        self.run_log = {
            "pipeline_version": "sprint2_v2",
            "experiment": experiment,
            "experiment_name": self.config["experiment_name"],
            "experiment_owner": self.config["experiment_owner"],
            "hypothesis": self.config["hypothesis"],
            "started_at": datetime.now().isoformat(),
            "stages_completed": [], "stages_failed": [],
        }

    # ── STAGE 1: ENVIRONMENT CHECK ────────────────────────────────
    def stage_1_environment_check(self):
        print("\n" + "=" * 60)
        print("  STAGE 1: ENVIRONMENT CHECK")
        print("=" * 60)
        print(f"  Experiment: {self.experiment} — {self.config['experiment_name']}")
        print(f"  Owner:      {self.config['experiment_owner']}")
        print(f"  Hypothesis: {self.config['hypothesis']}")

        import torch
        assert torch.cuda.is_available(), "No GPU!"
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Accept direct paths (auto-detected by notebook) or construct from dataset names
        if self.config.get("images_input_path"):
            self.images_input = Path(self.config["images_input_path"])
        else:
            # Try common Kaggle mount patterns
            for candidate in [
                Path(f"/kaggle/input/{self.config['images_dataset']}"),
                Path(f"/kaggle/input/datasets/jasonroggy/{self.config['images_dataset']}"),
            ]:
                if candidate.exists():
                    self.images_input = candidate; break
            else:
                # Last resort: find images_part1 anywhere
                for p in Path("/kaggle/input").rglob("images_part1"):
                    self.images_input = p.parent; break

        if self.config.get("labels_input_path"):
            self.labels_input = Path(self.config["labels_input_path"])
        else:
            for candidate in [
                Path(f"/kaggle/input/{self.config['labels_dataset']}"),
                Path(f"/kaggle/input/datasets/vaibhavbairathi/{self.config['labels_dataset']}"),
            ]:
                if candidate.exists():
                    self.labels_input = candidate; break
            else:
                for p in Path("/kaggle/input").rglob("master_index.csv"):
                    self.labels_input = p.parent.parent; break

        assert hasattr(self, 'images_input') and self.images_input and self.images_input.exists(), \
            f"Images dataset not found. Check Kaggle inputs."
        assert hasattr(self, 'labels_input') and self.labels_input and self.labels_input.exists(), \
            f"Labels dataset not found. Check Kaggle inputs."

        try:
            from clearml import Task
            self.task = Task.init(
                project_name="IronGear/WristAssist",
                task_name=f"sprint2_exp{self.experiment}_{self.config['experiment_name']}",
                tags=["sprint2", f"exp_{self.experiment}",
                      self.config["experiment_owner"].split()[0].lower(),
                      f"imgsz{self.config['imgsz']}",
                      "mixup" if self.config.get("mixup", 0) > 0 else "no_mixup",
                      "copy_paste" if self.config.get("copy_paste", 0) > 0 else "no_copy_paste"],
                reuse_last_task_id=True)
            self.task.connect(self.config, name="pipeline_config")
            self.clearml_ok = True
            print(f"  ClearML: {self.task.id}")
        except Exception as e:
            print(f"  ClearML unavailable: {e}")

        print(f"\n  Config: imgsz={self.config['imgsz']}, batch={self.config['batch']}, "
              f"epochs={self.config['epochs']}, mixup={self.config['mixup']}, "
              f"copy_paste={self.config['copy_paste']}, cls={self.config['cls']}")
        self.run_log["stages_completed"].append("environment_check")
        print("\n  Stage 1 PASSED")

    # ── STAGE 2: DATA EXTRACTION ──────────────────────────────────
    def stage_2_data_extraction(self):
        print("\n" + "=" * 60)
        print("  STAGE 2: DATA EXTRACTION")
        print("=" * 60)
        import pandas as pd

        self.labels_src = None
        for c in [self.labels_input/"yolo_dataset"/"labels", self.labels_input/"labels"]:
            if c.exists(): self.labels_src = c; break
        assert self.labels_src, f"labels/ not found in {self.labels_input}"

        master_csv = None
        for c in [self.labels_input/"reports"/"master_index.csv",
                   self.labels_input/"master_index.csv"]:
            if c.exists(): master_csv = c; break
        assert master_csv, "master_index.csv not found"
        self.master = pd.read_csv(master_csv)
        print(f"  master_index.csv: {len(self.master):,} rows")

        print("  Scanning images...")
        self.all_images = {}
        for p in self.images_input.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                self.all_images[p.stem] = p
        print(f"  {len(self.all_images):,} images indexed")
        self.run_log["stages_completed"].append("data_extraction")
        print("\n  Stage 2 PASSED")

    # ── STAGE 3: DATA VALIDATION ──────────────────────────────────
    def stage_3_data_validation(self):
        print("\n" + "=" * 60)
        print("  STAGE 3: DATA VALIDATION")
        print("=" * 60)
        expected = set(self.config["class_names"].keys())
        for split in ["train", "val", "test"]:
            d = self.labels_src / split
            if not d.exists(): print(f"  MISSING: {split}"); continue
            counts = Counter()
            for lf in d.glob("*.txt"):
                with open(lf) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 5: counts[int(p[0])] += 1
            print(f"\n  {split}: {len(list(d.glob('*.txt'))):,} labels")
            for cid in sorted(counts):
                print(f"    {cid}: {self.config['class_names'].get(cid,'?'):<25} {counts[cid]:>7,}")
            bad = set(counts) - expected
            if bad: print(f"    UNEXPECTED IDs: {bad}")

        overlap = set(self.master["stem"]) & set(self.all_images)
        print(f"\n  Image coverage: {len(overlap):,}/{len(self.master):,}")
        self.run_log["stages_completed"].append("data_validation")
        print("\n  Stage 3 PASSED")

    # ── STAGE 4: DATA PREPARATION ─────────────────────────────────
    def stage_4_data_preparation(self):
        print("\n" + "=" * 60)
        print("  STAGE 4: DATA PREPARATION")
        print("=" * 60)
        from tqdm import tqdm

        for split in ["train", "val", "test"]:
            (self.yolo_dir/"images"/split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir/"labels"/split).mkdir(parents=True, exist_ok=True)

        dst_labels = self.yolo_dir / "labels"
        if dst_labels.exists(): shutil.rmtree(dst_labels)
        shutil.copytree(self.labels_src, dst_labels)
        print("  Labels copied")

        linked = missing = 0
        for _, row in tqdm(self.master.iterrows(), total=len(self.master), desc="  Images"):
            dst = self.yolo_dir/"images"/row["split"]/f"{row['stem']}{row['ext']}"
            if dst.exists(): linked += 1; continue
            src = self.all_images.get(row["stem"])
            if src: os.symlink(src, dst); linked += 1
            else: missing += 1
        print(f"  Linked: {linked:,} | Missing: {missing}")

        print("  Linking oversampled images...")
        os_fixed = 0
        ti = self.yolo_dir/"images"/"train"
        tl = self.yolo_dir/"labels"/"train"
        for lf in tl.glob("*_os*.txt"):
            stem = lf.stem
            if list(ti.glob(f"{stem}.*")): continue
            orig_stem = re.sub(r"_os\d+$", "", stem)
            orig = list(ti.glob(f"{orig_stem}.*"))
            if not orig:
                raw = self.all_images.get(orig_stem)
                if raw: orig = [raw]
            if orig:
                src = orig[0].resolve() if orig[0].is_symlink() else orig[0]
                dst = ti/f"{stem}{orig[0].suffix}"
                if not dst.exists(): os.symlink(src, dst); os_fixed += 1
        print(f"  Oversampled linked: {os_fixed:,}")

        for split in ["train", "val", "test"]:
            ni = len(list((self.yolo_dir/"images"/split).glob("*")))
            nl = len(list((self.yolo_dir/"labels"/split).glob("*.txt")))
            print(f"    {split}: {ni:,} imgs, {nl:,} lbls {'OK' if ni==nl else 'MISMATCH'}")

        cfg = {"path": str(self.yolo_dir), "train": "images/train",
               "val": "images/val", "test": "images/test",
               "nc": self.config["nc"], "names": self.config["class_names"]}
        self.yaml_path = self.yolo_dir / "dataset.yaml"
        with open(self.yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        self.run_log["stages_completed"].append("data_preparation")
        print("\n  Stage 4 PASSED")

    # ── STAGE 5: MODEL TRAINING ───────────────────────────────────
    def stage_5_model_training(self):
        print("\n" + "=" * 60)
        print(f"  STAGE 5: MODEL TRAINING (Experiment {self.experiment})")
        print("=" * 60)
        from ultralytics import YOLO

        resume_pt = None
        for lp in Path("/kaggle/working").rglob("last.pt"):
            resume_pt = str(lp); break

        model = YOLO(resume_pt) if resume_pt else YOLO(c.get("model_name", "yolo11l.pt"))
        print(f"  {'Resuming from ' + resume_pt if resume_pt else 'Fresh start'}")

        c = self.config
        t0 = time.time()
        model.train(
            data=str(self.yaml_path), epochs=c["epochs"], imgsz=c["imgsz"],
            batch=c["batch"], optimizer=c["optimizer"], lr0=c["lr0"], lrf=c["lrf"],
            cos_lr=c["cos_lr"], warmup_epochs=c["warmup_epochs"],
            device=c["device"], workers=c["workers"], cache=c["cache"],
            patience=c["patience"], mosaic=c["mosaic"], mixup=c["mixup"],
            copy_paste=c["copy_paste"], degrees=c["degrees"], scale=c["scale"],
            fliplr=c["fliplr"], flipud=c["flipud"], cls=c["cls"], box=c["box"],
            seed=c["seed"], resume=bool(resume_pt),
            project="/kaggle/working/Sprint2",
            name=f"exp{self.experiment}_{c['experiment_name']}", exist_ok=True)

        self.train_duration = (time.time() - t0) / 60
        best_pts = list(Path("/kaggle/working/Sprint2").rglob("best.pt"))
        assert best_pts, "best.pt not found!"
        self.best_pt = str(best_pts[0])
        print(f"\n  Training: {self.train_duration:.1f} min | Model: {self.best_pt}")
        self.run_log["stages_completed"].append("model_training")
        print("\n  Stage 5 PASSED")

    # ── STAGE 6: MODEL EVALUATION ─────────────────────────────────
    def stage_6_model_evaluation(self):
        print("\n" + "=" * 60)
        print("  STAGE 6: MODEL EVALUATION")
        print("=" * 60)
        from ultralytics import YOLO
        model = YOLO(self.best_pt)
        ev = model.val(data=str(self.yaml_path), split="test",
                       imgsz=self.config["imgsz"], batch=self.config["batch"], device="0")

        names = list(self.config["class_names"].values())
        per_class = {names[i]: float(ev.box.ap50[i])
                     for i in range(min(len(names), len(ev.box.ap50)))}

        self.eval_results = {
            "map50": round(float(ev.box.map50), 4),
            "map50_95": round(float(ev.box.map), 4),
            "precision": round(float(ev.box.mp), 4),
            "recall": round(float(ev.box.mr), 4),
            "per_class_ap50": {k: round(v, 4) for k, v in per_class.items()},
        }

        s1 = self.config["sprint1_results"]["per_class_ap50"]
        print(f"\n  mAP@0.5: {self.eval_results['map50']:.4f}")
        for n, ap in per_class.items():
            s1v = s1.get(n)
            d = f"({'+' if ap>s1v else ''}{ap-s1v:.3f})" if s1v else "(NEW)"
            print(f"    {n:<25} {ap:.4f} {d}")

        if self.clearml_ok and self.task:
            lg = self.task.get_logger()
            lg.report_scalar("evaluation", "mAP@0.5", 0, self.eval_results["map50"])
            for n, ap in per_class.items():
                lg.report_scalar("per_class_AP50", n, 0, ap)
        self.run_log["stages_completed"].append("model_evaluation")
        print("\n  Stage 6 PASSED")

    # ── STAGE 7: MODEL VALIDATION (GATE) ──────────────────────────
    def stage_7_model_validation(self):
        print("\n" + "=" * 60)
        print(f"  STAGE 7: APPROVAL GATE (Experiment {self.experiment})")
        print("=" * 60)
        m = self.eval_results["map50"]
        f_ap = self.eval_results["per_class_ap50"].get("fracture", 0)
        g_m = m > self.config["min_map50"]
        g_f = f_ap >= self.config["min_fracture_ap"]
        self.gate_passed = g_m and g_f
        print(f"  mAP:     {m:.4f} (>{self.config['min_map50']})  {'PASS' if g_m else 'FAIL'}")
        print(f"  Frac AP: {f_ap:.4f} (>={self.config['min_fracture_ap']}) {'PASS' if g_f else 'FAIL'}")
        print(f"\n  {'APPROVED' if self.gate_passed else 'NOT APPROVED'}")
        if self.clearml_ok and self.task:
            self.task.add_tags(["gate-passed"] if self.gate_passed else ["gate-failed"])
        self.run_log["stages_completed"].append("model_validation")
        self.run_log["gate_passed"] = self.gate_passed
        print("\n  Stage 7 COMPLETE")

    # ── STAGE 8: ARTIFACT EXPORT ──────────────────────────────────
    def stage_8_artifact_export(self):
        print("\n" + "=" * 60)
        print("  STAGE 8: ARTIFACT EXPORT")
        print("=" * 60)
        (self.output_dir/"models").mkdir(parents=True, exist_ok=True)
        (self.output_dir/"figures").mkdir(exist_ok=True)

        shutil.copy2(self.best_pt, self.output_dir/"models"/"best.pt")
        train_dir = Path(self.best_pt).parent.parent
        for fig in ["results.png","confusion_matrix.png","confusion_matrix_normalized.png",
                     "F1_curve.png","P_curve.png","R_curve.png","PR_curve.png"]:
            fp = train_dir/fig
            if fp.exists(): shutil.copy2(fp, self.output_dir/"figures"/fig)
        rc = train_dir/"results.csv"
        if rc.exists(): shutil.copy2(rc, self.output_dir/"results.csv")

        summary = {
            "sprint": 2, "experiment": self.experiment,
            "experiment_name": self.config["experiment_name"],
            "experiment_owner": self.config["experiment_owner"],
            "hypothesis": self.config["hypothesis"],
            "completed_at": datetime.now().isoformat(),
            "training_minutes": round(self.train_duration, 1),
            "gate_passed": self.gate_passed,
            **self.eval_results,
            "config": {k: self.config[k] for k in
                       ["imgsz","batch","epochs","patience","mixup","copy_paste","cls","box"]},
            "sprint1_baseline": self.config["sprint1_results"],
        }
        with open(self.output_dir/"eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.run_log["completed_at"] = datetime.now().isoformat()
        with open(self.output_dir/"pipeline_run.json", "w") as f:
            json.dump(self.run_log, f, indent=2)

        if self.clearml_ok and self.task:
            self.task.upload_artifact("best_model", str(self.output_dir/"models"/"best.pt"))
        print("  Artifacts exported")
        self.run_log["stages_completed"].append("artifact_export")
        print("\n  Stage 8 PASSED")

    # ── RUN ────────────────────────────────────────────────────────
    def run(self, dry_run=False):
        print("\n" + "=" * 60)
        print(f"  WRISTASSIST AI — SPRINT 2 ML PIPELINE")
        print(f"  Experiment {self.experiment}: {self.config['experiment_name']}")
        print(f"  Owner: {self.config['experiment_owner']}")
        print("=" * 60)

        t_start = time.time()
        try:
            self.stage_1_environment_check()
            self.stage_2_data_extraction()
            self.stage_3_data_validation()
            self.stage_4_data_preparation()
            if dry_run:
                print("\n  DRY RUN COMPLETE"); return self.run_log
            self.stage_5_model_training()
            self.stage_6_model_evaluation()
            self.stage_7_model_validation()
            self.stage_8_artifact_export()
        except Exception as e:
            self.run_log["stages_failed"].append({"error": str(e)})
            print(f"\n  PIPELINE FAILED: {e}"); raise
        finally:
            self.run_log["total_minutes"] = round((time.time()-t_start)/60, 1)
            if self.clearml_ok and self.task: self.task.close()

        print("\n" + "=" * 60)
        print(f"  EXPERIMENT {self.experiment} COMPLETE")
        print(f"  mAP: {self.eval_results['map50']:.4f} | "
              f"Gate: {'PASS' if self.gate_passed else 'FAIL'} | "
              f"Time: {self.run_log['total_minutes']:.0f}min")
        print("=" * 60)
        return self.run_log


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT COMPARISON
# ══════════════════════════════════════════════════════════════════════
def compare_experiments(summary_paths):
    exps = []
    for p in summary_paths:
        with open(p) as f: exps.append(json.load(f))
    if not exps: print("No experiments."); return

    print("=" * 80)
    print("  EXPERIMENT COMPARISON")
    print("=" * 80)
    labels = [f"Exp {e['experiment']}" for e in exps]
    print(f"{'Metric':<28}" + "".join(f"{l:>14}" for l in labels) + f"{'Sprint 1':>14}")
    print("-" * 80)
    print(f"{'mAP@0.5':<28}" + "".join(f"{e['map50']:>14.4f}" for e in exps) + f"{0.685:>14.4f}")

    s1 = exps[0].get("sprint1_baseline",{}).get("per_class_ap50",{})
    for cls in ["fracture","metal_implant","periosteal_reaction","pronator_sign","text"]:
        vals = "".join(f"{e['per_class_ap50'].get(cls,0):>14.4f}" for e in exps)
        s1v = f"{s1[cls]:>14.4f}" if cls in s1 else f"{'N/A':>14}"
        print(f"  {cls:<26}" + vals + s1v)

    print("-" * 80)
    print(f"{'Gate':<28}" + "".join(f"{'PASS' if e['gate_passed'] else 'FAIL':>14}" for e in exps))
    print(f"{'Config':<28}")
    for k in ["imgsz","batch","epochs","mixup","copy_paste","cls"]:
        print(f"  {k:<26}" + "".join(f"{str(e['config'].get(k,'')):>14}" for e in exps))

    passed = [e for e in exps if e["gate_passed"]]
    if passed:
        best = max(passed, key=lambda e: e["map50"])
        print(f"\n  RECOMMENDED: Experiment {best['experiment']} "
              f"({best['experiment_name']}) — mAP={best['map50']:.4f}")
    else:
        print("\n  No experiment passed the gate.")
    print("=" * 80)
    return exps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", choices=["A","B","C"], default="A")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare", nargs="+")
    args = parser.parse_args()
    if args.compare:
        compare_experiments(args.compare)
    else:
        Pipeline(experiment=args.experiment).run(dry_run=args.dry_run)
