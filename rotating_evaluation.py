"""Leakage-safe OOF evaluation for rotating-test cross-validation runs."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from metrics import regression_metrics, threshold_tie_index
from tasks import adaptation_mode, is_regression
from train import load_metadata


def _payloads(rows: pd.DataFrame) -> list[dict]:
    payloads = []
    for path in rows["artifact"]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        payload["artifact"] = str(path)
        payloads.append(payload)
    return payloads


def _fold_threshold(payloads: list[dict], threshold_config: dict) -> tuple[float, dict]:
    y_true = np.asarray([p["target"] for p in payloads], dtype=int)
    y_prob = np.asarray([p["image_probability"] for p in payloads], dtype=float)
    objective = threshold_config.get("objective", "balanced_accuracy")
    count = int(threshold_config.get("num_thresholds", 1000))
    grid = np.linspace(0.0, 1.0, count)
    values = []
    for threshold in grid:
        pred = y_prob >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        scores = {
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
        if objective not in scores:
            raise ValueError(f"Unsupported threshold objective: {objective!r}")
        values.append(scores[objective])
    index = threshold_tie_index(
        values, grid, threshold_config.get("tie_break", "plateau_midpoint")
    )
    return float(grid[index]), {
        "objective": objective,
        "objective_value": float(values[index]),
        "selected_index": int(index),
        "num_thresholds": count,
        "tie_break": threshold_config.get("tie_break", "plateau_midpoint"),
    }


def _classification_metrics(y_true, y_prob, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def evaluate_rotating_run(
    run_dir: Path, output_dir: Path | None = None, log_wandb: bool = False
) -> dict:
    metadata = load_metadata(run_dir / "metadata.pth")
    cv = metadata.get("cv", {})
    if cv.get("strategy") != "rotating_test":
        raise ValueError(f"{run_dir} is not a rotating-test CV run")
    if len(cv.get("completed_folds", [])) != cv["n_splits"]:
        raise ValueError(f"{run_dir} is incomplete")

    config = metadata["config"]
    manifest_path = Path(cv["artifact_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = pd.read_csv(manifest_path)
    test_rows = manifest[manifest["split"] == "test"].copy()
    if test_rows["subject_id"].duplicated().any():
        raise ValueError("OOF test artifacts contain duplicate subjects")
    expected = set(cv["subject_folds"])
    if set(test_rows["subject_id"]) != expected:
        raise ValueError("OOF test artifacts do not cover every manifest subject once")

    thresholds = {}
    prediction_rows = []
    if is_regression(config):
        for payload in _payloads(test_rows):
            prediction_rows.append(
                {
                    "subject_id": payload["subject_id"],
                    "true_age": float(payload["target"]),
                    "predicted_age": float(payload["age_prediction_years"]),
                    "residual": float(payload["age_prediction_years"] - payload["target"]),
                    "fold": int(payload["fold"]),
                    "seed": int(payload["seed"]),
                    "cv_random_seed": payload["cv_random_seed"],
                    "best_epoch": int(payload["best_epoch"]),
                    "checkpoint": payload["checkpoint"],
                    "architecture": payload["architecture"],
                    "artifact": payload["artifact"],
                }
            )
        frame = pd.DataFrame(prediction_rows).sort_values("subject_id")
        metrics = regression_metrics(frame["true_age"], frame["predicted_age"])
    else:
        for fold in range(1, cv["n_splits"] + 1):
            val_rows = manifest[
                (manifest["fold"] == fold) & (manifest["split"] == "validation")
            ]
            threshold, provenance = _fold_threshold(
                _payloads(val_rows), config.get("threshold", {})
            )
            thresholds[fold] = {"threshold": threshold, **provenance}
            fold_test = test_rows[test_rows["fold"] == fold]
            for payload in _payloads(fold_test):
                probability = float(payload["image_probability"])
                prediction_rows.append(
                    {
                        "subject_id": payload["subject_id"],
                        "true_label": int(payload["target"]),
                        "image_logit": float(payload["image_logit"]),
                        "probability": probability,
                        "prediction": int(probability >= threshold),
                        "threshold": threshold,
                        "fold": fold,
                        "seed": int(payload["seed"]),
                        "cv_random_seed": payload["cv_random_seed"],
                        "best_epoch": int(payload["best_epoch"]),
                        "checkpoint": payload["checkpoint"],
                        "architecture": payload["architecture"],
                        "artifact": payload["artifact"],
                    }
                )
        frame = pd.DataFrame(prediction_rows).sort_values("subject_id")
        metrics = _classification_metrics(
            frame["true_label"].to_numpy(),
            frame["probability"].to_numpy(),
            frame["prediction"].to_numpy(),
        )

    output_dir = output_dir or Path(config.get("evaluation", {}).get("output_dir", "evaluations"))
    destination = output_dir / metadata["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "oof_predictions.csv", index=False)
    summary = {
        "run_id": metadata["run_id"],
        "task": config.get("task", {}).get("name", "ad_classification"),
        "architecture": config["model"]["name"],
        "adaptation_mode": adaptation_mode(config),
        "config_sha256": str(manifest["config_sha256"].iloc[0]),
        "comparison_sha256": str(manifest["comparison_sha256"].iloc[0]),
        "seed": int(config.get("seed", 0)),
        "cv_random_seed": config["cv"].get("random_seed"),
        "folds": cv["n_splits"],
        "best_epochs": [
            cv["fold_results"][fold]["epoch"] for fold in sorted(cv["fold_results"])
        ],
        "thresholds": thresholds,
        "metrics": metrics,
    }
    with (destination / "oof_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved rotating OOF evaluation to {destination}")
    if log_wandb or config.get("evaluation", {}).get("log_wandb", False):
        import wandb

        run = wandb.init(
            entity=config.get("wandb_entity"),
            project=config["wandb_project"],
            id=metadata["run_id"],
            resume="allow",
            mode=config.get("wandb_mode", "online"),
            reinit="create_new",
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                run.summary[f"OOF Test {key}"] = value
        run.summary["OOF Predictions"] = str(destination / "oof_predictions.csv")
        run.finish()
    return summary


def sweep_run_dirs(sweep_id: str, checkpoint_root: Path) -> list[Path]:
    import wandb

    runs = wandb.Api().sweep(sweep_id).runs
    paths = [checkpoint_root / run.id for run in runs]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Local checkpoints are missing for sweep runs: "
            + ", ".join(path.name for path in missing)
        )
    return paths


def compare_rotating_runs(run_dirs: list[Path], output_dir: Path) -> dict:
    summaries = [evaluate_rotating_run(path, output_dir=output_dir) for path in run_dirs]
    tasks = {summary["task"] for summary in summaries}
    if len(tasks) != 1:
        raise ValueError(f"Cannot aggregate mixed tasks: {sorted(tasks)}")
    rows = []
    for summary in summaries:
        row = {key: value for key, value in summary.items() if key not in {"metrics", "thresholds"}}
        row["best_epochs"] = json.dumps(row["best_epochs"])
        row.update(summary["metrics"])
        row.pop("confusion_matrix", None)
        rows.append(row)
    frame = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "cv_random_seed", "folds", "n"}
    ]
    groups = ["task", "architecture", "adaptation_mode", "comparison_sha256"]
    aggregate = frame.groupby(groups, dropna=False)[metric_columns].agg(["mean", "std"])
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", "-".join(path.name for path in run_dirs[:3]))
    destination = output_dir / "comparisons" / slug
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "runs.csv", index=False)
    aggregate.to_csv(destination / "aggregate.csv")
    return {"runs": frame, "aggregate": aggregate, "output_dir": destination}
