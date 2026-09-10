"""OOF evaluation for rotating-test cross-validation runs."""

import csv
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

from metrics import regression_metrics, select_cv_thresholds
from tasks import adaptation_mode, is_regression
from train import load_metadata


def _payloads(rows: pd.DataFrame) -> list[dict]:
    payloads = []
    for path in rows["artifact"]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        payload["artifact"] = str(path)
        payloads.append(payload)
    return payloads


CALIBRATION_INCLUSIVE_NOTE = (
    "The shared threshold uses all rotating validation partitions. Every subject is "
    "also tested in another rotation, so threshold-dependent OOF metrics are "
    "calibration-inclusive; pooled ROC-AUC and average precision are threshold-free."
)


def _shared_threshold_selection(
    manifest: pd.DataFrame, n_splits: int, threshold_config: dict, expected: set[str]
) -> tuple[dict, int]:
    """Choose one cut from every fold's validation predictions, never test rows."""
    strategy = threshold_config.get("strategy", "cv_common_threshold")
    if strategy != "cv_common_threshold":
        raise ValueError(
            "Rotating-test evaluation requires threshold.strategy="
            f"'cv_common_threshold', got {strategy!r}"
        )

    fold_predictions = {}
    validation_subjects = []
    for fold in range(1, n_splits + 1):
        rows = manifest[
            (manifest["fold"] == fold) & (manifest["split"] == "validation")
        ]
        payloads = _payloads(rows)
        if not payloads:
            raise ValueError(f"Fold {fold} has no validation artifacts")
        fold_predictions[fold] = (
            np.asarray([payload["target"] for payload in payloads], dtype=int),
            np.asarray(
                [payload["image_probability"] for payload in payloads], dtype=float
            ),
        )
        validation_subjects.extend(payload["subject_id"] for payload in payloads)

    if len(validation_subjects) != len(set(validation_subjects)):
        raise ValueError("Validation artifacts contain duplicate subjects")
    if set(validation_subjects) != expected:
        raise ValueError(
            "Validation artifacts do not cover every manifest subject exactly once"
        )

    selection = select_cv_thresholds(
        strategy="cv_common_threshold",
        fold_predictions=fold_predictions,
        objective=threshold_config.get("objective", "balanced_accuracy"),
        num_thresholds=int(threshold_config.get("num_thresholds", 1000)),
        tie_break=threshold_config.get("tie_break", "plateau_midpoint"),
    )
    return selection, len(validation_subjects)


def _save_threshold_selection(
    destination: Path, selection: dict, validation_subjects: int
) -> dict:
    """Persist the shared cut and its full validation-only selection curve."""
    curve = selection.get("curve", [])
    selection_path = destination / "threshold_selection.json"
    curve_path = destination / "threshold_curve.csv"
    record = {key: value for key, value in selection.items() if key != "curve"}
    record.update(
        {
            "threshold": float(selection["shared_threshold"]),
            "threshold_source": "all_validation_folds",
            "validation_folds": sorted(selection["fold_thresholds"]),
            "validation_subjects": int(validation_subjects),
            "thresholded_metrics_calibration_inclusive": True,
            "calibration_note": CALIBRATION_INCLUSIVE_NOTE,
            "artifacts": {
                "threshold_selection": str(selection_path),
                "threshold_curve": str(curve_path),
            },
        }
    )
    with selection_path.open("w") as handle:
        json.dump(record, handle, indent=2)
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    return record


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
    threshold_selection = None
    validation_subject_count = None
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
        threshold_selection, validation_subject_count = _shared_threshold_selection(
            manifest,
            cv["n_splits"],
            config.get("threshold", {}),
            expected,
        )
        threshold = float(threshold_selection["shared_threshold"])
        for fold in range(1, cv["n_splits"] + 1):
            thresholds[fold] = {
                "threshold": threshold,
                "strategy": "cv_common_threshold",
                "threshold_source": "all_validation_folds",
            }
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
                        "threshold_source": "all_validation_folds",
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
    threshold_record = (
        _save_threshold_selection(
            destination, threshold_selection, validation_subject_count
        )
        if threshold_selection is not None
        else None
    )
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
    if threshold_record is not None:
        summary["threshold_selection"] = threshold_record
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
        if threshold_record is not None:
            run.summary["OOF Shared Threshold"] = threshold_record["threshold"]
            run.summary["OOF Threshold Calibration Inclusive"] = True
            run.summary["OOF Threshold Selection"] = str(
                destination / "threshold_selection.json"
            )
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
        row = {
            key: value
            for key, value in summary.items()
            if key not in {"metrics", "thresholds", "threshold_selection"}
        }
        if "threshold_selection" in summary:
            row["threshold"] = summary["threshold_selection"]["threshold"]
            row["thresholded_metrics_calibration_inclusive"] = True
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
