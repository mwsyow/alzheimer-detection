"""Generate a refit configuration from a completed cross-validation run."""

import argparse
import copy
import json
import math
import re
import statistics
from pathlib import Path

import torch

from metrics import (
    DEFAULT_NUM_THRESHOLDS,
    DEFAULT_THRESHOLD_OBJECTIVE,
    DEFAULT_THRESHOLD_TIE_BREAK,
)
from train import CV_BEST_FILENAME, load_metadata, normalize_threshold_config

EPOCH_RULES = ("median", "mean")


def round_half_up(value: float) -> int:
    """Round a non-negative epoch statistic to the nearest integer."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"epoch statistic must be finite and non-negative, got {value}"
        )
    return math.floor(value + 0.5)


def recommend_refit_epochs(best_epoch_indices: list[int], rule: str = "median") -> dict:
    """Turn zero-based fold best epochs into a refit training count."""
    if rule not in EPOCH_RULES:
        raise ValueError(
            f"unsupported epoch rule {rule!r}; expected one of {EPOCH_RULES}"
        )
    if not best_epoch_indices:
        raise ValueError("cannot recommend refit epochs without fold best epochs")
    if any(
        not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0
        for epoch in best_epoch_indices
    ):
        raise ValueError("fold best epochs must be non-negative integers")

    fold_counts = [epoch + 1 for epoch in best_epoch_indices]
    mean_count = statistics.fmean(fold_counts)
    median_count = statistics.median(fold_counts)
    selected = median_count if rule == "median" else mean_count
    return {
        "rule": rule,
        "best_epoch_indices": list(best_epoch_indices),
        "fold_epoch_counts": fold_counts,
        "mean_epoch_count": mean_count,
        "median_epoch_count": median_count,
        "epochs": round_half_up(selected),
    }


def _fold_result(cv_state: dict, fold_number: int):
    results = cv_state.get("fold_results", {})
    return results.get(fold_number) or results.get(str(fold_number))


def load_fold_best_epochs(run_dir: Path, metadata: dict) -> list[int]:
    """Read every fold's best epoch, falling back to its selected checkpoint."""
    cv_state = metadata.get("cv")
    if not cv_state:
        raise ValueError(f"{run_dir} is not a cross-validation run")
    folds = cv_state.get("folds", [])
    if not folds:
        raise ValueError("CV metadata contains no folds")

    cv_budget = metadata.get("config", {}).get("epochs")
    if not isinstance(cv_budget, int) or isinstance(cv_budget, bool) or cv_budget < 1:
        raise ValueError("CV metadata config must contain a positive integer epochs")

    expected = list(range(1, len(folds) + 1))
    completed_raw = cv_state.get("completed_folds")
    completed = (
        None if completed_raw is None else sorted(int(fold) for fold in completed_raw)
    )
    if completed is not None and completed != expected:
        raise ValueError(
            f"CV run is incomplete: completed folds {completed}, expected {expected}"
        )

    epochs = []
    missing = []
    for fold_number in expected:
        result = _fold_result(cv_state, fold_number)
        epoch = result.get("epoch") if isinstance(result, dict) else None
        if epoch is None:
            checkpoint_path = run_dir / f"split_{fold_number}" / CV_BEST_FILENAME
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            epoch = checkpoint.get("epoch")
            del checkpoint
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"fold {fold_number} has invalid best epoch {epoch!r}")
        if epoch >= cv_budget:
            raise ValueError(
                f"fold {fold_number} best epoch {epoch} is outside the configured "
                f"CV budget of {cv_budget} epochs"
            )
        epochs.append(epoch)

    if missing:
        listed = "\n  ".join(missing)
        raise ValueError(f"No best epoch is available for these folds:\n  {listed}")
    if len(epochs) != len(folds):
        raise ValueError("a best epoch is required for every CV fold")
    return epochs


def canonical_threshold_config() -> dict:
    return {
        "strategy": "cv_common_threshold",
        "objective": DEFAULT_THRESHOLD_OBJECTIVE,
        "num_thresholds": DEFAULT_NUM_THRESHOLDS,
        "tie_break": DEFAULT_THRESHOLD_TIE_BREAK,
    }


def refit_wandb_name(base_name: str | None, run_id: str) -> str:
    if not base_name:
        return f"refit-{run_id}"
    return base_name if base_name.endswith("-refit") else f"{base_name}-refit"


def safe_filename_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return stem or "refit"


def default_output_path(metadata: dict) -> Path:
    config = metadata["config"]
    base_name = config.get("wandb_name") or metadata["run_id"]
    return Path("configs") / f"{safe_filename_stem(base_name)}_refit.json"


def build_refit_config(metadata: dict, epochs: int) -> dict:
    """Copy the resolved CV recipe and change only refit/evaluation policy."""
    if "config" not in metadata or "run_id" not in metadata:
        raise ValueError("metadata must contain config and run_id")
    config = copy.deepcopy(metadata["config"])
    config["epochs"] = epochs
    config.setdefault("cv", {})["enabled"] = False
    config["refit"] = {"enabled": True}
    config.setdefault("early_stopping", {})["enabled"] = False
    config["wandb_name"] = refit_wandb_name(
        config.get("wandb_name"), str(metadata["run_id"])
    )
    config["threshold"] = canonical_threshold_config()
    return normalize_threshold_config(config)


def write_refit_config(config: dict, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a refit JSON from a completed CV run."
    )
    parser.add_argument(
        "--cv-run",
        type=Path,
        required=True,
        metavar="CV_RUN_DIR",
        help="Completed run directory, for example checkpoints/<cv_run_id>.",
    )
    parser.add_argument(
        "--epoch-rule",
        choices=EPOCH_RULES,
        default="median",
        help="Aggregate fold best epoch counts with median (default) or mean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to configs/<wandb_name>_refit.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = args.cv_run / "metadata.pth"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No CV metadata found at {metadata_path}")
    metadata = load_metadata(metadata_path)
    best_epochs = load_fold_best_epochs(args.cv_run, metadata)
    recommendation = recommend_refit_epochs(best_epochs, args.epoch_rule)
    config = build_refit_config(metadata, recommendation["epochs"])
    output_path = args.output or default_output_path(metadata)

    cv_budget = metadata["config"]["epochs"]
    boundary_folds = [
        fold
        for fold, epoch in enumerate(best_epochs, start=1)
        if epoch == cv_budget - 1
    ]
    if boundary_folds:
        print(
            "WARNING: best checkpoint is at the final allowed CV epoch for fold(s) "
            f"{boundary_folds}; the recommended refit budget may be censored."
        )

    write_refit_config(config, output_path, force=args.force)
    print(f"CV run: {metadata['run_id']}")
    print(f"Fold best epoch indices: {recommendation['best_epoch_indices']}")
    print(f"Fold epoch counts: {recommendation['fold_epoch_counts']}")
    print(f"Mean epoch count: {recommendation['mean_epoch_count']:.3f}")
    print(f"Median epoch count: {recommendation['median_epoch_count']:.3f}")
    print(
        f"Recommended refit epochs ({recommendation['rule']}): "
        f"{recommendation['epochs']}"
    )
    print(f"Saved refit config: {output_path}")
    print(f"Train: uv run python train.py --config {output_path}")
    print(
        "Evaluate: uv run python evaluate.py --checkpoint "
        "checkpoints/<refit_run_id>/last.pth "
        f"--threshold-from {args.cv_run}"
    )


if __name__ == "__main__":
    main()
