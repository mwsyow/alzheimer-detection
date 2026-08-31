import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import build_dataset_source
from metrics import (
    CV_THRESHOLD_STRATEGIES,
    DEFAULT_FPR_ROUNDING,
    DEFAULT_NUM_THRESHOLDS,
    DEFAULT_THRESHOLD_OBJECTIVE,
    DEFAULT_THRESHOLD_TIE_BREAK,
    FPR_ROUNDING_POLICIES,
    WANDB_METRIC_LABELS,
    aggregate_fold_metrics,
    collect_predictions,
    compute_metrics,
    save_confusion_matrix_plot,
    select_cv_thresholds,
    unpack_predictions,
)
from models import build_model
from train import (
    CV_BEST_FILENAME,
    build_loss,
    deep_update,
    find_metadata_path,
    load_config,
    load_metadata,
    normalize_threshold_config,
)

# Evaluation may override only settings that cannot change what the model computes.
# Everything else describes how the model was built and how its inputs were preprocessed,
# so it has to match the checkpoint: swapping transforms or model params at evaluation
# time would silently invalidate every number reported, which is exactly how a whole
# sweep was once spent training from scratch while its config claimed otherwise.
EVAL_OVERRIDABLE_KEYS = frozenset(
    {
        "threshold",
        "evaluation",
        "device",
        "wandb_entity",
        "wandb_project",
        "wandb_mode",
        "wandb_name",
        # Safe because evaluation runs under model.eval(): BatchNorm uses its running
        # statistics, so batch size cannot move a prediction.
        "dataloader",
    }
)

# Bookkeeping that aggregate_fold_metrics averages because it happens to be numeric.
# "Test Mean fold" is 3.0 for any 5-fold run, and the operating_fpr pair records how the
# shared cut was chosen rather than how the model scored. All three stay in summary.json
# and the per-fold files; none of them belongs in a run's headline summary.
NON_RESULT_METRICS = frozenset(
    {"fold", "operating_fpr_target", "operating_fpr_realised_on_val"}
)

# Rows of the comparison table, in reporting order. Ranking metrics first: they are the
# only ones that need no operating point.
COMPARISON_METRICS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "npv",
    "fpr",
    "loss",
    "threshold",
)
COMPARISON_LABELS = WANDB_METRIC_LABELS
COMPARISON_COLUMNS = ("Metric", "CV Validation", "Test Ensemble", "Test Mean")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def load_checkpoint_and_metadata(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_metadata(find_metadata_path(checkpoint_path))
    return checkpoint, metadata


def resolve_test_idx(metadata: dict) -> list[int]:
    """The held-out test set, which CV shares across every fold."""
    if "cv" in metadata:
        return metadata["cv"]["test_idx"]
    return metadata["split"]["test_idx"]


def detect_fold(checkpoint: dict, checkpoint_path: Path):
    """Which CV fold produced a checkpoint.

    Prefer the value stored in the checkpoint; fall back to the split_<k> directory
    name so checkpoints written before the fold field still resolve.
    """
    fold = checkpoint.get("fold")
    if fold is not None:
        return fold
    match = re.fullmatch(r"split_(\d+)", checkpoint_path.parent.name)
    return int(match.group(1)) if match else None


def resolve_eval_config(metadata: dict, config_path: Path = None) -> tuple[dict, dict]:
    """The run's own config, with evaluation-time settings layered over it.

    Returns (config, report). Only EVAL_OVERRIDABLE_KEYS are honoured; the rest are
    properties of the trained run. They are ignored rather than rejected because the
    expected usage is passing the very config that trained the run -- but any whose value
    differs from the run's own is named, since a silent mismatch there is the one that
    invalidates results.
    """
    config = deep_update({}, metadata["config"])
    report = {"config_path": None if config_path is None else str(config_path),
              "overridden": [], "ignored_differing": []}
    if config_path is None:
        return config, report

    supplied = load_config(config_path, normalize_threshold=False)
    overrides = {}
    for key, value in supplied.items():
        if key in EVAL_OVERRIDABLE_KEYS:
            if config.get(key) != value:
                report["overridden"].append(key)
            overrides[key] = value
        elif config.get(key) != value:
            report["ignored_differing"].append(key)

    if report["ignored_differing"]:
        print(
            f"Using the run's own values for {', '.join(sorted(report['ignored_differing']))} "
            f"-- {config_path} differs but these describe the trained model, not its "
            "evaluation."
        )
    return deep_update(config, overrides), report


def discover_fold_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Map fold number -> best_model.pth for every fold present in a run directory."""
    found = {}
    for path in sorted(run_dir.glob(f"split_*/{CV_BEST_FILENAME}")):
        match = re.fullmatch(r"split_(\d+)", path.parent.name)
        if match:
            found[int(match.group(1))] = path
    return found


def stored_threshold(checkpoint: dict, checkpoint_path=None) -> float:
    """The operating point a checkpoint recorded, or a clear failure.

    Checkpoints written while training still tuned a threshold each epoch carry one.
    Newer ones do not, because training now computes only threshold-free metrics and
    the cut is chosen once at evaluation time. Falling back to 0.5 here reported an
    arbitrary operating point as though it had been validated.
    """
    threshold = checkpoint.get("threshold")
    if threshold is None:
        where = f" in {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(
            f"No threshold stored{where}, because training no longer selects one. "
            "Choose an operating point explicitly: --threshold <float> to pin one, "
            "--threshold-from <cv_run_dir> to inherit the one that run's ensemble was "
            "scored at, or --threshold-strategy vertical_average to select one from a "
            "cross-validation run's own folds."
        )
    return float(threshold)


def evaluate_checkpoint(
    checkpoint: dict,
    config: dict,
    test_loader,
    device: torch.device,
    threshold_override: float = None,
    checkpoint_path=None,
):
    """Run one model over the test set and score it at a given or stored threshold."""
    model = build_model(config, initialize_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loss = build_loss(config)

    threshold = (
        threshold_override
        if threshold_override is not None
        else stored_threshold(checkpoint, checkpoint_path)
    )

    results = collect_predictions(model, test_loader, loss, device)
    metrics = compute_metrics(
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        threshold=threshold,
        loss=results["loss"],
    )
    # Historical field name kept so old and new test_metrics.json stay comparable.
    metrics["test_loss"] = metrics.pop("loss")
    return metrics, results, threshold


def build_predictions_frame(source, test_idx, results, threshold):
    return pd.DataFrame(
        {
            "image": [source.item_id(idx) for idx in test_idx],
            "label": results["y_true"],
            "pred": (results["y_prob"] >= threshold).astype(int),
            "prob_class_1": results["y_prob"],
            "threshold": threshold,
        }
    )


def save_outputs(
    output_dir: Path,
    metrics: dict,
    predictions: pd.DataFrame,
    save_predictions: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    if save_predictions:
        predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def log_to_wandb(metadata, config, metrics: dict, output_dir: Path):
    """Attach one checkpoint's test metrics to the run that produced it.

    output_dir is passed in rather than rebuilt from the config: the same checkpoint is
    evaluated at several operating points, each writing into its own subdirectory, so
    reconstructing the path here would find the wrong plot or none at all.
    """
    run = wandb.init(
        id=metadata["run_id"],
        resume="must",
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        run.summary[f"Test {key}"] = value

    run.log({"Test Confusion Matrix": wandb.Image(str(output_dir / "confusion_matrix.png"))})
    run.finish()


def resolve_evaluation_config(config: dict, args) -> dict:
    """Where outputs go and what gets written: CLI over config over defaults."""
    block = config.get("evaluation", {})
    return {
        "output_dir": Path(block.get("output_dir", "evaluations")),
        # store_true cannot express "off", so the flag can only turn logging on.
        "log_wandb": bool(args.log_wandb or block.get("log_wandb", False)),
        "save_predictions": bool(block.get("save_predictions", True)),
    }


def resolve_threshold_config(config: dict, args) -> dict:
    """Threshold-selection settings: CLI over config over module defaults."""
    # Evaluation starts from stored metadata. Metadata with no strategy at all keeps
    # the historical fallback rather than silently changing old results.
    block = normalize_threshold_config(
        config, legacy_missing_strategy=True
    )["threshold"]
    return {
        "strategy": args.threshold_strategy or block["strategy"],
        "fpr_rounding": (
            args.fpr_rounding or block.get("fpr_rounding", DEFAULT_FPR_ROUNDING)
        ),
        "fpr_grid": block.get("fpr_grid", 101),
        "threshold_grid": block.get("threshold_grid", 0),
        "objective": block.get("objective", DEFAULT_THRESHOLD_OBJECTIVE),
        "num_thresholds": block.get("num_thresholds", DEFAULT_NUM_THRESHOLDS),
        "tie_break": block.get("tie_break", DEFAULT_THRESHOLD_TIE_BREAK),
        "value": block.get("value"),
    }


def require_complete_predictions(fold_paths: dict, fold_predictions: dict) -> None:
    """Refuse to average over folds when any fold's predictions are absent.

    Every offender is listed rather than only the first -- a resumed run can mix folds
    from before and after predictions were stored, so failing on one would hide the rest.
    """
    missing = [str(path) for fold, path in sorted(fold_paths.items()) if fold not in fold_predictions]
    if not missing:
        return
    listed = "\n  ".join(missing)
    raise ValueError(
        "These checkpoints store no validation predictions, so a shared operating "
        f"point cannot be chosen from them:\n  {listed}\n"
        "They were written before predictions were stored. Either retrain, or "
        "evaluate with --threshold-strategy per_fold_youden to use each fold's own "
        "stored threshold, or pin one with --threshold."
    )


def load_fold_validation_predictions(
    fold_paths: dict, strict: bool = True
) -> tuple[dict, dict]:
    """Every fold's stored validation predictions and its own tuned threshold.

    Deliberately two-pass: this reads each checkpoint, keeps only the small prediction
    tensors, and drops the reference before the caller reloads the weights it needs. One
    extra read per fold buys not holding five DenseNet121 state dicts in memory at once.

    Under strict it raises when any checkpoint predates prediction storage, which is what
    the averaging strategies need since they are not defined without every fold. Lenient
    mode returns whatever is present, for the reporting path where a missing fold falls
    back to its recorded metrics rather than invalidating the run.
    """
    predictions, stored_thresholds = {}, {}
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        packed = checkpoint.get("val_predictions")
        if packed is not None:
            predictions[fold_number] = unpack_predictions(packed)
        # None for a fold trained after threshold selection left training. Reported
        # rather than defaulted, so per_fold_youden can fail on exactly those folds.
        stored_thresholds[fold_number] = checkpoint.get("threshold")
        del checkpoint

    if strict:
        require_complete_predictions(fold_paths, predictions)
    return predictions, stored_thresholds


def select_fold_thresholds(
    fold_paths: dict, config: dict, args, fold_predictions: dict = None
) -> dict:
    """One operating point per fold, plus a record of how it was chosen.

    fold_predictions may be passed in when the caller has already read the checkpoints,
    which saves re-reading five state dicts to get at five small arrays.
    """
    if args.threshold is not None:
        return {
            "strategy": "cli_override",
            "shared_threshold": args.threshold,
            "fold_thresholds": {fold: args.threshold for fold in fold_paths},
            "skipped_folds": [],
        }

    if getattr(args, "threshold_from", None) is not None:
        threshold, record = inherited_operating_point(args.threshold_from, args)
        return {
            **record,
            "shared_threshold": threshold,
            "fold_thresholds": {fold: threshold for fold in fold_paths},
            "skipped_folds": [],
        }

    threshold_config = resolve_threshold_config(config, args)
    strategy = threshold_config["strategy"]

    if strategy == "per_fold_youden":
        # Reads the thresholds already in the checkpoints, so runs trained before
        # predictions were stored stay evaluable exactly as they were.
        stored = load_stored_thresholds(fold_paths)
        return select_cv_thresholds(strategy=strategy, fold_stored_thresholds=stored)

    if strategy == "fixed":
        return select_cv_thresholds(
            strategy=strategy,
            fold_stored_thresholds={fold: None for fold in fold_paths},
            fixed_value=threshold_config["value"],
        )

    if fold_predictions is None:
        fold_predictions, _ = load_fold_validation_predictions(fold_paths)
    else:
        require_complete_predictions(fold_paths, fold_predictions)
    return select_cv_thresholds(
        strategy=strategy,
        fold_predictions=fold_predictions,
        fpr_rounding=threshold_config["fpr_rounding"],
        fpr_grid=threshold_config["fpr_grid"],
        threshold_grid=threshold_config["threshold_grid"],
        objective=threshold_config["objective"],
        num_thresholds=threshold_config["num_thresholds"],
        tie_break=threshold_config["tie_break"],
        fixed_value=threshold_config["value"],
    )


def load_stored_thresholds(fold_paths: dict) -> dict:
    """Just the tuned threshold each fold wrote, with no requirement on predictions.

    Every offender is listed rather than only the first: a resumed run can mix folds
    trained before and after threshold selection left training, so failing on fold 1
    would hide that folds 3 and 5 are in the same state.
    """
    stored, missing = {}, []
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("threshold") is None:
            missing.append(str(path))
        else:
            stored[fold_number] = float(checkpoint["threshold"])
        del checkpoint

    if missing:
        listed = "\n  ".join(missing)
        raise ValueError(
            "per_fold_youden needs a threshold stored by each fold, and these store "
            f"none:\n  {listed}\n"
            "They were trained after threshold selection left the training loop. Use "
            "--threshold-strategy vertical_average to choose one cut from the folds' "
            "stored predictions, or pin one with --threshold."
        )
    return stored


def threshold_definition(config: dict) -> dict:
    """Semantic threshold settings, independent of legacy key spelling."""
    block = normalize_threshold_config(
        config, legacy_missing_strategy=True
    )["threshold"]
    strategy = block["strategy"]
    definition = {"strategy": strategy}
    if strategy == "cv_common_threshold":
        for key in ("objective", "num_thresholds", "tie_break"):
            definition[key] = block[key]
    elif strategy == "fixed":
        definition["value"] = block["value"]
    elif strategy == "vertical_average":
        definition["fpr_rounding"] = block.get(
            "fpr_rounding", DEFAULT_FPR_ROUNDING
        )
        definition["fpr_grid"] = block.get("fpr_grid", 101)
    elif strategy == "threshold_average":
        definition["threshold_grid"] = block.get("threshold_grid", 0)
    return definition


def validate_donor_refit_compatibility(
    donor_metadata: dict, refit_metadata: dict
) -> None:
    """Reject a donor that is not the same training recipe and data partition."""
    donor_config = donor_metadata["config"]
    refit_config = refit_metadata["config"]
    mismatches = []
    for key in ("dataset", "model", "transforms", "loss", "optimizer", "split"):
        if donor_config.get(key) != refit_config.get(key):
            mismatches.append(key)
    if threshold_definition(donor_config) != threshold_definition(refit_config):
        mismatches.append("threshold")
    if mismatches:
        raise ValueError(
            "Threshold donor and refit are incompatible in: "
            + ", ".join(mismatches)
        )

    donor_cv = donor_metadata.get("cv")
    refit_split = refit_metadata.get("split")
    if not donor_cv or not refit_split:
        raise ValueError(
            "--threshold-from needs CV donor metadata and refit split metadata"
        )
    if list(donor_cv["test_idx"]) != list(refit_split["test_idx"]):
        raise ValueError("Threshold donor and refit test indices are not identical")
    donor_development = sorted(
        {index for fold in donor_cv["folds"] for index in fold["val_idx"]}
    )
    if donor_development != sorted(refit_split["train_idx"]):
        raise ValueError(
            "Threshold donor CV development pool and refit training pool differ"
        )


def _stored_cv_selection(source_run_dir: Path, donor_metadata: dict):
    selection = donor_metadata.get("cv", {}).get("threshold_selection")
    if selection is not None:
        return selection
    path = source_run_dir / "threshold_selection.json"
    if path.exists():
        with path.open() as handle:
            return json.load(handle)
    return None


def inherited_operating_point(
    source_run_dir: Path, args, refit_metadata: dict = None
) -> tuple[float, dict]:
    """The exact cut another cross-validation run's ensemble was scored at.

    This is how a single refit model gets an operating point without holding anything
    back: it borrows the one the folds of the same architecture already agreed on. The
    threshold block comes from the donor run's own config rather than the model being
    evaluated, so the number reproduces that run's summary.json exactly and can be
    checked against it. CLI overrides still win, keeping the documented precedence.
    """

    donor_metadata = load_metadata(source_run_dir / "metadata.pth")
    fold_paths = discover_fold_checkpoints(source_run_dir)
    if "cv" not in donor_metadata:
        if not fold_paths:
            raise FileNotFoundError(
                f"No split_*/{CV_BEST_FILENAME} under {source_run_dir}. "
                "--threshold-from needs a cross-validation run directory."
            )
        raise ValueError("--threshold-from donor metadata is not a CV run")
    if refit_metadata is not None:
        validate_donor_refit_compatibility(donor_metadata, refit_metadata)

    stored_selection = _stored_cv_selection(source_run_dir, donor_metadata)
    if stored_selection is not None and stored_selection.get("threshold") is not None:
        threshold = float(stored_selection["threshold"])
        return threshold, {
            "strategy": "inherited_from_cv",
            "source_run_dir": str(source_run_dir),
            "source_run_id": donor_metadata["run_id"],
            "folds_used": sorted(fold_paths),
            "cv_selection": stored_selection,
            "threshold": threshold,
            "threshold_source": "stored CV selected threshold",
        }

    if not fold_paths:
        raise FileNotFoundError(
            f"No split_*/{CV_BEST_FILENAME} under {source_run_dir}. "
            "--threshold-from needs a cross-validation run directory."
        )

    donor_config = donor_metadata["config"]
    # An old donor without a stored selection is recomputed from its own definition,
    # never from evaluation-time overrides intended for the target model.
    donor_args = copy.copy(args)
    donor_args.threshold_from = None
    donor_args.threshold_strategy = None
    donor_args.fpr_rounding = None
    selection = select_fold_thresholds(fold_paths, donor_config, donor_args)
    threshold, source_phrase = ensemble_operating_point(
        selection, selection["fold_thresholds"]
    )
    return threshold, {
        "strategy": "inherited_from_cv",
        "source_run_dir": str(source_run_dir),
        "source_run_id": donor_metadata["run_id"],
        "folds_used": sorted(fold_paths),
        "cv_selection": selection,
        "threshold": threshold,
        "threshold_source": source_phrase,
    }


def resolve_single_threshold(
    checkpoint: dict,
    checkpoint_path: Path,
    args,
    metadata: dict = None,
    config: dict = None,
):
    """The operating point for one checkpoint, and a record of where it came from.

    Three ways, in precedence order: pinned on the command line, inherited from a
    cross-validation run's ensemble, or read back from the checkpoint itself for a run
    trained while training still tuned one.
    """
    if args.threshold is not None:
        return args.threshold, {
            "strategy": "cli_override",
            "threshold": args.threshold,
            "threshold_source": "pinned with --threshold",
        }
    if args.threshold_from is not None:
        return inherited_operating_point(
            args.threshold_from, args, refit_metadata=metadata
        )
    if config is not None:
        block = normalize_threshold_config(
            config, legacy_missing_strategy=True
        )["threshold"]
        if block["strategy"] == "fixed":
            threshold = float(block["value"])
            return threshold, {
                "strategy": "fixed",
                "threshold": threshold,
                "threshold_source": "threshold.value from the fixed strategy",
            }
    threshold = stored_threshold(checkpoint, checkpoint_path)
    return threshold, {
        "strategy": "checkpoint_stored",
        "threshold": threshold,
        "threshold_source": "the threshold this run tuned during training",
    }


def threshold_output_slug(record: dict) -> str:
    """A directory name naming where the cut came from.

    The same checkpoint is deliberately evaluated at more than one operating point --
    the inherited cut and a chosen one are two of the reported benchmarks -- so the
    outputs cannot share a path or the second run would overwrite the first.
    """
    strategy = record["strategy"]
    if strategy == "inherited_from_cv":
        return f"threshold_from_{record['source_run_id']}"
    if strategy == "checkpoint_stored":
        return "threshold_stored"
    return f"threshold_{record['threshold']:.3f}"


def describe_selection(selection: dict) -> str:
    """One line naming the operating point, for the console."""
    strategy = selection["strategy"]
    if strategy == "vertical_average":
        return (
            f"threshold selection: {strategy} -> target FPR "
            f"{selection['target_fpr']:.3f} (specificity "
            f"{selection['target_specificity']:.3f}), mean TPR "
            f"{selection['mean_tpr_at_target']:.3f}"
        )
    if selection["shared_threshold"] is not None:
        return f"threshold selection: {strategy} -> threshold {selection['shared_threshold']:.4f}"
    cuts = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(selection["fold_thresholds"].items()))
    return f"threshold selection: {strategy} -> per-fold {cuts}"


def ensemble_operating_point(
    selection: dict, fold_thresholds: dict
) -> tuple[float, str]:
    """The cut the ensemble inherits from the folds, and a phrase naming where it came from.

    A shared threshold when the strategy produced one; otherwise the mean of the per-fold
    cuts, which is what `vertical_average` leaves behind since it equalises false positive
    rate rather than probability.
    """
    shared = selection.get("shared_threshold")
    if shared is not None:
        return float(shared), f"{selection['strategy']} shared threshold"
    return (
        float(np.mean(list(fold_thresholds.values()))),
        f"mean of the per-fold {selection['strategy']} thresholds",
    )


def cv_validation_aggregate(
    metadata: dict, fold_thresholds: dict = None, fold_predictions: dict = None
) -> dict:
    """Mean +/- std of the folds' *validation* metrics, at the selected operating point.

    Recomputed from each fold's stored best-epoch predictions rather than read back from
    the metrics training recorded, so the validation column sits at the same cut as the
    test columns beside it. Read from the run's own files, never wandb, so evaluation
    stays usable offline.

    Still optimistic, and still here as the reference the test columns are read against
    rather than as a result: the cut was chosen on these same predictions.

    Folds trained before predictions were stored fall back to their recorded metrics,
    which sit at whatever threshold that epoch tuned for itself. A run mixing the two is
    therefore mixing operating points -- the fallback keeps old runs evaluable, it does
    not make them consistent.
    """
    fold_results = metadata.get("cv", {}).get("fold_results", {})
    fold_predictions = fold_predictions or {}
    fold_thresholds = fold_thresholds or {}

    ordered = []
    for key in sorted(fold_results):
        result = fold_results.get(key)
        if not result:
            continue
        predictions = fold_predictions.get(key)
        threshold = fold_thresholds.get(key)
        if predictions is None or threshold is None:
            ordered.append(result.get("metrics", {}))
            continue
        y_true, y_prob = predictions
        # loss needs logits, which the stored probabilities cannot reconstruct, so it is
        # carried over from training rather than recomputed. It is threshold-free, so
        # the value is the same one either route would give.
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
        recorded_loss = result.get("metrics", {}).get("loss")
        if recorded_loss is not None:
            metrics["loss"] = recorded_loss
        ordered.append(metrics)

    return aggregate_fold_metrics([m for m in ordered if m])


def build_comparison_rows(
    validation_mean: dict, ensemble: dict, test_mean: dict
) -> list:
    """One row per metric, one column per column of COMPARISON_COLUMNS.

    Kept as plain lists so it lands in summary.json unchanged and only becomes a
    wandb.Table at logging time.
    """
    rows = []
    for key in COMPARISON_METRICS:
        # Per-fold metrics rename loss -> test_loss, so that the field name in
        # test_metrics.json stays what it has always been.
        test_key = "test_loss" if key == "loss" else key
        values = [
            validation_mean.get(key),
            ensemble.get(key),
            test_mean.get(test_key),
        ]
        if all(value is None for value in values):
            continue
        rows.append([COMPARISON_LABELS.get(key, key), *values])
    return rows


def print_comparison(summary: dict, n_folds: int, test_n: int):
    std = summary["aggregate"]["std"]
    label_to_key = {label: key for key, label in COMPARISON_LABELS.items()}
    print(
        f"\n{n_folds} folds on {test_n} test samples "
        f"(validation / test ensemble / test mean +/- std):"
    )

    def cell(value):
        return "     -" if value is None else f"{value:6.3f}"

    for label, validation, ensemble, test in summary["comparison"]:
        key = label_to_key.get(label)
        key = "test_loss" if key == "loss" else key
        spread = f" +/- {std[key]:.3f}" if key in std else ""
        print(
            f"  {label:20s} {cell(validation)} / {cell(ensemble)} / {cell(test)}{spread}"
        )


def evaluate_cv_run(run_dir: Path, args):
    """Every fold's best model on the shared test set, plus mean/std and an ensemble."""
    if not getattr(args, "allow_cv_test_evaluation", False):
        raise ValueError(
            "Refusing to evaluate a CV directory on the test set without "
            "--allow-cv-test-evaluation. For final reporting, train a fresh refit and "
            "evaluate its last.pth with --threshold-from checkpoints/<cv-run>."
        )

    metadata = load_metadata(run_dir / "metadata.pth")
    config, config_report = resolve_eval_config(metadata, args.config)
    device = resolve_device(args.device or config.get("device", "auto"))
    config["device"] = str(device)

    fold_paths = discover_fold_checkpoints(run_dir)
    if not fold_paths:
        raise FileNotFoundError(
            f"No split_*/{CV_BEST_FILENAME} under {run_dir}. Has any fold finished?"
        )

    evaluation = resolve_evaluation_config(config, args)
    # Read once, leniently, and share: the threshold selection needs every fold's
    # predictions and the validation column needs whichever ones exist.
    fold_predictions, _ = load_fold_validation_predictions(fold_paths, strict=False)
    selection = select_fold_thresholds(
        fold_paths, config, args, fold_predictions=fold_predictions
    )
    fold_thresholds = selection["fold_thresholds"]
    print(describe_selection(selection))

    test_idx = resolve_test_idx(metadata)
    source = build_dataset_source(config)
    test_loader = source.test_loader(test_idx)
    output_root = evaluation["output_dir"] / metadata["run_id"]

    fold_metrics = []
    fold_probs = []
    y_true = None
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        metrics, results, threshold = evaluate_checkpoint(
            checkpoint, config, test_loader, device, fold_thresholds[fold_number]
        )
        metrics["fold"] = fold_number
        metrics["checkpoint"] = str(path)
        # What the shared operating point asked for, and what this fold could actually
        # realise on its own validation split. They differ because a fold's achievable
        # false positive rates are quantised by its negative count.
        if "target_fpr" in selection:
            metrics["operating_fpr_target"] = selection["target_fpr"]
            metrics["operating_fpr_realised_on_val"] = selection[
                "fold_realised_fpr"
            ].get(fold_number)

        fold_dir = output_root / f"fold_{fold_number}"
        save_outputs(
            fold_dir,
            metrics,
            build_predictions_frame(source, test_idx, results, threshold),
            save_predictions=evaluation["save_predictions"],
        )
        save_confusion_matrix_plot(
            confusion_matrix_values=metrics["confusion_matrix"],
            output_path=fold_dir / "confusion_matrix.png",
            title=f"Test Confusion Matrix (fold {fold_number})",
        )

        fold_metrics.append(metrics)
        fold_probs.append(results["y_prob"])
        # Same loader and ordering for every fold, so the labels are shared.
        y_true = results["y_true"]
        print(
            f"fold {fold_number}: auc={metrics['roc_auc']:.3f} "
            f"balanced_acc={metrics['balanced_accuracy']:.3f} "
            f"acc={metrics['accuracy']:.3f} threshold={threshold:.4f}"
        )

    ensemble_prob = np.mean(fold_probs, axis=0)
    ensemble_threshold, threshold_source = ensemble_operating_point(
        selection, fold_thresholds
    )
    ensemble_metrics = compute_metrics(
        y_true, ensemble_prob, threshold=ensemble_threshold
    )
    ensemble_metrics["n"] = int(len(y_true))
    ensemble_metrics["threshold_source"] = threshold_source
    # roc_auc and average_precision are the honest read on the ensemble; the thresholded
    # metrics below them are comparability, not a validated operating point. Averaging
    # the folds' probabilities does not average their calibration, so the inherited cut
    # sits at an unknown place on the ensemble's own score scale -- and every sample in
    # the CV pool was trained on by all but one of the models, so no held-out data is
    # left on which a better cut could be chosen.
    ensemble_metrics["note"] = (
        f"Thresholded at {ensemble_threshold:.4f} ({threshold_source}), which is "
        "inherited rather than validated: the ensemble has no held-out data of its own "
        "on which to choose an operating point, and averaging the folds' probabilities "
        "does not average their calibration. roc_auc and average_precision need no "
        "threshold and are unaffected."
    )
    ensemble_dir = output_root / "ensemble"
    save_outputs(
        ensemble_dir,
        ensemble_metrics,
        pd.DataFrame(
            {
                "image": [source.item_id(idx) for idx in test_idx],
                "label": y_true,
                "pred": (ensemble_prob >= ensemble_threshold).astype(int),
                "prob_class_1": ensemble_prob,
                "threshold": ensemble_threshold,
            }
        ),
        save_predictions=evaluation["save_predictions"],
    )
    save_confusion_matrix_plot(
        confusion_matrix_values=ensemble_metrics["confusion_matrix"],
        output_path=ensemble_dir / "confusion_matrix.png",
        title="Test Confusion Matrix (ensemble)",
    )

    aggregate = aggregate_fold_metrics(fold_metrics)
    validation = cv_validation_aggregate(
        metadata, fold_thresholds=fold_thresholds, fold_predictions=fold_predictions
    )
    summary = {
        "run_id": metadata["run_id"],
        "folds_evaluated": sorted(fold_paths),
        "test_n": len(test_idx),
        "threshold_selection": selection,
        "eval_config": config_report,
        "aggregate": aggregate,
        "ensemble": ensemble_metrics,
        "cv_validation": validation,
        "comparison": build_comparison_rows(
            validation_mean=validation["mean"],
            ensemble=ensemble_metrics,
            test_mean=aggregate["mean"],
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_comparison(summary, n_folds=len(fold_metrics), test_n=len(test_idx))
    print(f"\nSaved evaluation outputs to {output_root}")

    if evaluation["log_wandb"]:
        log_cv_to_wandb(metadata, config, summary)
    return summary


def prune_stale_summary_keys(run) -> list[str]:
    """Drop summary entries that are no longer written, on a run that already has them.

    A wandb summary persists: a key this script stops writing does not disappear, it just
    sits at whatever it last held. Removing them here means re-evaluating an older run
    cleans up its summary instead of leaving `Test Mean operating_fpr_target` and the
    per-fold `Fold ...` values next to the current numbers.
    """
    retired = {
        f"Test {statistic} {metric}"
        for statistic in ("Mean", "Std")
        for metric in NON_RESULT_METRICS
    }
    stale = [
        key
        for key in list(run.summary.keys())
        if key in retired or key == "Fold" or key.startswith("Fold ")
    ]
    for key in stale:
        del run.summary[key]
    return stale


def log_cv_to_wandb(metadata, config, summary: dict):
    run = wandb.init(
        id=metadata["run_id"],
        resume="must",
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    prune_stale_summary_keys(run)

    for key, value in summary["aggregate"]["mean"].items():
        if key not in NON_RESULT_METRICS:
            run.summary[f"Test Mean {key}"] = value
    for key, value in summary["aggregate"]["std"].items():
        if key not in NON_RESULT_METRICS:
            run.summary[f"Test Std {key}"] = value
    for key, value in summary["ensemble"].items():
        if isinstance(value, (int, float)) and key not in NON_RESULT_METRICS:
            run.summary[f"Test Ensemble {key}"] = value

    run.log(
        {
            "Metric Comparison": wandb.Table(
                columns=list(COMPARISON_COLUMNS), data=summary["comparison"]
            )
        }
    )
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MRI checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "A checkpoint file, or a run directory (checkpoints/<run_id>) to evaluate "
            "every cross-validation fold and report mean +/- std plus an ensemble."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON config supplying evaluation-time settings, the same file format "
            "train.py takes. Only the threshold, evaluation, device, wandb and "
            "dataloader blocks are read; everything else comes from the run's own "
            "metadata, since it describes the trained model rather than its evaluation."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-wandb", action="store_true")
    # Two ways to supply an operating point outright. Mutually exclusive at the argparse
    # level so passing both is a usage error rather than one silently winning.
    operating_point = parser.add_mutually_exclusive_group()
    operating_point.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Pin the decision threshold for every fold, bypassing threshold selection "
            "entirely. Defaults to whatever the configured strategy chooses, or to the "
            "threshold the checkpoint recorded for a single non-CV model."
        ),
    )
    operating_point.add_argument(
        "--threshold-from",
        type=Path,
        default=None,
        metavar="CV_RUN_DIR",
        help=(
            "Inherit the operating point another cross-validation run's ensemble was "
            "scored at, given that run's directory (checkpoints/<run_id>). This is how "
            "a single model refit on all the training data gets a threshold without "
            "holding anything back: the folds of the same architecture already chose "
            "one, and reusing it makes the two directly comparable."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=CV_THRESHOLD_STRATEGIES,
        default=None,
        help="Override threshold.strategy for a cross-validation run.",
    )
    parser.add_argument(
        "--allow-cv-test-evaluation",
        action="store_true",
        help=(
            "Explicitly permit legacy diagnostic evaluation of every CV fold on the "
            "test set. Final reporting should evaluate a fresh refit with "
            "--threshold-from instead."
        ),
    )
    parser.add_argument(
        "--fpr-rounding",
        choices=FPR_ROUNDING_POLICIES,
        default=None,
        help=(
            "Override threshold.fpr_rounding. How a fold resolves a target false "
            "positive rate it cannot hit exactly."
        ),
    )
    args = parser.parse_args()

    # A run directory means cross-validation: evaluate every fold on the shared test set.
    if args.checkpoint.is_dir():
        evaluate_cv_run(args.checkpoint, args)
        return

    checkpoint, metadata = load_checkpoint_and_metadata(args.checkpoint)
    config, config_report = resolve_eval_config(metadata, args.config)
    evaluation = resolve_evaluation_config(config, args)
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    config["device"] = str(device)

    test_idx = resolve_test_idx(metadata)
    source = build_dataset_source(config)
    test_loader = source.test_loader(test_idx)

    # A single checkpoint has no folds to average across, so its operating point is
    # supplied rather than derived: pinned, inherited from a CV run, or read back from a
    # run trained while training still tuned one.
    threshold, threshold_record = resolve_single_threshold(
        checkpoint, args.checkpoint, args, metadata=metadata, config=config
    )
    print(f"threshold: {threshold:.4f} ({threshold_record['threshold_source']})")
    metrics, results, _ = evaluate_checkpoint(
        checkpoint, config, test_loader, device, threshold, args.checkpoint
    )
    metrics["threshold_source"] = threshold_record["threshold_source"]

    fold = detect_fold(checkpoint, args.checkpoint)
    output_dir = evaluation["output_dir"] / metadata["run_id"]
    if fold is not None:
        metrics["fold"] = fold
        output_dir = output_dir / f"fold_{fold}"
    # One checkpoint is deliberately scored at more than one operating point, so the
    # outputs are kept apart by where the cut came from.
    output_dir = output_dir / threshold_output_slug(threshold_record)

    predictions = build_predictions_frame(source, test_idx, results, threshold)
    save_outputs(
        output_dir, metrics, predictions, save_predictions=evaluation["save_predictions"]
    )
    save_confusion_matrix_plot(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
    )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "run_id": metadata["run_id"],
                "checkpoint": str(args.checkpoint),
                "test_n": len(test_idx),
                "threshold_selection": threshold_record,
                "eval_config": config_report,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")

    if evaluation["log_wandb"]:
        log_to_wandb(
            metadata=metadata,
            config=config,
            metrics=metrics,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
