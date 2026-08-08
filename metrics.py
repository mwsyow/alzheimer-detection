from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

DEFAULT_THRESHOLD = 0.5

# How a cross-validation run turns K folds into one operating point.
#   vertical_average  -- Fawcett (2006) Alg. 3: average TPR over a shared FPR grid,
#                        pick a target FPR, solve each fold back to its own cut.
#   threshold_average -- Fawcett (2006) Alg. 4, as implemented by sklearn's
#                        TunedThresholdClassifierCV: average balanced accuracy over a
#                        shared threshold grid, take one argmax, share that one cut.
#   per_fold_youden   -- each fold keeps the threshold it tuned during training.
CV_THRESHOLD_STRATEGIES = ("vertical_average", "threshold_average", "per_fold_youden")
# A config written before threshold selection existed has no strategy, and must
# re-evaluate exactly as it did before.
DEFAULT_CV_THRESHOLD_STRATEGY = "per_fold_youden"

# A fold cannot always realise a target FPR: with n negatives only multiples of 1/n
# exist, so it lands on the target or steps past it in one direction.
FPR_ROUNDING_POLICIES = ("at_least", "nearest", "at_most")
# Never fall short of the target FPR, which means never trading away sensitivity --
# the right default when a missed case costs more than a false alarm.
DEFAULT_FPR_ROUNDING = "at_least"

# Metric key -> suffix used when logging to wandb, e.g. "Validation Balanced Accuracy".
WANDB_METRIC_LABELS = {
    "loss": "Loss",
    "roc_auc": "AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "threshold": "Threshold",
}

# Training metrics are computed at the validation-tuned threshold, so there is no
# separate "Training Threshold" to report.
VALIDATION_LOG_METRICS = (
    "loss",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "threshold",
)
TRAINING_LOG_METRICS = tuple(
    key for key in VALIDATION_LOG_METRICS if key != "threshold"
)


def summarize_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
):
    """Turn accumulated logits/labels into a loss and numpy arrays for metric code.

    The loss is computed once over every sample instead of averaging per-batch means,
    which is both exactly sample-weighted and correct when the loss carries class
    weights.
    """
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    weight = getattr(loss_fn, "weight", None)
    loss_device = weight.device if weight is not None else logits.device
    loss = loss_fn(logits.to(loss_device), labels.to(loss_device)).item()

    return {
        "loss": loss,
        "logits": logits,
        "y_true": labels.numpy(),
        "y_prob": logits.softmax(dim=1)[:, 1].numpy(),
    }


def collect_predictions(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device = None,
):
    """Run the whole loader in eval mode and accumulate raw logits.

    ROC-AUC is not batch-decomposable, so metrics have to be computed from the full
    set of predictions rather than averaged across batches.
    """
    model.eval()
    logit_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in loader:
            if device is not None:
                images = images.to(device)
            logit_batches.append(model(images).detach().cpu())
            label_batches.append(labels.detach().cpu())

    return summarize_predictions(
        logits=torch.cat(logit_batches),
        labels=torch.cat(label_batches),
        loss_fn=loss_fn,
    )


def select_threshold(y_true, y_prob):
    """Pick the decision threshold that maximises balanced accuracy on the ROC curve.

    Returns (threshold, balanced_accuracy). Falls back to 0.5 when only one class is
    present and the ROC curve is undefined.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD, float("nan")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)
    balanced_accuracies = (true_positive_rate + (1.0 - false_positive_rate)) / 2.0
    best_index = int(np.argmax(balanced_accuracies))

    threshold = float(thresholds[best_index])
    if not np.isfinite(threshold):
        # roc_curve prepends an infinite threshold for the "predict nothing" corner.
        threshold = 1.0

    return threshold, float(balanced_accuracies[best_index])


def pack_predictions(y_true, y_prob) -> dict:
    """Tensor-only view of one split's predictions, for storing in a checkpoint.

    Checkpoints are loaded without ``weights_only=False``, and that guard rejects numpy
    arrays, so the ndarrays from summarize_predictions cannot be stored as they are.
    A str-keyed dict of tensors is allowed, and needs no change at any load site.

    The clone is not redundant: torch.as_tensor shares the numpy buffer, and torch.save
    serialises a tensor's whole underlying storage rather than its view.
    """
    return {
        "y_true": torch.as_tensor(np.asarray(y_true)).long().clone(),
        "y_prob": torch.as_tensor(np.asarray(y_prob)).float().clone(),
    }


def unpack_predictions(packed: dict):
    """Inverse of pack_predictions, tolerating values that are already numpy."""

    def to_numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    return to_numpy(packed["y_true"]), to_numpy(packed["y_prob"])


def plateau_argmax(values, atol: float = 1e-12) -> int:
    """Index at the middle of the widest maximal plateau of ``values``.

    The maximum of an averaged metric curve is nearly always a flat run rather than a
    single point. np.argmax returns that run's leftmost index, which for a threshold or
    FPR sweep systematically biases the chosen operating point toward one end. The
    median of the maximiser set is not usable either -- when the set is not contiguous
    the median need not itself be a maximiser -- so this takes the midpoint of the
    longest contiguous run.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("plateau_argmax needs at least one value")

    best = np.nanmax(values)
    maximisers = np.flatnonzero(values >= best - atol)
    runs = np.split(maximisers, np.flatnonzero(np.diff(maximisers) != 1) + 1)
    widest = max(runs, key=len)
    return int(widest[len(widest) // 2])


def usable_folds(fold_predictions: dict) -> tuple[dict, list]:
    """Split folds into those with both classes present and those without.

    roc_curve does not raise on a single-class split -- it warns and returns values that
    are silently meaningless -- so every caller that averages curves has to filter first.
    """
    usable, skipped = {}, []
    for fold, (y_true, y_prob) in fold_predictions.items():
        if len(np.unique(np.asarray(y_true))) < 2:
            skipped.append(fold)
        else:
            usable[fold] = (np.asarray(y_true), np.asarray(y_prob))
    return usable, sorted(skipped)


def threshold_at_fpr(
    y_true,
    y_prob,
    target_fpr: float,
    rounding: str = DEFAULT_FPR_ROUNDING,
):
    """Probability cut realising ``target_fpr`` on this split, and what it truly realises.

    Returns (threshold, realised_fpr, realised_tpr). The realised rate is returned rather
    than assumed because a split's achievable FPRs are quantised by its negative count:
    with 20 negatives only multiples of 0.05 exist, so the target is rarely hit exactly.

    Unlike select_threshold, which falls back to 0.5 on a single-class split, this raises
    -- a fallback operating point would be reported as though it had been validated.
    """
    if rounding not in FPR_ROUNDING_POLICIES:
        raise ValueError(
            f"Unsupported fpr_rounding: {rounding!r}. "
            f"Expected one of {sorted(FPR_ROUNDING_POLICIES)}."
        )

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "threshold_at_fpr needs both classes present; a false positive rate is "
            "undefined without negatives."
        )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)

    if rounding == "at_least":
        allowed = np.flatnonzero(false_positive_rate >= target_fpr - 1e-12)
    elif rounding == "at_most":
        allowed = np.flatnonzero(false_positive_rate <= target_fpr + 1e-12)
    else:
        allowed = np.arange(len(false_positive_rate))
    # roc_curve always spans [0, 1], so at_least and at_most are both non-empty.
    gaps = np.abs(false_positive_rate[allowed] - target_fpr)
    closest = allowed[gaps == gaps.min()]
    # Ties break toward the larger FPR, i.e. toward sensitivity, matching at_least.
    realised_fpr = false_positive_rate[closest].max()

    # Several ROC vertices can share one FPR at different TPRs; the last is the lowest
    # cut, so it buys the most sensitivity at no cost in specificity.
    index = int(np.flatnonzero(false_positive_rate == realised_fpr)[-1])

    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        # roc_curve prepends an infinite threshold for the "predict nothing" corner.
        threshold = 1.0

    return threshold, float(false_positive_rate[index]), float(true_positive_rate[index])


def balanced_accuracy_curve(y_true, y_prob, thresholds) -> np.ndarray:
    """Balanced accuracy of ``y_prob >= t`` at every t, evaluated exactly.

    Balanced accuracy is a step function of the threshold, so a shared grid can be
    evaluated directly. sklearn's _mean_interpolated_score interpolates instead only
    because it is handed a grid that need not contain a fold's own ROC vertices.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.asarray(thresholds, dtype=float)

    predicted = y_prob[None, :] >= thresholds[:, None]
    sensitivity = predicted[:, y_true == 1].mean(axis=1)
    specificity = 1.0 - predicted[:, y_true == 0].mean(axis=1)
    return (sensitivity + specificity) / 2.0


def vertical_average_operating_point(
    fold_predictions: dict,
    fpr_grid: int = 101,
    rounding: str = DEFAULT_FPR_ROUNDING,
) -> dict:
    """Fawcett (2006) Alg. 3: average the folds' ROC curves along the FPR axis.

    Each fold's ROC is read as a function R_i(fpr) = tpr, the curves are averaged on a
    shared FPR grid, and the target FPR maximising (mean_tpr + 1 - fpr) / 2 is chosen
    once from that average. Each fold then solves that target back into its own cut.

    Because FPR counts only ranks within a fold, this is invariant to rescaling any one
    fold's probabilities -- which is the reason to prefer it when folds are calibrated
    differently, as an unevenly trained CV run generally is.
    """
    usable, skipped = usable_folds(fold_predictions)
    if len(usable) < 2:
        raise ValueError(
            f"vertical_average needs at least 2 folds with both classes present, got "
            f"{len(usable)} (skipped: {skipped})."
        )
    if fpr_grid < 2:
        raise ValueError(f"threshold.fpr_grid must be at least 2, got {fpr_grid}")

    grid = np.linspace(0.0, 1.0, fpr_grid)
    curves = []
    for y_true, y_prob in usable.values():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
        curves.append(np.interp(grid, false_positive_rate, true_positive_rate))

    mean_tpr = np.mean(curves, axis=0)
    sd_tpr = np.std(curves, axis=0, ddof=1)
    youden = (mean_tpr + (1.0 - grid)) / 2.0
    index = plateau_argmax(youden)
    target_fpr = float(grid[index])

    thresholds, realised_fpr, realised_tpr = {}, {}, {}
    for fold, (y_true, y_prob) in usable.items():
        cut, fold_fpr, fold_tpr = threshold_at_fpr(y_true, y_prob, target_fpr, rounding)
        thresholds[fold] = cut
        realised_fpr[fold] = fold_fpr
        realised_tpr[fold] = fold_tpr
    for fold in skipped:
        thresholds[fold] = DEFAULT_THRESHOLD

    return {
        "strategy": "vertical_average",
        "target_fpr": target_fpr,
        "target_specificity": float(1.0 - target_fpr),
        "mean_tpr_at_target": float(mean_tpr[index]),
        "sd_tpr_at_target": float(sd_tpr[index]),
        "mean_balanced_accuracy": float(youden[index]),
        "rounding": rounding,
        "fpr_grid": int(fpr_grid),
        "shared_threshold": None,
        "fold_thresholds": thresholds,
        "fold_realised_fpr": realised_fpr,
        "fold_realised_tpr": realised_tpr,
        "skipped_folds": skipped,
    }


def threshold_average_operating_point(
    fold_predictions: dict,
    threshold_grid: int = 0,
) -> dict:
    """Fawcett (2006) Alg. 4: average the folds' curves along the threshold axis.

    Every fold contributes balanced accuracy as a function of the threshold; the curves
    are averaged on a shared grid and one argmax is taken. Averaging the curves before
    the argmax is the point -- argmax is non-linear, so the mean of K folds' own peaks
    is not the peak of their mean, and a fold whose curve is flat should not cast a
    full-weight vote for an arbitrary tiebreak.

    Unlike vertical_average this compares probabilities across folds, so it assumes the
    folds are calibrated alike. Grid 0 uses the union of the folds' own ROC thresholds.
    """
    usable, skipped = usable_folds(fold_predictions)
    if len(usable) < 2:
        raise ValueError(
            f"threshold_average needs at least 2 folds with both classes present, got "
            f"{len(usable)} (skipped: {skipped})."
        )

    if threshold_grid:
        grid = np.linspace(0.0, 1.0, threshold_grid)
    else:
        vertices = [np.array([0.0, 1.0])]
        for y_true, y_prob in usable.values():
            _, _, thresholds = roc_curve(y_true, y_prob)
            vertices.append(thresholds[np.isfinite(thresholds)])
        grid = np.unique(np.concatenate(vertices))

    curves = [
        balanced_accuracy_curve(y_true, y_prob, grid) for y_true, y_prob in usable.values()
    ]
    mean_curve = np.mean(curves, axis=0)
    index = plateau_argmax(mean_curve)
    threshold = float(grid[index])

    return {
        "strategy": "threshold_average",
        "shared_threshold": threshold,
        "mean_balanced_accuracy": float(mean_curve[index]),
        "sd_balanced_accuracy": float(np.std([c[index] for c in curves], ddof=1)),
        "threshold_grid": int(len(grid)),
        # Every fold gets the same cut, including any that could not contribute a curve.
        "fold_thresholds": {fold: threshold for fold in fold_predictions},
        "skipped_folds": skipped,
    }


def select_cv_thresholds(
    strategy: str = DEFAULT_CV_THRESHOLD_STRATEGY,
    fold_predictions: dict = None,
    fold_stored_thresholds: dict = None,
    fpr_rounding: str = DEFAULT_FPR_ROUNDING,
    fpr_grid: int = 101,
    threshold_grid: int = 0,
) -> dict:
    """One operating point per fold, under the configured strategy.

    Always returns "strategy", "fold_thresholds", "shared_threshold" and
    "skipped_folds"; the rest of the keys are strategy-specific diagnostics worth
    recording in the run summary.

    Only the averaging strategies need fold_predictions. per_fold_youden reads the
    thresholds already written into the checkpoints, which is what lets runs trained
    before predictions were stored be re-evaluated exactly as before.
    """
    if strategy not in CV_THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unsupported threshold.cv_strategy: {strategy!r}. "
            f"Expected one of {sorted(CV_THRESHOLD_STRATEGIES)}."
        )

    if strategy == "per_fold_youden":
        if not fold_stored_thresholds:
            raise ValueError("per_fold_youden needs the thresholds stored per fold")
        return {
            "strategy": strategy,
            "shared_threshold": None,
            "fold_thresholds": dict(fold_stored_thresholds),
            "skipped_folds": [],
        }

    if not fold_predictions:
        raise ValueError(f"{strategy} needs each fold's validation predictions")

    if strategy == "vertical_average":
        return vertical_average_operating_point(
            fold_predictions, fpr_grid=fpr_grid, rounding=fpr_rounding
        )
    return threshold_average_operating_point(
        fold_predictions, threshold_grid=threshold_grid
    )


def ranking_metrics(y_true, y_prob, loss: float = None) -> dict:
    """Threshold-free metrics, plus the shape of the score distribution.

    For the fold ensemble, which has no defensible operating point of its own: averaging
    the folds' probabilities does not average their calibration, and every sample in the
    CV pool was trained on by all but one of the models, so no held-out data exists on
    which a cut could be validated. Discrimination is what the ensemble actually buys.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    def defined(value):
        """None for an undefined score, so aggregation skips it rather than averaging it.

        nan would pass aggregate_fold_metrics' int/float check and poison the mean.
        """
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    if len(np.unique(y_true)) < 2:
        # Both scores are undefined on a single-class split, but sklearn does not say so
        # consistently: roc_auc_score warns and returns nan while average_precision_score
        # warns and returns a perfectly finite 0.0. Neither is a value worth reporting,
        # and 0.0 is indistinguishable from a genuinely terrible score downstream.
        roc_auc = average_precision = None
    else:
        try:
            roc_auc = defined(roc_auc_score(y_true, y_prob))
            average_precision = defined(average_precision_score(y_true, y_prob))
        except ValueError:
            roc_auc = average_precision = None

    metrics = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "prob_mean": float(y_prob.mean()),
        "prob_std": float(y_prob.std()),
        "n": int(y_true.size),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def compute_metrics(
    y_true,
    y_prob,
    threshold: float = DEFAULT_THRESHOLD,
    loss: float = None,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        average_precision = average_precision_score(y_true, y_prob)
    except ValueError:
        roc_auc = None
        average_precision = None

    # Every scalar is cast to a plain float: sklearn hands back numpy scalars, which
    # torch.load rejects under its weights_only default when they reach metadata.pth.
    metrics = {
        "threshold": float(threshold),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "average_precision": (
            None if average_precision is None else float(average_precision)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "predicted_positive_rate": float(y_pred.mean()),
        "prob_mean": float(y_prob.mean()),
        "prob_std": float(y_prob.std()),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict]):
    """Mean and sample std across CV folds for every scalar metric.

    Non-scalar entries (the confusion matrix) and metrics that came back None for any
    fold are skipped; confusion matrices are summed instead.
    """
    if not fold_metrics:
        return {"n_folds": 0, "mean": {}, "std": {}}

    scalar_keys = [
        key
        for key in fold_metrics[0]
        if key != "confusion_matrix"
        and all(isinstance(m.get(key), (int, float)) for m in fold_metrics)
    ]

    mean = {}
    std = {}
    for key in scalar_keys:
        values = [float(m[key]) for m in fold_metrics]
        mean[key] = float(np.mean(values))
        # Ddof=1: these folds are a sample, and it is the spread we report as +/-.
        std[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    aggregate = {"n_folds": len(fold_metrics), "mean": mean, "std": std}

    matrices = [m.get("confusion_matrix") for m in fold_metrics]
    if all(m is not None for m in matrices):
        aggregate["confusion_matrix_total"] = np.sum(
            np.array(matrices), axis=0
        ).tolist()

    return aggregate


def to_wandb_logs(metrics: dict, prefix: str, keys=VALIDATION_LOG_METRICS):
    return {
        f"{prefix} {WANDB_METRIC_LABELS[key]}": metrics[key]
        for key in keys
        if metrics.get(key) is not None
    }


def save_confusion_matrix_plot(
    confusion_matrix_values: list[list[int]],
    output_path: Path,
    title: str = "Test Confusion Matrix",
):
    # Imported lazily: matplotlib is only a dev dependency, and train.py imports this
    # module without ever plotting.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion_matrix_values, cmap="Blues")
    class_names = ["CDR 0", "CDR > 0"]

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_value = max(max(row) for row in confusion_matrix_values)
    threshold = max_value / 2 if max_value else 0
    for row_idx, row in enumerate(confusion_matrix_values):
        for col_idx, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
