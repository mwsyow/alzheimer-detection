from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_THRESHOLDS = 1000
DEFAULT_THRESHOLD_OBJECTIVE = "balanced_accuracy"
DEFAULT_THRESHOLD_TIE_BREAK = "plateau_midpoint"
THRESHOLD_OBJECTIVES = (
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
)
THRESHOLD_TIE_BREAKS = (
    "plateau_midpoint",
    "lowest",
    "highest",
    "closest_to_0_5",
)

# How a cross-validation run turns K folds into one operating point.
#   vertical_average  -- Fawcett (2006) Alg. 3: average TPR over a shared FPR grid,
#                        pick a target FPR, solve each fold back to its own cut.
#   threshold_average -- Fawcett (2006) Alg. 4, as implemented by sklearn's
#                        TunedThresholdClassifierCV: average balanced accuracy over a
#                        shared threshold grid, take one argmax, share that one cut.
#   per_fold_youden   -- each fold keeps the threshold it tuned during training.
CV_THRESHOLD_STRATEGIES = (
    "cv_common_threshold",
    "fixed",
    "vertical_average",
    "per_fold_youden",
    "threshold_average",
)
# New configuration files are normalised to this strategy before training metadata is
# written. DEFAULT_CV_THRESHOLD_STRATEGY deliberately remains the historical fallback
# for metadata created before a strategy field existed.
DEFAULT_NEW_CV_THRESHOLD_STRATEGY = "cv_common_threshold"
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
#
# A registry of every label, not a selection of what to report: evaluate.py's comparison
# table iterates it in both directions -- forwards to name a row, backwards to find that
# row's std -- so dropping a key here silently drops a column there. The threshold-
# dependent entries are unreachable from training and reachable from evaluation.
WANDB_METRIC_LABELS = {
    "loss": "Loss",
    "roc_auc": "AUC",
    "average_precision": "Average Precision",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "npv": "NPV",
    "fpr": "FPR",
    "threshold": "Threshold",
}

# What training logs each epoch, for both splits. Every one of these is threshold-free:
# training picks no operating point, so it has none to report metrics at. A cut is chosen
# once, in evaluate.py, from the validation predictions the checkpoints carry -- which is
# the only place it can be applied to data it was not chosen on.
EPOCH_LOG_METRICS = ("loss", "roc_auc", "average_precision")


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

    Nothing calls this any more. It is kept because it is the definition of the numbers
    the `per_fold_youden` strategy reads back: every threshold stored in a checkpoint
    from before selection left the training loop came from here, chosen on the same
    validation split it was then used to score. Do not reintroduce it into training --
    that is what made those per-epoch operating-point metrics optimistic.
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


def pack_predictions(y_true, y_prob, indices=None) -> dict:
    """Tensor-only view of one split's predictions, for storing in a checkpoint.

    Checkpoints are loaded without ``weights_only=False``, and that guard rejects numpy
    arrays, so the ndarrays from summarize_predictions cannot be stored as they are.
    A str-keyed dict of tensors is allowed, and needs no change at any load site.

    The clone is not redundant: torch.as_tensor shares the numpy buffer, and torch.save
    serialises a tensor's whole underlying storage rather than its view.
    """
    packed = {
        "y_true": torch.as_tensor(np.asarray(y_true)).long().clone(),
        "y_prob": torch.as_tensor(np.asarray(y_prob)).float().clone(),
    }
    if indices is not None:
        packed["indices"] = torch.as_tensor(np.asarray(indices)).long().clone()
    return packed


def unpack_predictions(packed: dict):
    """Inverse of pack_predictions, tolerating values that are already numpy."""

    def to_numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    return to_numpy(packed["y_true"]), to_numpy(packed["y_prob"])


def unpack_prediction_bundle(packed: dict):
    """Return labels, probabilities, and optional sample indices.

    ``unpack_predictions`` intentionally keeps its historical two-value return shape;
    callers that verify out-of-fold provenance use this indexed variant instead.
    """

    y_true, y_prob = unpack_predictions(packed)
    indices = packed.get("indices")
    if indices is not None:
        indices = (
            indices.detach().cpu().numpy()
            if torch.is_tensor(indices)
            else np.asarray(indices)
        )
    return y_true, y_prob, indices


def _binary_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _threshold_diagnostics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = _binary_counts(y_true, y_pred)

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def threshold_tie_index(
    values, thresholds, tie_break=DEFAULT_THRESHOLD_TIE_BREAK
) -> int:
    """Resolve equal maxima on an ordered threshold grid deterministically."""
    if tie_break not in THRESHOLD_TIE_BREAKS:
        raise ValueError(
            f"Unsupported threshold.tie_break: {tie_break!r}. "
            f"Expected one of {sorted(THRESHOLD_TIE_BREAKS)}."
        )
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or values.size == 0 or values.shape != thresholds.shape:
        raise ValueError(
            "threshold values and grid must be non-empty one-dimensional arrays"
        )
    best = np.nanmax(values)
    maximisers = np.flatnonzero(
        np.isclose(values, best, rtol=0.0, atol=1e-12)
    )

    if tie_break == "lowest":
        return int(maximisers[0])
    if tie_break == "highest":
        return int(maximisers[-1])
    if tie_break == "closest_to_0_5":
        distances = np.abs(thresholds[maximisers] - 0.5)
        return int(
            maximisers[np.flatnonzero(np.isclose(distances, distances.min()))[0]]
        )

    runs = np.split(maximisers, np.flatnonzero(np.diff(maximisers) != 1) + 1)
    widest_length = max(len(run) for run in runs)
    widest = next(run for run in runs if len(run) == widest_length)
    # Lower central candidate for an even-sized plateau. It is still an evaluated
    # member of T, and makes the tie deterministic without inventing a non-grid cut.
    return int(widest[(len(widest) - 1) // 2])


def common_threshold_operating_point(
    fold_predictions: dict,
    objective: str = DEFAULT_THRESHOLD_OBJECTIVE,
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    tie_break: str = DEFAULT_THRESHOLD_TIE_BREAK,
) -> dict:
    """Select one numerical cut by maximising the mean fold objective.

    This API deliberately has no test arguments: only stored out-of-fold validation
    labels and probabilities can influence the selected threshold.
    """
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(
            f"Unsupported threshold.objective: {objective!r}. "
            f"Expected one of {sorted(THRESHOLD_OBJECTIVES)}."
        )
    if not isinstance(num_thresholds, int) or num_thresholds < 2:
        raise ValueError(
            "threshold.num_thresholds must be an integer >= 2, "
            f"got {num_thresholds!r}"
        )
    if len(fold_predictions or {}) < 2:
        raise ValueError("cv_common_threshold needs at least 2 folds")

    prepared = {}
    for fold, (y_true, y_prob) in sorted(fold_predictions.items()):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
            raise ValueError(
                f"fold {fold} labels and probabilities must be equal-length vectors"
            )
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(f"fold {fold} contains labels outside {{0, 1}}")
        if not np.all(np.isfinite(y_prob)) or np.any(
            (y_prob < 0.0) | (y_prob > 1.0)
        ):
            raise ValueError(
                f"fold {fold} probabilities must be finite and in [0, 1]"
            )
        prepared[fold] = (y_true.astype(int), y_prob)

    grid = np.linspace(0.0, 1.0, num_thresholds)
    rows = []
    objective_curve = []
    for threshold in grid:
        fold_metrics = {
            fold: _threshold_diagnostics(y_true, y_prob, float(threshold))
            for fold, (y_true, y_prob) in prepared.items()
        }

        def aggregate(key):
            values = [metrics[key] for metrics in fold_metrics.values()]
            return float(np.mean(values)), float(np.std(values, ddof=1))

        mean_objective, std_objective = aggregate(objective)
        mean_balanced, std_balanced = aggregate("balanced_accuracy")
        mean_sensitivity, std_sensitivity = aggregate("sensitivity")
        mean_specificity, std_specificity = aggregate("specificity")
        row = {
            "threshold": float(threshold),
            "mean_objective": mean_objective,
            "std_objective": std_objective,
            "mean_balanced_accuracy": mean_balanced,
            "std_balanced_accuracy": std_balanced,
            "mean_sensitivity": mean_sensitivity,
            "std_sensitivity": std_sensitivity,
            "mean_specificity": mean_specificity,
            "std_specificity": std_specificity,
        }
        row.update(
            {
                f"fold_{fold}_balanced_accuracy": metrics["balanced_accuracy"]
                for fold, metrics in fold_metrics.items()
            }
        )
        rows.append(row)
        objective_curve.append(mean_objective)

    selected_index = threshold_tie_index(objective_curve, grid, tie_break)
    selected = rows[selected_index]
    threshold = selected["threshold"]
    return {
        "strategy": "cv_common_threshold",
        "objective": objective,
        "num_thresholds": num_thresholds,
        "tie_break": tie_break,
        "selected_index": selected_index,
        "shared_threshold": threshold,
        "fold_thresholds": {fold: threshold for fold in prepared},
        "skipped_folds": [],
        "mean_objective": selected["mean_objective"],
        "std_objective": selected["std_objective"],
        "mean_balanced_accuracy": selected["mean_balanced_accuracy"],
        "std_balanced_accuracy": selected["std_balanced_accuracy"],
        "mean_sensitivity": selected["mean_sensitivity"],
        "std_sensitivity": selected["std_sensitivity"],
        "mean_specificity": selected["mean_specificity"],
        "std_specificity": selected["std_specificity"],
        "curve": rows,
    }


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
    objective: str = DEFAULT_THRESHOLD_OBJECTIVE,
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    tie_break: str = DEFAULT_THRESHOLD_TIE_BREAK,
    fixed_value: float = None,
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
            f"Unsupported threshold.strategy/threshold.cv_strategy: {strategy!r}. "
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

    if strategy == "fixed":
        if (
            fixed_value is None
            or not np.isfinite(fixed_value)
            or not 0.0 <= fixed_value <= 1.0
        ):
            raise ValueError("fixed threshold strategy needs threshold.value in [0, 1]")
        folds = set((fold_predictions or {}).keys()) | set(
            (fold_stored_thresholds or {}).keys()
        )
        return {
            "strategy": strategy,
            "shared_threshold": float(fixed_value),
            "fold_thresholds": {fold: float(fixed_value) for fold in sorted(folds)},
            "skipped_folds": [],
        }

    if not fold_predictions:
        raise ValueError(f"{strategy} needs each fold's validation predictions")

    if strategy == "vertical_average":
        return vertical_average_operating_point(
            fold_predictions, fpr_grid=fpr_grid, rounding=fpr_rounding
        )
    if strategy == "threshold_average":
        return threshold_average_operating_point(
            fold_predictions, threshold_grid=threshold_grid
        )
    return common_threshold_operating_point(
        fold_predictions,
        objective=objective,
        num_thresholds=num_thresholds,
        tie_break=tie_break,
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
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    if len(np.unique(y_true)) < 2:
        roc_auc = None
        average_precision = None
    else:
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
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "fpr": float(fpr),
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


def to_wandb_logs(metrics: dict, prefix: str, keys=EPOCH_LOG_METRICS):
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
