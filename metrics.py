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
