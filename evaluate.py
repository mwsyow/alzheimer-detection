import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from monai.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Subset

from datasets import Dataset, get_data
from models import build_model
from train import build_loss, build_transforms


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def load_checkpoint_and_metadata(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata_path = checkpoint_path.parent / "metadata.pth"
    metadata = torch.load(metadata_path, map_location="cpu")
    return checkpoint, metadata


def build_test_loader(config, test_idx: list[int]):
    img_paths = [Path(path) for path in sorted(glob.glob(config["image_glob"]))]
    if not img_paths:
        raise FileNotFoundError(
            f"No MRI images matched image_glob={config['image_glob']!r}"
        )

    dataset_items = get_data(img_paths, Path(config["label_path"]))
    mri_dataset = Dataset(data=dataset_items, transform=build_transforms(config))
    test_set = Subset(mri_dataset, test_idx)
    dataloader_config = config["dataloader"]
    return DataLoader(
        test_set,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    ), dataset_items


def evaluate(model, loss, test_loader, device: torch.device):
    model.eval()
    test_losses = []
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_losses.append(loss(outputs, labels).item())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    return {
        "test_loss": sum(test_losses) / len(test_losses),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def compute_metrics(results):
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = None

    return {
        "test_loss": results["test_loss"],
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auroc,
        "confusion_matrix": cm.tolist(),
    }


def save_outputs(output_dir: Path, metrics: dict, predictions: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def save_confusion_matrix_plot(confusion_matrix_values: list[list[int]], output_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion_matrix_values, cmap="Blues")
    class_names = ["CDR 0", "CDR > 0"]

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")

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


def log_to_wandb(metadata, config, metrics: dict):
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

    plot_path = Path("evaluations") / metadata["run_id"] / "confusion_matrix.png"
    run.log({"Test Confusion Matrix": wandb.Image(str(plot_path))})
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MRI checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-wandb", action="store_true")
    args = parser.parse_args()

    checkpoint, metadata = load_checkpoint_and_metadata(args.checkpoint)
    config = metadata["config"]
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    config["device"] = str(device)

    test_idx = metadata["split"]["test_idx"]
    test_loader, dataset_items = build_test_loader(config, test_idx)

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loss = build_loss(config)

    results = evaluate(model, loss, test_loader, device)
    metrics = compute_metrics(results)

    output_dir = Path("evaluations") / metadata["run_id"]
    predictions = pd.DataFrame(
        {
            "image": [dataset_items[idx]["image"] for idx in test_idx],
            "label": results["y_true"],
            "pred": results["y_pred"],
            "prob_class_1": results["y_prob"],
        }
    )
    save_outputs(output_dir, metrics, predictions)
    save_confusion_matrix_plot(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")

    if args.log_wandb:
        log_to_wandb(
            metadata=metadata,
            config=config,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
