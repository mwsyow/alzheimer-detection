import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import wandb

from datasets import build_test_loader
from metrics import (
    DEFAULT_THRESHOLD,
    collect_predictions,
    compute_metrics,
    save_confusion_matrix_plot,
)
from models import build_model
from train import build_loss


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def load_checkpoint_and_metadata(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata_path = checkpoint_path.parent / "metadata.pth"
    metadata = torch.load(metadata_path, map_location="cpu")
    return checkpoint, metadata


def save_outputs(output_dir: Path, metrics: dict, predictions: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the decision threshold. Defaults to the value tuned on validation "
            f"and stored in the checkpoint, or {DEFAULT_THRESHOLD} if absent."
        ),
    )
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

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint.get("threshold", DEFAULT_THRESHOLD)

    results = collect_predictions(model, test_loader, loss, device)
    metrics = compute_metrics(
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        threshold=threshold,
        loss=results["loss"],
    )
    # Historical field name kept so old and new test_metrics.json stay comparable.
    metrics["test_loss"] = metrics.pop("loss")

    output_dir = Path("evaluations") / metadata["run_id"]
    predictions = pd.DataFrame(
        {
            "image": [dataset_items[idx]["image"] for idx in test_idx],
            "label": results["y_true"],
            "pred": (results["y_prob"] >= threshold).astype(int),
            "prob_class_1": results["y_prob"],
            "threshold": threshold,
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
