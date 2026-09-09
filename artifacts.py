"""Per-subject feature export from a fold's selected checkpoint."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch

from tasks import adaptation_mode, is_regression, task_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def comparison_fingerprint(config: dict) -> str:
    """Fingerprint scientific settings while excluding seeds and output locations."""
    comparable = json.loads(json.dumps(dict(config), default=str))
    comparable.pop("seed", None)
    comparable.pop("device", None)
    comparable.pop("wandb_name", None)
    comparable.get("cv", {}).pop("random_seed", None)
    comparable.get("cv", {}).pop("manifest_path", None)
    comparable.pop("evaluation", None)
    comparable.pop("artifacts", None)
    comparable.get("checkpoint", {}).pop("dir", None)
    return config_fingerprint(comparable)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def export_fold_artifacts(
    *,
    model,
    source,
    split_indices: dict,
    checkpoint_path: Path,
    checkpoint: dict,
    config: dict,
    fold: int,
    device: torch.device,
    loss_fn,
    output_root: Path,
) -> list[dict]:
    """Export F/hI and predictions for every split using evaluation transforms."""
    model.eval()
    checkpoint_hash = file_sha256(checkpoint_path)
    fingerprint = config_fingerprint(config)
    comparable_fingerprint = comparison_fingerprint(config)
    rows = []
    split_keys = {"train": "train_idx", "validation": "val_idx", "test": "test_idx"}

    with torch.no_grad():
        for split_name, index_key in split_keys.items():
            indices = list(split_indices[index_key])
            loader = source.loader(indices, mode="test", shuffle=False)
            cursor = 0
            for images, targets in loader:
                images = images.to(device)
                outputs = model.forward_with_features(images)
                batch_size = images.shape[0]
                batch_indices = indices[cursor : cursor + batch_size]
                cursor += batch_size

                raw_output = outputs["output"].detach().cpu()
                if is_regression(config):
                    predictions = loss_fn.inverse(raw_output.reshape(-1)).cpu()
                    probabilities = logits = None
                else:
                    logits = raw_output
                    image_logits = logits[:, 1] - logits[:, 0]
                    probabilities = torch.sigmoid(image_logits)

                for offset, dataset_index in enumerate(batch_indices):
                    item = source.items[dataset_index]
                    subject_id = source.subject_id(dataset_index)
                    directory = output_root / f"fold_{fold}" / split_name
                    directory.mkdir(parents=True, exist_ok=True)
                    artifact_path = directory / (
                        f"{_safe_name(subject_id)}__idx{dataset_index}.pt"
                    )
                    target = (
                        float(item["age"])
                        if is_regression(config)
                        else int(item["label"])
                    )
                    payload = {
                        "F": outputs["F"][offset].detach().cpu().float().clone(),
                        "hI": outputs["hI"][offset].detach().cpu().float().clone(),
                        "output": raw_output[offset].float().clone(),
                        "subject_id": subject_id,
                        "image_id": source.item_id(dataset_index),
                        "dataset_index": dataset_index,
                        "target": target,
                        "task": task_name(config),
                        "seed": int(config.get("seed", 0)),
                        "cv_random_seed": config["cv"].get("random_seed"),
                        "fold": fold,
                        "split": split_name,
                        "best_epoch": int(checkpoint["epoch"]),
                        "checkpoint": str(checkpoint_path),
                        "run_metadata": str(checkpoint_path.parent.parent / "metadata.pth"),
                        "checkpoint_sha256": checkpoint_hash,
                        "architecture": config["model"]["name"],
                        "adaptation_mode": adaptation_mode(config),
                        "config_sha256": fingerprint,
                        "comparison_sha256": comparable_fingerprint,
                    }
                    if is_regression(config):
                        payload["age_prediction_standardized"] = float(
                            raw_output[offset].reshape(-1)[0]
                        )
                        payload["age_prediction_years"] = float(predictions[offset])
                    else:
                        payload["logits"] = logits[offset].float().clone()
                        payload["image_logit"] = float(image_logits[offset])
                        payload["image_probability"] = float(probabilities[offset])

                    temporary = artifact_path.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(artifact_path)
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "image_id": source.item_id(dataset_index),
                            "dataset_index": dataset_index,
                            "target": target,
                            "task": task_name(config),
                            "seed": int(config.get("seed", 0)),
                            "cv_random_seed": config["cv"].get("random_seed"),
                            "fold": fold,
                            "split": split_name,
                            "best_epoch": int(checkpoint["epoch"]),
                            "checkpoint": str(checkpoint_path),
                            "run_metadata": str(
                                checkpoint_path.parent.parent / "metadata.pth"
                            ),
                            "checkpoint_sha256": checkpoint_hash,
                            "architecture": config["model"]["name"],
                            "adaptation_mode": adaptation_mode(config),
                            "config_sha256": fingerprint,
                            "comparison_sha256": comparable_fingerprint,
                            "artifact": str(artifact_path),
                            "F_shape": json.dumps(list(payload["F"].shape)),
                            "hI_shape": json.dumps(list(payload["hI"].shape)),
                        }
                    )

    manifest = output_root / f"fold_{fold}" / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return rows


def consolidate_manifests(output_root: Path) -> Path:
    paths = sorted(output_root.glob("fold_*/manifest.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold manifests under {output_root}")
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    output = output_root / "manifest.csv"
    combined.to_csv(output, index=False)
    return output
