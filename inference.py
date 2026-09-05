"""Small public interface for loading trained Alzheimer classifiers."""

from pathlib import Path

import torch
from torch import nn

from models import build_model


def _find_metadata_path(checkpoint_path: Path) -> Path:
    """Find metadata for either a single-run or cross-validation checkpoint."""
    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        candidate = directory / "metadata.pth"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No metadata.pth beside or above {checkpoint_path}. Expected it in "
        f"{checkpoint_path.parent} or {checkpoint_path.parent.parent}."
    )


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Return a trained model ready for inference.

    The checkpoint must remain beside its run's ``metadata.pth``. Cross-validation
    checkpoints may be one directory below it in ``split_<fold>/``.

    Only load checkpoints and metadata from trusted sources: PyTorch checkpoint files
    are serialized artifacts, and this project's metadata requires full deserialization.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain model_state_dict."
        )

    metadata_path = _find_metadata_path(checkpoint_path)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, dict) or "config" not in metadata:
        raise ValueError(f"Metadata {metadata_path} does not contain a model config.")

    # The trained state dict is complete, so loading the original pretraining source
    # again would only add unnecessary downloads and filesystem dependencies.
    model = build_model(metadata["config"], initialize_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


__all__ = ["load_model"]
