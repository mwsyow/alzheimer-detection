import pytest
import torch

from inference import load_model
from models import build_model


MODEL_CONFIG = {
    "model": {
        "name": "Simple3DCNN",
        "params": {
            "in_channels": 1,
            "num_classes": 2,
            "channels": [2],
            "use_batch_norm": False,
        },
    }
}


def write_run(tmp_path, *, nested=False):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "split_1" if nested else run_dir
    checkpoint_dir.mkdir(parents=True)

    source_model = build_model(MODEL_CONFIG)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.25)

    torch.save({"config": MODEL_CONFIG}, run_dir / "metadata.pth")
    checkpoint_path = checkpoint_dir / "model.pth"
    torch.save(
        {"model_state_dict": source_model.state_dict()},
        checkpoint_path,
    )
    return checkpoint_path, source_model


def test_load_model_restores_weights_and_enables_inference_mode(tmp_path):
    checkpoint_path, source_model = write_run(tmp_path)

    loaded = load_model(checkpoint_path)

    assert not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"
    for expected, actual in zip(source_model.parameters(), loaded.parameters()):
        torch.testing.assert_close(actual, expected)


def test_load_model_finds_metadata_above_cv_fold(tmp_path):
    checkpoint_path, _ = write_run(tmp_path, nested=True)

    loaded = load_model(checkpoint_path)

    assert not loaded.training


def test_load_model_rejects_a_non_model_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"config": MODEL_CONFIG}, run_dir / "metadata.pth")
    checkpoint_path = run_dir / "model.pth"
    torch.save({"epoch": 3}, checkpoint_path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_model(checkpoint_path)
