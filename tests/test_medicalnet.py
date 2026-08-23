from types import SimpleNamespace

import pytest
import torch
from torch import nn

import models
from models import build_model
from train import build_optimizer


def medicalnet_config(*, freeze_backbone=False, params=None):
    return {
        "model": {
            "name": "ResNet10",
            "params": params
            or {"num_classes": 2, "in_channels": 1, "spatial_dims": 3},
            "pretrained": {
                "enabled": True,
                "source": "medicalnet",
                "freeze_backbone": freeze_backbone,
            },
        },
        "optimizer": {"name": "AdamW", "params": {"lr": 1e-4}},
    }


@pytest.fixture(scope="session")
def medicalnet_checkpoint(tmp_path_factory):
    model = build_model(medicalnet_config(), initialize_pretrained=False)
    state_dict = {
        f"module.{key}": value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("fc.")
    }
    path = tmp_path_factory.mktemp("medicalnet") / "resnet_10_23dataset.pth"
    torch.save({"state_dict": state_dict}, path)
    return path, state_dict


def test_medicalnet_loads_every_backbone_tensor_and_replaces_head(
    monkeypatch, medicalnet_checkpoint
):
    checkpoint_path, checkpoint_state = medicalnet_checkpoint
    monkeypatch.setattr(models, "download_medicalnet_resnet10", lambda: checkpoint_path)

    model = build_model(medicalnet_config())

    assert model.conv1.weight.shape == (64, 1, 7, 7, 7)
    assert torch.equal(model.conv1.weight, checkpoint_state["module.conv1.weight"])
    assert model.fc.out_features == 2
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32, 32))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_medicalnet_rejects_incompatible_width_before_download(monkeypatch):
    called = False

    def download():
        nonlocal called
        called = True

    monkeypatch.setattr(models, "download_medicalnet_resnet10", download)
    config = medicalnet_config(params={"widen_factor": 0.5})

    with pytest.raises(ValueError, match="widen_factor=0.5"):
        build_model(config)
    assert not called


def test_frozen_backbone_stays_in_eval_and_optimizer_contains_only_head(
    monkeypatch, medicalnet_checkpoint
):
    checkpoint_path, _ = medicalnet_checkpoint
    monkeypatch.setattr(models, "download_medicalnet_resnet10", lambda: checkpoint_path)
    config = medicalnet_config(freeze_backbone=True)

    model = build_model(config)
    model.train()
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"fc.weight", "fc.bias"}
    assert all(not layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm3d))
    assert model.fc.training

    optimizer = build_optimizer(config, model)
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimized == {id(model.fc.weight), id(model.fc.bias)}


def test_skipping_initialization_never_downloads_and_reapplies_freeze(monkeypatch):
    def fail_download():
        raise AssertionError("download should not be called")

    monkeypatch.setattr(models, "download_medicalnet_resnet10", fail_download)
    config = medicalnet_config(freeze_backbone=True)
    model = build_model(config, initialize_pretrained=False)

    assert model.fc.out_features == 2
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"fc.weight", "fc.bias"}
    optimizer = build_optimizer(config, model)
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimized == {id(model.fc.weight), id(model.fc.bias)}


def test_hub_download_is_pinned_and_checksum_verified(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )
    monkeypatch.setattr(models, "_sha256", lambda _: models.MEDICALNET_RESNET10_SHA256)

    assert models.download_medicalnet_resnet10() == checkpoint
    assert calls == [
        {
            "repo_id": models.MEDICALNET_RESNET10_REPO,
            "filename": models.MEDICALNET_RESNET10_FILENAME,
            "revision": models.MEDICALNET_RESNET10_REVISION,
        }
    ]


def test_hub_download_rejects_bad_checksum(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"corrupt")
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **_: str(checkpoint)),
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        models.download_medicalnet_resnet10()
