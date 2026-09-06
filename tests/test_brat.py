import json

import pytest
import torch

from models import DenseNet121, build_model


def brat_config(path, *, model_name="DenseNet121"):
    params = {"num_classes": 2, "in_channels": 1, "spatial_dims": 3}
    return {
        "model": {
            "name": model_name,
            "params": params,
            "pretrained": {
                "enabled": True,
                "source": "brat",
                "pretrained_weights_path": str(path),
                "freeze_backbone": False,
            },
        }
    }


@pytest.fixture(scope="module")
def brat_checkpoint(tmp_path_factory):
    source = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
    state_dict = {
        f"visual_encoder.densenet.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    state_dict["query_tokens"] = torch.zeros(1, 32, 768)
    path = tmp_path_factory.mktemp("brat") / "brat_t1c_densenet121.bin"
    torch.save(state_dict, path)
    return path, source.state_dict()


def test_brat_loads_every_backbone_tensor_and_replaces_head(brat_checkpoint):
    checkpoint_path, source_state = brat_checkpoint
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("class_layers.out.")
    }

    model.load_brat_weights(str(checkpoint_path))

    classifier_keys = {"class_layers.out.weight", "class_layers.out.bias"}
    for key, value in model.state_dict().items():
        if key in classifier_keys:
            assert torch.equal(value, original_head[key])
        else:
            assert torch.equal(value, source_state[key])


def test_brat_build_model_has_finite_forward_pass(brat_checkpoint):
    checkpoint_path, _ = brat_checkpoint
    model = build_model(brat_config(checkpoint_path))
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 32, 64, 64))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_brat_rejects_incomplete_backbone(tmp_path, brat_checkpoint):
    _, source_state = brat_checkpoint
    state_dict = {
        f"visual_encoder.densenet.{key}": value
        for key, value in source_state.items()
        if key != "features.conv0.weight"
    }
    path = tmp_path / "incomplete.bin"
    torch.save(state_dict, path)

    with pytest.raises(RuntimeError, match="features.conv0.weight"):
        build_model(brat_config(path))


def test_brat_rejects_wrong_prefix(tmp_path):
    path = tmp_path / "wrong-prefix.bin"
    torch.save({"features.conv0.weight": torch.zeros(1)}, path)

    with pytest.raises(RuntimeError, match="contains no keys beneath"):
        build_model(brat_config(path))


def test_brat_rejects_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="BRAT checkpoint does not exist"):
        build_model(brat_config("missing-brat-checkpoint.bin"))


def test_brat_source_requires_densenet121():
    with pytest.raises(ValueError, match="supported only by DenseNet121"):
        build_model(brat_config("unused.bin", model_name="ResNet10"))


def test_brat_source_requires_manual_checkpoint_path():
    config = brat_config("unused.bin")
    del config["model"]["pretrained"]["pretrained_weights_path"]

    with pytest.raises(ValueError, match="pretrained_weights_path"):
        build_model(config)


def test_resume_build_skips_checkpoint_read():
    model = build_model(
        brat_config("not-present-on-resume.bin"), initialize_pretrained=False
    )
    assert isinstance(model, DenseNet121)


def test_paired_brat_configs_differ_only_in_name_and_transforms():
    with open("configs/brat_densenet121_current.json") as handle:
        current = json.load(handle)
    with open("configs/brat_densenet121_source_aligned.json") as handle:
        source_aligned = json.load(handle)

    for config in (current, source_aligned):
        config.pop("wandb_name")
        config.pop("transforms")
    assert current == source_aligned


def test_source_aligned_config_matches_published_brat_preprocessing():
    with open("configs/brat_densenet121_source_aligned.json") as handle:
        transforms = json.load(handle)["transforms"]

    assert transforms["pixdim"] == [1.0, 1.0, 1.0]
    assert transforms["axcodes"] == "SAR"
    assert transforms["spatial_size"] == [32, 256, 256]
    assert transforms["normalize_nonzero"] is True
    assert transforms["normalize_channel_wise"] is True
    assert transforms["scale_channel_wise"] is True
    assert transforms["intensity_order"] == "normalize_then_scale"
