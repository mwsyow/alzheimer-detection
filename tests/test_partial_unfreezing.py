import pytest
from torch import nn

from models import build_model
from train import build_optimizer


MODEL_PARAMS = {
    "DenseNet121": {"spatial_dims": 3, "in_channels": 1, "num_classes": 2},
    "ResNet10": {"spatial_dims": 3, "in_channels": 1, "num_classes": 2},
    "EfficientNetBN": {
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 2,
        "model_name": "efficientnet-b0",
    },
    "EfficientNet": {
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 2,
        "model_name": "efficientnet-b0",
    },
}


def config(
    model_name="ResNet10",
    *,
    stages=None,
    enabled=True,
    freeze_backbone=True,
):
    pretrained = {
        "enabled": enabled,
        "freeze_backbone": freeze_backbone,
    }
    if stages is not None:
        pretrained["unfreeze_last_stages"] = stages
    return {
        "model": {
            "name": model_name,
            "params": MODEL_PARAMS[model_name],
            "pretrained": pretrained,
        },
        "optimizer": {"name": "AdamW", "params": {"lr": 1e-4}},
    }


def trainable_names(model):
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


@pytest.mark.parametrize(
    ("model_name", "classifier_prefix"),
    [
        ("DenseNet121", "class_layers.out."),
        ("ResNet10", "fc."),
        ("EfficientNetBN", "_fc."),
        ("EfficientNet", "_fc."),
    ],
)
def test_zero_stages_preserves_classifier_only_training(model_name, classifier_prefix):
    model = build_model(config(model_name, stages=0), initialize_pretrained=False)

    names = trainable_names(model)

    assert names
    assert all(name.startswith(classifier_prefix) for name in names)


@pytest.mark.parametrize(
    ("model_name", "allowed_prefixes", "required_prefix"),
    [
        (
            "DenseNet121",
            ("features.denseblock4.", "features.norm5.", "class_layers.out."),
            "features.denseblock4.",
        ),
        ("ResNet10", ("layer4.", "fc."), "layer4."),
        (
            "EfficientNetBN",
            ("_blocks.6.", "_conv_head.", "_bn1.", "_fc."),
            "_blocks.6.",
        ),
        (
            "EfficientNet",
            ("_blocks.6.", "_conv_head.", "_bn1.", "_fc."),
            "_blocks.6.",
        ),
    ],
)
def test_one_stage_trains_only_the_architecture_tail(
    model_name, allowed_prefixes, required_prefix
):
    model = build_model(config(model_name, stages=1), initialize_pretrained=False)

    names = trainable_names(model)

    assert names
    assert all(name.startswith(allowed_prefixes) for name in names)
    assert any(name.startswith(required_prefix) for name in names)


@pytest.mark.parametrize(
    ("model_name", "stage_count"),
    [
        ("DenseNet121", 5),
        ("ResNet10", 5),
        ("EfficientNetBN", 8),
        ("EfficientNet", 8),
    ],
)
def test_maximum_stage_count_trains_every_parameter(model_name, stage_count):
    model = build_model(
        config(model_name, stages=stage_count), initialize_pretrained=False
    )

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_resnet_intermediate_depth_has_a_hard_stage_boundary():
    model = build_model(config("ResNet10", stages=2), initialize_pretrained=False)
    names = trainable_names(model)

    assert any(name.startswith("layer3.") for name in names)
    assert any(name.startswith("layer4.") for name in names)
    assert not any(name.startswith("layer2.") for name in names)
    assert not model.bn1.weight.requires_grad


def test_selected_normalization_and_dropout_train_but_frozen_normalization_does_not():
    model = build_model(
        config("EfficientNetBN", stages=1), initialize_pretrained=False
    )

    model.train()

    assert model.training
    assert not model._bn0.training
    assert model._bn1.training
    assert model._dropout.training
    assert model._fc.training
    model.eval()
    assert not any(module.training for module in model.modules())


def test_optimizer_contains_exactly_the_partially_trainable_parameters():
    cfg = config("ResNet10", stages=1)
    model = build_model(cfg, initialize_pretrained=False)

    optimizer = build_optimizer(cfg, model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_stage_count_must_be_an_integer(value):
    with pytest.raises(TypeError, match="must be an integer"):
        build_model(config(stages=value), initialize_pretrained=False)


def test_null_stage_count_is_rejected():
    cfg = config()
    cfg["model"]["pretrained"]["unfreeze_last_stages"] = None
    with pytest.raises(TypeError, match="must be an integer"):
        build_model(cfg, initialize_pretrained=False)


def test_negative_stage_count_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        build_model(config(stages=-1), initialize_pretrained=False)


def test_stage_count_above_architecture_maximum_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 5 for ResNet10"):
        build_model(config(stages=6), initialize_pretrained=False)


def test_partial_unfreezing_requires_pretraining():
    with pytest.raises(ValueError, match="requires pretraining"):
        build_model(config(stages=1, enabled=False), initialize_pretrained=False)


def test_partial_unfreezing_rejects_an_unsupported_architecture():
    cfg = {
        "model": {
            "name": "Simple3DCNN",
            "params": {"channels": [2]},
            "pretrained": {
                "enabled": True,
                "freeze_backbone": True,
                "unfreeze_last_stages": 1,
            },
        }
    }
    with pytest.raises(ValueError, match="unsupported for Simple3DCNN"):
        build_model(cfg, initialize_pretrained=False)


def test_partial_and_full_fine_tuning_are_rejected_as_conflicting():
    with pytest.raises(ValueError, match="conflicts with freeze_backbone=false"):
        build_model(
            config(stages=1, freeze_backbone=False), initialize_pretrained=False
        )


def test_legacy_full_fine_tuning_still_trains_every_parameter():
    model = build_model(
        config(stages=None, freeze_backbone=False), initialize_pretrained=False
    )

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_legacy_frozen_config_keeps_all_batchnorm_in_eval_mode():
    model = build_model(config(stages=None), initialize_pretrained=False)

    model.train()

    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.BatchNorm3d)
    )
