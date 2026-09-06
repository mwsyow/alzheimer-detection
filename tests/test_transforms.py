"""Tests for the augmentation pipeline build_transforms assembles.

Two failure modes are guarded here. First, a misnamed transforms key: unlike
model.params, which reaches a constructor and raises TypeError, an unrecognised
transform key would be a silent no-op -- the sweep completes and every trial is
identical, with nothing in the logs saying why. Second, ordering: the pipeline puts
spatial augmentation before normalisation and splits intensity augmentation either
side of it, and the claim that moving RandRotate90d earlier is a no-op for existing
configs is asserted rather than assumed.
"""

import numpy as np
import nibabel as nib
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import NormalizeIntensityd, RandRotate90d

from datasets import (
    KNOWN_TRANSFORM_KEYS,
    DatasetBackend,
    build_transforms,
    validate_transform_config,
)

BASE = {
    "resize": False,
    "normalize_intensity": True,
    "normalize_nonzero": True,
    "normalize_channel_wise": True,
}


class InMemoryBackend(DatasetBackend):
    """A backend whose "image" is already materialised, so no file touches disk."""

    name = "in-memory"

    def build_items(self):
        return []


def names(transform_config, mode="train"):
    compose = build_transforms(
        InMemoryBackend({}), {"transforms": transform_config}, mode
    )
    return [type(t).__name__ for t in compose.transforms]


def test_unknown_transform_key_raises():
    with pytest.raises(ValueError, match="rand_flipp"):
        validate_transform_config({**BASE, "rand_flipp": True})


def test_unknown_key_names_the_known_ones():
    with pytest.raises(ValueError, match="rand_bias_field"):
        validate_transform_config({**BASE, "totally_made_up": 1})


def test_invalid_intensity_order_raises():
    with pytest.raises(ValueError, match="intensity_order"):
        validate_transform_config({**BASE, "intensity_order": "normalize_twice"})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"spacing": True}, "pixdim"),
        ({"orientation": True}, "axcodes"),
    ],
)
def test_enabled_spatial_metadata_transform_requires_its_setting(config, message):
    with pytest.raises(ValueError, match=message):
        validate_transform_config({**BASE, **config})


def test_every_documented_key_is_accepted():
    config = {key: False for key in KNOWN_TRANSFORM_KEYS}
    config.update(
        {
            "pixdim": [1.0, 1.0, 1.0],
            "axcodes": "SAR",
            "intensity_order": "scale_then_normalize",
        }
    )
    validate_transform_config(config)


TOGGLES = (
    "rand_flip",
    "rand_affine",
    "rand_rotate90",
    "rand_bias_field",
    "rand_gaussian_noise",
    "rand_scale_intensity",
    "rand_shift_intensity",
)


def test_every_toggle_is_a_known_key():
    assert set(TOGGLES) <= KNOWN_TRANSFORM_KEYS


def test_augmentation_is_train_only():
    config = {**BASE, **{key: True for key in TOGGLES}}
    train, val = names(config, "train"), names(config, "val")
    assert not [n for n in val if n.startswith("Rand")]
    assert [n for n in train if n.startswith("Rand")]


def test_spatial_augmentation_precedes_normalisation():
    order = names({**BASE, "rand_flip": True, "rand_affine": True})
    assert order.index("RandFlipd") < order.index("NormalizeIntensityd")
    assert order.index("RandAffined") < order.index("NormalizeIntensityd")


def test_spacing_orientation_and_resize_precede_intensity_transforms():
    order = names(
        {
            **BASE,
            "spacing": True,
            "pixdim": [1.0, 1.0, 1.0],
            "orientation": True,
            "axcodes": "SAR",
            "resize": True,
            "spatial_size": [32, 256, 256],
            "scale_intensity": True,
            "intensity_order": "normalize_then_scale",
        },
        mode="val",
    )
    assert order[:6] == [
        "EnsureChannelFirstd",
        "Spacingd",
        "Orientationd",
        "Resized",
        "NormalizeIntensityd",
        "ScaleIntensityd",
    ]


def test_las_volume_is_reoriented_to_sar_using_its_affine():
    volume = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    las_affine = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transform = build_transforms(
        InMemoryBackend({}),
        {
            "transforms": {
                "resize": False,
                "normalize_intensity": False,
                "orientation": True,
                "axcodes": "SAR",
            }
        },
        "val",
    )

    result = transform({"image": MetaTensor(volume, affine=las_affine), "label": 0})[
        "image"
    ]

    assert result.shape == (1, 4, 3, 2)
    assert nib.aff2axcodes(result.affine.numpy()) == ("S", "A", "R")
    expected = volume.permute(2, 1, 0).flip(2).unsqueeze(0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("intensity_order", "expected"),
    [
        ("scale_then_normalize", ["ScaleIntensityd", "NormalizeIntensityd"]),
        ("normalize_then_scale", ["NormalizeIntensityd", "ScaleIntensityd"]),
    ],
)
def test_intensity_order_is_configurable(intensity_order, expected):
    order = names(
        {
            **BASE,
            "scale_intensity": True,
            "intensity_order": intensity_order,
        }
    )
    intensity = [name for name in order if name in expected]
    assert intensity == expected


def test_scale_channel_wise_reaches_monai_transform():
    compose = build_transforms(
        InMemoryBackend({}),
        {
            "transforms": {
                **BASE,
                "scale_intensity": True,
                "scale_channel_wise": True,
            }
        },
        "val",
    )
    scale = next(t for t in compose.transforms if type(t).__name__ == "ScaleIntensityd")
    assert scale.scaler.channel_wise is True


def test_bias_field_before_normalisation_noise_after():
    order = names({**BASE, "rand_bias_field": True, "rand_gaussian_noise": True})
    assert order.index("RandBiasFieldd") < order.index("NormalizeIntensityd")
    assert order.index("RandGaussianNoised") > order.index("NormalizeIntensityd")


def test_config_without_new_keys_builds_the_original_pipeline():
    """A config predating these keys must be unaffected."""
    assert names({**BASE, "rand_rotate90": True}) == [
        "EnsureChannelFirstd",
        "RandRotate90d",
        "NormalizeIntensityd",
        "EnsureTyped",
        "EnsureTyped",
    ]


def test_rotate90_commutes_with_normalisation():
    """Why RandRotate90d could be moved ahead of NormalizeIntensityd.

    A 90-degree rotation permutes voxels, so the nonzero set NormalizeIntensityd
    reduces over is unchanged and the two operations commute exactly. Without this,
    reordering would silently alter every run that used rand_rotate90.
    """
    rng = np.random.default_rng(0)
    volume = rng.random((1, 8, 10, 8), dtype=np.float32)
    volume[volume < 0.3] = 0.0  # a background to exercise nonzero=True

    rotate = RandRotate90d(keys=["image"], prob=1.0, spatial_axes=(0, 2))
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)

    rotate.set_random_state(seed=0)
    rotate_then_normalize = normalize(rotate({"image": torch.tensor(volume)}))["image"]
    rotate.set_random_state(seed=0)
    normalize_then_rotate = rotate(normalize({"image": torch.tensor(volume)}))["image"]

    torch.testing.assert_close(
        torch.as_tensor(rotate_then_normalize), torch.as_tensor(normalize_then_rotate)
    )
