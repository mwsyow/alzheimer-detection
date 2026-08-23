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
import pytest
import torch
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


def test_every_documented_key_is_accepted():
    validate_transform_config({key: False for key in KNOWN_TRANSFORM_KEYS})


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
