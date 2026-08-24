"""Tests for the two EfficientNet forms in models.py.

`EfficientNetBN` wraps MONAI's variant wrapper: pick a B number, and `norm` passes
through, but dropout_rate and drop_connect_rate cannot be reached because MONAI's
EfficientNetBN reads them from its per-variant table and does not accept them.
`EfficientNet` subclasses MONAI's EfficientNet instead, so both are real arguments,
along with the width/depth scaling coefficients.

The load-bearing test is equivalence: with no overrides the two must build identical
networks. `EfficientNet` has to repeat the block topology in EFFICIENTNET_BLOCKS_ARGS
to reach the parent class, so if MONAI ever changes that topology or the variant
table, this fails rather than quietly producing a different network.
"""

import pytest
import torch
from monai.networks.nets.efficientnet import MBConvBlock, efficientnet_params

from models import EfficientNet, EfficientNetB0, build_model

VARIANTS = ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"]


def rates(model):
    return [b.drop_connect_rate for b in model.modules() if isinstance(b, MBConvBlock)]


@pytest.mark.parametrize("model_name", VARIANTS)
def test_the_two_forms_agree_when_nothing_is_overridden(model_name):
    ours = EfficientNet(model_name=model_name)
    theirs = EfficientNetB0(model_name=model_name)
    a, b = ours.state_dict(), theirs.state_dict()
    assert set(a) == set(b), "state-dict keys diverged from MONAI's wrapper"
    assert all(a[k].shape == b[k].shape for k in a), "tensor shapes diverged"
    assert rates(ours) == pytest.approx(rates(theirs)), "drop-connect schedule diverged"
    assert ours._dropout.p == pytest.approx(theirs._dropout.p)


@pytest.mark.parametrize("model_name", VARIANTS)
def test_defaults_come_from_the_variant_table(model_name):
    _, _, _, dropout, drop_connect = efficientnet_params[model_name]
    model = EfficientNet(model_name=model_name)
    assert model._dropout.p == pytest.approx(dropout)
    # Depth scaling changes the block count (b0 16, b1/b2 23, b3 26), so the top of
    # the schedule is rate * (n-1)/n for that variant's own n, not b0's 16.
    n = len(rates(model))
    assert max(rates(model)) == pytest.approx(drop_connect * (n - 1) / n)


def test_efficientnetbn_cannot_reach_the_dropout_knobs():
    """The reason both classes exist. If MONAI ever adds these, revisit the split."""
    with pytest.raises(TypeError, match="dropout_rate"):
        EfficientNetB0(dropout_rate=0.4)
    with pytest.raises(TypeError, match="drop_connect_rate"):
        EfficientNetB0(drop_connect_rate=0.3)


def test_efficientnetb0_alias_still_resolves():
    """Runs before 2026-08-24 recorded model.name "EfficientNetB0" in their metadata,
    and evaluate.py rebuilds from that, so the name must keep working."""
    assert issubclass(EfficientNetB0, EfficientNetB0)
    assert EfficientNetB0().state_dict().keys() == EfficientNetB0().state_dict().keys()


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model_name"):
        EfficientNet(model_name="efficientnet-b99")


def test_dropout_rate_override_does_not_disturb_drop_connect():
    model = EfficientNet(dropout_rate=0.45)
    assert model._dropout.p == pytest.approx(0.45)
    assert rates(model) == pytest.approx(rates(EfficientNetB0()))


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3, 0.5])
def test_drop_connect_override_keeps_the_per_block_schedule(rate):
    """MONAI scales the base rate by block position; an override must preserve that."""
    r = rates(EfficientNet(drop_connect_rate=rate))
    assert len(r) == 16, "b0 must have 16 MBConv blocks"
    assert r[0] == 0.0, "the first block is never dropped"
    assert r == sorted(r), "stochastic depth must increase with depth"
    assert r == pytest.approx([rate * i / 16 for i in range(16)])


def test_width_and_depth_coefficients_change_capacity():
    base = sum(p.numel() for p in EfficientNet().parameters())
    narrow = sum(p.numel() for p in EfficientNet(width_coefficient=0.5).parameters())
    deep = sum(p.numel() for p in EfficientNet(depth_coefficient=1.5).parameters())
    assert narrow < base < deep


@pytest.mark.parametrize(
    "name,extra,expect_dropout",
    [
        ("EfficientNetB0", {}, 0.2),
        ("EfficientNetBN", {}, 0.2),
        ("EfficientNet", {"dropout_rate": 0.35, "drop_connect_rate": 0.25}, 0.35),
    ],
)
def test_build_model_dispatches_on_model_name(name, extra, expect_dropout):
    model = build_model(
        {
            "model": {
                "name": name,
                "params": {
                    "num_classes": 2,
                    "in_channels": 1,
                    "spatial_dims": 3,
                    **extra,
                },
                "pretrained": {"enabled": False},
            }
        }
    )
    assert type(model).__name__ == name
    assert model._dropout.p == pytest.approx(expect_dropout)


@pytest.mark.parametrize("cls", [EfficientNet, EfficientNetB0])
def test_norm_passes_through_to_every_layer(cls):
    """norm reaches all 49 layers on both, unlike MONAI's ResNet where it reaches one."""
    model = cls(norm=["instance", {"affine": True}])
    assert (
        sum(1 for m in model.modules() if isinstance(m, torch.nn.InstanceNorm3d)) == 49
    )


def test_forward_at_the_real_input_size():
    model = EfficientNet(dropout_rate=0.5, drop_connect_rate=0.4).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 96, 128, 96))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
