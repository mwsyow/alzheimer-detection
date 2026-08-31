"""The benchmark configs must differ ONLY in the model, never in the protocol.

The architecture comparison is only meaningful if every arm is evaluated identically.
That is fragile in practice: the reg and aug sweeps were silently confounded when
configs/resnet10.json was re-synced to the cluster mid-sweep, so cells within one sweep
ran different architectures. These tests make the shared protocol an assertion rather
than an intention.

Two arms deliberately tune weight_decay and label_smoothing (EfficientNet, per run
0iun6ope). Those are model hyperparameters, not protocol, so they are exempt and the
exemption is listed explicitly rather than left implicit.
"""

import glob
import json
import os

import pytest

from datasets import resolve_split_mode
from models import build_model

# The cross-validation arms, and their refit siblings. Partitioned because a refit
# deliberately differs on epochs and cv.enabled: it has no validation split, so the
# equal-draws reasoning behind the shared epoch budget does not apply to it. Each family
# is held to its own protocol, and every refit is checked against its own CV sibling.
ALL_BENCH = sorted(glob.glob("configs/bench_*.json"))
REFIT_SUFFIX = "_refit.json"
BENCH = [p for p in ALL_BENCH if not p.endswith(REFIT_SUFFIX)]
REFIT = [p for p in ALL_BENCH if p.endswith(REFIT_SUFFIX)]


def cv_sibling(refit_path):
    """The refit config's cross-validation counterpart, which it must otherwise match."""
    return refit_path[: -len(REFIT_SUFFIX)] + ".json"

# Fields that must be byte-identical across every arm.
PROTOCOL_FIELDS = [
    "epochs",
    "early_stopping",
    "dataloader",
    "cv",
    "split",
    "transforms",
    "dataset",
    "threshold",
]
# Nested fields that must match, given as dotted paths.
PROTOCOL_NESTED = [
    "optimizer.name",
    "optimizer.params.lr",
    "loss.name",
    "checkpoint.monitor",
    "checkpoint.mode",
    "checkpoint.min_delta",
]
# Deliberate per-arm exemptions: tuned model hyperparameters, not protocol.
TUNED_EXEMPT = [
    "optimizer.params.weight_decay",
    "loss.params.label_smoothing",
    # Not protocol: each arm needs a distinct run name so the 20 parent runs can be
    # attributed to arms without relying on wandb's generated animal names.
    "wandb_name",
]

EXPECTED_ARMS = {
    "configs/bench_simple3dcnn.json": "Simple3DCNN",
    "configs/bench_resnet10.json": "ResNet10",
    "configs/bench_resnet10_medicalnet.json": "ResNet10",
    "configs/bench_efficientnet_b0.json": "EfficientNetBN",
    "configs/bench_densenet121.json": "DenseNet121",
    "configs/bench_densenet121_scratch.json": "DenseNet121",
}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def dotted(config, path):
    for key in path.split("."):
        config = config[key]
    return config


def test_all_arms_present():
    assert set(BENCH) == set(EXPECTED_ARMS), "benchmark arm set changed"


@pytest.mark.parametrize("field", PROTOCOL_FIELDS)
def test_protocol_block_identical_across_arms(field):
    values = {path: load(path)[field] for path in BENCH}
    reference = values[BENCH[0]]
    differing = {p: v for p, v in values.items() if v != reference}
    assert not differing, f"{field} differs across arms: {differing}"


@pytest.mark.parametrize("path", PROTOCOL_NESTED)
def test_nested_protocol_identical_across_arms(path):
    values = {f: dotted(load(f), path) for f in BENCH}
    reference = values[BENCH[0]]
    differing = {f: v for f, v in values.items() if v != reference}
    assert not differing, f"{path} differs across arms: {differing}"


def test_the_protocol_values_are_the_ones_the_benchmark_specifies():
    """Guards the two settings the whole comparison depends on."""
    for path in BENCH:
        c = load(path)
        assert c["epochs"] == 60, (
            f"{path}: epoch budget must clear EfficientNet's slowest fold (41)"
        )
        assert c["early_stopping"]["enabled"] is False, (
            f"{path}: Best Validation AUC is a max over epochs, so early stopping "
            "would give arms unequal numbers of draws"
        )
        assert c["checkpoint"]["min_delta"] == 0.0, (
            f"{path}: 0.01 gates slow-improving arms out"
        )
        assert c["dataloader"]["batch_size"] == 8, (
            f"{path}: batch size is protocol, not tuned"
        )
        assert c["cv"]["enabled"] and c["cv"]["n_splits"] == 5, f"{path}: 5-fold CV"
        assert c["split"]["random_seed"] == 42, (
            f"{path}: the test split must never move"
        )
        assert c["transforms"]["spatial_size"] == [96, 128, 96], (
            f"{path}: shared input size"
        )


def test_no_augmentation_anywhere():
    for path in BENCH:
        on = [
            k
            for k, v in load(path)["transforms"].items()
            if k.startswith("rand_") and v is True
        ]
        assert not on, f"{path}: augmentation must be off for every arm, found {on}"


def test_only_the_exempt_fields_may_vary():
    """Whatever differs outside the model block must be a declared exemption."""

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    ref = dict(flat({k: v for k, v in load(BENCH[0]).items() if k != "model"}))
    for path in BENCH[1:]:
        cur = dict(flat({k: v for k, v in load(path).items() if k != "model"}))
        diff = {k for k in set(ref) | set(cur) if ref.get(k) != cur.get(k)}
        unexpected = diff - set(TUNED_EXEMPT)
        assert not unexpected, (
            f"{path} differs from {BENCH[0]} outside the model block: {unexpected}"
        )


@pytest.mark.parametrize("path", BENCH)
def test_each_arm_builds_the_intended_architecture(path):
    c = load(path)
    assert c["model"]["name"] == EXPECTED_ARMS[path]
    # initialize_pretrained=False so this needs no network and no checkpoint on disk
    model = build_model(c, initialize_pretrained=False)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_efficientnet_keeps_the_batchnorm_momentum_fix():
    """Not optional: at MONAI's default 0.01 this arm reports ~0.64 instead of ~0.80."""
    p = load("configs/bench_efficientnet_b0.json")["model"]["params"]
    assert p["norm"][0] == "batch"
    assert p["norm"][1]["momentum"] == 0.1


def test_resnet_arms_share_an_identical_architecture():
    """Scratch vs MedicalNet must differ ONLY in whether weights are loaded."""
    a = load("configs/bench_resnet10.json")["model"]
    b = load("configs/bench_resnet10_medicalnet.json")["model"]
    assert a["params"] == b["params"], "the two ResNet arms must be the same network"
    assert a["pretrained"]["enabled"] is False
    assert (
        b["pretrained"]["enabled"] is True and b["pretrained"]["source"] == "medicalnet"
    )


def test_every_arm_has_a_distinct_run_name():
    """20 parents with generated names is how a results table gets built from the wrong rows."""
    names = {p: load(p)["wandb_name"] for p in BENCH}
    assert all(names.values()), (
        f"wandb_name unset in {[p for p, n in names.items() if not n]}"
    )
    assert len(set(names.values())) == len(BENCH), f"duplicate run names: {names}"


def test_densenet_arms_share_an_identical_architecture():
    """The pretraining ablation: scratch vs pretrained must differ ONLY in the load.

    This bounds the exposure to the unresolved provenance of 86_acc_model.pth. If the
    two arms score the same, the checkpoint contributes nothing and any OASIS overlap
    in it is moot.
    """
    a = load("configs/bench_densenet121_scratch.json")["model"]
    b = load("configs/bench_densenet121.json")["model"]
    assert a["params"] == b["params"], "the two DenseNet arms must be the same network"
    assert a["pretrained"]["enabled"] is False
    assert b["pretrained"]["enabled"] is True
    assert b["pretrained"]["pretrained_weights_path"].endswith("86_acc_model.pth")


# ----------------------------------------------------------------- the refit siblings

# What a refit config is allowed to change relative to its CV sibling, and nothing more.
# The user's constraint is that a refit is "the same thing as turning off cv and joining
# train_size and val_size together", using whatever else the same file already says. That
# is asserted here rather than trusted.
REFIT_EXEMPT = {"epochs", "cv.enabled", "refit.enabled", "wandb_name"}


@pytest.mark.parametrize("path", REFIT)
def test_a_refit_config_matches_its_cv_sibling(path):
    sibling = cv_sibling(path)
    assert os.path.exists(sibling), f"{path} has no CV sibling at {sibling}"

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    ref = dict(flat(load(sibling)))
    cur = dict(flat(load(path)))
    diff = {k for k in set(ref) | set(cur) if ref.get(k) != cur.get(k)}
    unexpected = diff - REFIT_EXEMPT
    assert not unexpected, f"{path} differs from {sibling} beyond a refit: {unexpected}"


@pytest.mark.parametrize("path", REFIT)
def test_a_refit_config_is_a_refit(path):
    c = load(path)
    assert c["refit"]["enabled"] is True
    assert c["cv"]["enabled"] is False, "cv and refit are alternatives"
    assert c["early_stopping"]["enabled"] is False
    assert c["split"]["random_seed"] == 42, "the test split must never move"


@pytest.mark.parametrize("path", REFIT)
def test_a_refit_epoch_budget_is_not_the_cv_protocols(path):
    """60 exists to give arms equal draws at a max over epochs. A refit takes no max.

    Every architecture measured so far peaks inside its first 15 epochs, so a 60-epoch
    refit would ship a model tens of epochs past its peak with nothing held out to
    notice. The budget has to come from this arm's own cross-validation run -- its
    `CV Fold Best Epochs` summary key, or metadata["cv"]["fold_results"][k]["epoch"].
    """
    epochs = load(path)["epochs"]
    assert epochs != 60, (
        f"{path}: epochs is still the CV protocol's 60, which exists for a max over "
        "epochs that a refit does not take. Set it from this arm's CV fold best epochs."
    )
    assert 1 <= epochs <= 40, f"{path}: implausible refit budget {epochs}"


def test_every_refit_arm_has_a_distinct_run_name():
    names = {path: load(path).get("wandb_name") for path in REFIT}
    assert all(names.values()), f"wandb_name unset in {[p for p, n in names.items() if not n]}"
    assert len(set(names.values())) == len(REFIT), f"duplicate run names: {names}"


@pytest.mark.parametrize("path", REFIT)
def test_a_refit_config_resolves_to_the_refit_mode(path):
    assert resolve_split_mode(load(path)) == "refit"
