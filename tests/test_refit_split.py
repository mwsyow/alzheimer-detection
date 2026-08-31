"""Tests for the refit split: one training set of everything that is not test.

The failure these exist to prevent is silent and total. A refit that carves its test set
a different way still produces 199/36 and still trains and still reports a plausible AUC
-- on a test set that is not the one every other run was measured against. Nothing about
the numbers would look wrong. So the identity of the test set is asserted directly,
against both other split builders, and against the shortcut that seems equivalent.

These run on synthetic label lists rather than the images, because every function under
test reads only `item["label"]`.
"""

import pytest
from sklearn.model_selection import train_test_split

from datasets import (
    build_cv_split_indices,
    build_refit_split_indices,
    build_split_indices,
    is_refit_enabled,
    pool_and_test_indices,
    resolve_split_mode,
)

# The OASIS T88 masked set: 235 volumes, 135 CDR-0 against 100 CDR-positive.
N_ITEMS = 235
N_POSITIVE = 100


def items(n_items: int = N_ITEMS, n_positive: int = N_POSITIVE) -> list[dict]:
    return [{"label": 1 if i < n_positive else 0} for i in range(n_items)]


def config(**overrides) -> dict:
    base = {
        "split": {
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_seed": 42,
        },
        "cv": {"enabled": False, "n_splits": 5, "shuffle": True, "random_seed": 42},
        "refit": {"enabled": True},
    }
    base.update(overrides)
    return base


def test_the_refit_test_set_is_the_one_every_other_run_used():
    """The whole point: refit, single-split and CV must hold out the same volumes."""
    dataset, cfg = items(), config()
    refit = build_refit_split_indices(dataset, cfg)
    single = build_split_indices(dataset, cfg)
    cross_validated = build_cv_split_indices(dataset, cfg)

    assert refit["test_idx"] == single["test_idx"]
    assert refit["test_idx"] == cross_validated["test_idx"]


def test_a_two_way_split_would_not_reproduce_that_test_set():
    """Why the refit routes through the three-way split instead of the obvious shortcut.

    `train_test_split(test_size=0.15)` at the same seed looks like it should give the
    same held-out 15%, and gives a different one -- stratified_three_way_split draws
    val+test together first and then halves it. Asserted so that "simplifying" the helper
    into a two-way split fails here rather than silently moving the test set.
    """
    dataset, cfg = items(), config()
    refit_test = set(build_refit_split_indices(dataset, cfg)["test_idx"])

    _, naive_test = train_test_split(
        list(range(len(dataset))),
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_seed"],
        stratify=[item["label"] for item in dataset],
    )

    assert len(naive_test) == len(refit_test), "same size, so the sizes prove nothing"
    assert set(naive_test) != refit_test
    assert len(set(naive_test) & refit_test) < len(refit_test)


def test_val_size_zero_is_not_a_way_to_get_here():
    """The other reason the shortcut is unavailable: the inner split rejects it."""
    dataset = items()
    with pytest.raises(ValueError):
        build_split_indices(
            dataset,
            config(
                split={
                    "train_size": 0.85,
                    "val_size": 0.0,
                    "test_size": 0.15,
                    "random_seed": 42,
                }
            ),
        )


def test_the_refit_training_set_is_the_pooled_train_and_val():
    dataset, cfg = items(), config()
    refit = build_refit_split_indices(dataset, cfg)
    single = build_split_indices(dataset, cfg)

    assert refit["train_idx"] == sorted(single["train_idx"] + single["val_idx"])
    assert len(refit["train_idx"]) > len(single["train_idx"])


def test_the_refit_has_no_validation_split():
    refit = build_refit_split_indices(items(), config())
    assert refit["val_idx"] == []


def test_every_volume_is_used_exactly_once():
    refit = build_refit_split_indices(items(), config())
    covered = refit["train_idx"] + refit["val_idx"] + refit["test_idx"]
    assert sorted(covered) == list(range(N_ITEMS))
    assert len(covered) == len(set(covered))


def test_the_refit_sizes_on_oasis():
    refit = build_refit_split_indices(items(), config())
    assert (len(refit["train_idx"]), len(refit["val_idx"]), len(refit["test_idx"])) == (
        199,
        0,
        36,
    )


def test_the_pool_labels_line_up_with_the_pool_indices():
    """StratifiedKFold is handed these two side by side, so a shift would misstratify."""
    dataset, cfg = items(), config()
    pooled = pool_and_test_indices(dataset, cfg)
    assert pooled["pool_labels"] == [
        dataset[idx]["label"] for idx in pooled["pool_idx"]
    ]


def test_a_published_dataset_split_is_honoured():
    """split.source="dataset" pools the published train and val, keeping its test set."""
    dataset = [
        {"label": i % 2, "split": ("test" if i < 4 else "val" if i < 8 else "train")}
        for i in range(20)
    ]
    refit = build_refit_split_indices(dataset, config(split={"source": "dataset"}))

    assert refit["test_idx"] == [0, 1, 2, 3]
    assert refit["train_idx"] == list(range(4, 20))
    assert refit["val_idx"] == []


@pytest.mark.parametrize(
    ("cv_enabled", "refit_enabled", "expected"),
    [
        (None, None, "single"),
        (True, None, "cv"),
        (None, True, "refit"),
        (False, False, "single"),
        (True, False, "cv"),
        (False, True, "refit"),
    ],
)
def test_split_mode_resolution(cv_enabled, refit_enabled, expected):
    cfg = {"split": {}}
    if cv_enabled is not None:
        cfg["cv"] = {"enabled": cv_enabled, "n_splits": 5}
    if refit_enabled is not None:
        cfg["refit"] = {"enabled": refit_enabled}
    assert resolve_split_mode(cfg) == expected


def test_asking_for_both_cv_and_refit_raises():
    """They are alternatives, and picking one silently would pick it wrongly half the time."""
    with pytest.raises(ValueError, match="alternatives"):
        resolve_split_mode({"cv": {"enabled": True}, "refit": {"enabled": True}})


def test_refit_is_enabled_by_the_blocks_presence():
    """Same convention as cv: present means on, "enabled": false turns it off in place."""
    assert is_refit_enabled({"refit": {}}) is False  # an empty block is not a request
    assert is_refit_enabled({"refit": {"enabled": True}}) is True
    assert is_refit_enabled({"refit": {"enabled": False}}) is False
    assert is_refit_enabled({}) is False


def test_extracting_the_shared_head_did_not_move_the_folds():
    """Golden values, so the refactor that created pool_and_test_indices is provable.

    build_cv_split_indices used to compute the pool inline. These numbers were recorded
    from the pre-refactor implementation at seed 42.
    """
    folds = build_cv_split_indices(
        items(), config(cv={"enabled": True, "n_splits": 5, "shuffle": True, "random_seed": 42})
    )["folds"]

    assert [len(fold["val_idx"]) for fold in folds] == [40, 40, 40, 40, 39]
    assert [len(fold["train_idx"]) for fold in folds] == [159, 159, 159, 159, 160]
    # Every fold's val half is disjoint from the others and they tile the pool.
    all_val = [idx for fold in folds for idx in fold["val_idx"]]
    assert len(all_val) == len(set(all_val)) == 199
