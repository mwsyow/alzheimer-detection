"""Tests for choosing one decision threshold from a cross-validation run.

The property that matters most here is monotone invariance: vertical averaging is worth
its extra complexity only because rescaling one fold's probabilities cannot move the
shared operating point. If a refactor breaks that, the complexity stays and the benefit
silently goes, so it is asserted directly rather than left to inspection.
"""

import argparse
import json

import numpy as np
import pytest
import torch

from evaluate import (
    discover_fold_checkpoints,
    ensemble_operating_point,
    inherited_operating_point,
    load_stored_thresholds,
    resolve_eval_config,
    stored_threshold,
)
from metrics import (
    DEFAULT_THRESHOLD,
    FPR_ROUNDING_POLICIES,
    balanced_accuracy_curve,
    pack_predictions,
    plateau_argmax,
    ranking_metrics,
    select_cv_thresholds,
    threshold_at_fpr,
    threshold_average_operating_point,
    unpack_predictions,
    usable_folds,
    vertical_average_operating_point,
)

N_PER_CLASS = 20
# Slope and intercept per fold, so the folds are deliberately calibrated differently.
FOLD_STYLES = ((1.6, 0.0), (1.1, -0.35), (1.4, 0.25))


def make_fold(slope, intercept, seed, n=N_PER_CLASS):
    generator = np.random.default_rng(seed)
    y_true = np.r_[np.zeros(n, int), np.ones(n, int)]
    latent = y_true * 1.45 + generator.normal(size=len(y_true))
    return y_true, 1.0 / (1.0 + np.exp(-(slope * latent + intercept)))


def squash(y_prob, temperature):
    """Rescale probabilities in logit space, leaving their ordering untouched."""
    logit = np.log(y_prob / (1.0 - y_prob))
    return 1.0 / (1.0 + np.exp(-logit / temperature))


@pytest.fixture
def folds():
    return {
        index + 1: make_fold(slope, intercept, 100 + index)
        for index, (slope, intercept) in enumerate(FOLD_STYLES)
    }


@pytest.fixture
def single_class_fold():
    generator = np.random.default_rng(0)
    return np.zeros(10, int), generator.random(10)


# --------------------------------------------------------------------------- invariance


def test_vertical_average_survives_rescaling_one_fold(folds):
    """The whole justification for vertical averaging, asserted."""
    before = vertical_average_operating_point(folds)

    rescaled = dict(folds)
    rescaled[3] = (folds[3][0], squash(folds[3][1], 4.0))
    after = vertical_average_operating_point(rescaled)

    assert after["target_fpr"] == before["target_fpr"]
    assert after["mean_tpr_at_target"] == before["mean_tpr_at_target"]
    assert after["mean_balanced_accuracy"] == before["mean_balanced_accuracy"]

    # Only the rescaled fold's own cut moves, absorbing the change.
    assert after["fold_thresholds"][1] == before["fold_thresholds"][1]
    assert after["fold_thresholds"][2] == before["fold_thresholds"][2]
    assert after["fold_thresholds"][3] != before["fold_thresholds"][3]
    # And it still lands on the same point of its own ROC curve.
    assert after["fold_realised_fpr"][3] == before["fold_realised_fpr"][3]
    assert after["fold_realised_tpr"][3] == before["fold_realised_tpr"][3]


def test_threshold_average_does_not_survive_rescaling_one_fold(folds):
    """The contrast that motivates preferring vertical averaging on this project.

    Not a defect in threshold averaging -- it compares probabilities across folds by
    design, so it is only sound once the folds are calibrated alike.
    """
    before = threshold_average_operating_point(folds)

    rescaled = dict(folds)
    rescaled[3] = (folds[3][0], squash(folds[3][1], 4.0))
    after = threshold_average_operating_point(rescaled)

    assert after["shared_threshold"] != before["shared_threshold"]


# ------------------------------------------------------------------------ plateau_argmax


def test_plateau_argmax_takes_the_middle_not_the_left_edge():
    values = np.array([0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3])
    assert plateau_argmax(values) == 4
    assert int(np.argmax(values)) == 2, "np.argmax picks the left edge; that is the bug"


def test_plateau_argmax_uses_the_longest_run_when_maximisers_are_split():
    values = np.array([0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
    index = plateau_argmax(values)
    assert index == 3
    # A median over all maximiser indices would not itself have to be a maximiser.
    assert values[index] == values.max()


def test_plateau_argmax_rejects_empty():
    with pytest.raises(ValueError):
        plateau_argmax([])


# ----------------------------------------------------------------------- threshold_at_fpr


@pytest.mark.parametrize("target", [0.05, 0.1, 0.2, 0.35, 0.5])
def test_realised_fpr_is_quantised_and_never_below_target(folds, target):
    """With n negatives only multiples of 1/n exist, so a fold steps rather than lands."""
    for y_true, y_prob in folds.values():
        _, realised, _ = threshold_at_fpr(y_true, y_prob, target, rounding="at_least")
        assert realised >= target - 1e-9
        assert realised * N_PER_CLASS == pytest.approx(round(realised * N_PER_CLASS))


def test_at_most_never_exceeds_target(folds):
    for y_true, y_prob in folds.values():
        _, realised, _ = threshold_at_fpr(y_true, y_prob, 0.22, rounding="at_most")
        assert realised <= 0.22 + 1e-9


def test_applying_the_cut_reproduces_the_reported_rates(folds):
    """The returned rates must describe what the threshold actually does."""
    for y_true, y_prob in folds.values():
        cut, realised_fpr, realised_tpr = threshold_at_fpr(y_true, y_prob, 0.2)
        predicted = y_prob >= cut
        assert predicted[y_true == 0].mean() == pytest.approx(realised_fpr)
        assert predicted[y_true == 1].mean() == pytest.approx(realised_tpr)


def test_threshold_at_fpr_raises_on_single_class(single_class_fold):
    with pytest.raises(ValueError, match="both classes"):
        threshold_at_fpr(*single_class_fold, 0.2)


def test_threshold_at_fpr_rejects_unknown_rounding(folds):
    with pytest.raises(ValueError, match="fpr_rounding"):
        threshold_at_fpr(*folds[1], 0.2, rounding="sideways")


@pytest.mark.parametrize("rounding", FPR_ROUNDING_POLICIES)
def test_degenerate_targets_stay_in_range(folds, rounding):
    for target in (0.0, 1.0):
        cut, realised, _ = threshold_at_fpr(*folds[1], target, rounding=rounding)
        assert 0.0 <= cut <= 1.0
        assert 0.0 <= realised <= 1.0


def test_identical_probabilities_do_not_crash():
    y_true = np.r_[np.zeros(10, int), np.ones(10, int)]
    cut, realised, _ = threshold_at_fpr(y_true, np.full(20, 0.4), 0.5)
    assert np.isfinite(cut)
    assert 0.0 <= realised <= 1.0


# ------------------------------------------------------------------- fold bookkeeping


def test_usable_folds_separates_single_class(folds, single_class_fold):
    mixed = {**folds, 4: single_class_fold}
    usable, skipped = usable_folds(mixed)
    assert sorted(usable) == [1, 2, 3]
    assert skipped == [4]


def test_skipped_folds_get_the_default_threshold(folds, single_class_fold):
    result = vertical_average_operating_point({**folds, 4: single_class_fold})
    assert result["skipped_folds"] == [4]
    assert result["fold_thresholds"][4] == DEFAULT_THRESHOLD
    assert 4 not in result["fold_realised_fpr"]


def test_averaging_needs_at_least_two_usable_folds(folds):
    with pytest.raises(ValueError, match="at least 2 folds"):
        vertical_average_operating_point({1: folds[1]})
    with pytest.raises(ValueError, match="at least 2 folds"):
        threshold_average_operating_point({1: folds[1]})


def test_vertical_average_rejects_a_degenerate_grid(folds):
    with pytest.raises(ValueError, match="fpr_grid"):
        vertical_average_operating_point(folds, fpr_grid=1)


# ------------------------------------------------------------------------- the dispatcher


def test_per_fold_youden_reuses_stored_thresholds_without_predictions():
    """What keeps runs trained before predictions were stored re-evaluable."""
    result = select_cv_thresholds(
        strategy="per_fold_youden", fold_stored_thresholds={1: 0.31, 2: 0.87}
    )
    assert result["fold_thresholds"] == {1: 0.31, 2: 0.87}
    assert result["shared_threshold"] is None


def test_dispatcher_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="cv_strategy"):
        select_cv_thresholds(strategy="vibes")


def test_averaging_strategies_require_predictions():
    with pytest.raises(ValueError, match="validation predictions"):
        select_cv_thresholds(strategy="vertical_average")


def test_per_fold_youden_requires_stored_thresholds():
    with pytest.raises(ValueError, match="stored per fold"):
        select_cv_thresholds(strategy="per_fold_youden")


@pytest.mark.parametrize("strategy", ["vertical_average", "threshold_average"])
def test_every_strategy_covers_every_fold(folds, strategy):
    result = select_cv_thresholds(strategy=strategy, fold_predictions=folds)
    assert set(result["fold_thresholds"]) == set(folds)
    assert result["strategy"] == strategy


def test_threshold_average_shares_one_cut(folds):
    result = select_cv_thresholds(strategy="threshold_average", fold_predictions=folds)
    assert len(set(result["fold_thresholds"].values())) == 1
    assert result["shared_threshold"] == next(iter(result["fold_thresholds"].values()))


# ------------------------------------------------------------- balanced_accuracy_curve


def test_balanced_accuracy_curve_matches_a_direct_computation(folds):
    y_true, y_prob = folds[1]
    grid = np.linspace(0.0, 1.0, 21)
    curve = balanced_accuracy_curve(y_true, y_prob, grid)
    for index, threshold in enumerate(grid):
        predicted = y_prob >= threshold
        expected = (
            predicted[y_true == 1].mean() + (1.0 - predicted[y_true == 0].mean())
        ) / 2.0
        assert curve[index] == pytest.approx(expected)


# ------------------------------------------------------------------------ serialisation


def test_packed_predictions_survive_a_weights_only_load(folds, tmp_path):
    """Checkpoints load without weights_only=False, which rejects numpy arrays."""
    y_true, y_prob = folds[1]
    checkpoint = {
        "epoch": 3,
        "threshold": 0.5,
        "fold": 1,
        "model_state_dict": {"weight": torch.zeros(2)},
        "val_predictions": pack_predictions(y_true, y_prob),
    }
    path = tmp_path / "best_model.pth"
    torch.save(checkpoint, path)

    loaded = torch.load(path, map_location="cpu")
    restored_true, restored_prob = unpack_predictions(loaded["val_predictions"])
    assert np.array_equal(restored_true, y_true)
    assert restored_prob == pytest.approx(y_prob, abs=1e-6)


def test_numpy_predictions_would_break_a_weights_only_load(folds, tmp_path):
    """Guards the reason pack_predictions exists at all."""
    path = tmp_path / "numpy.pth"
    torch.save({"val_predictions": {"y_true": folds[1][0]}}, path)
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu")


def test_packing_does_not_carry_the_whole_source_buffer(tmp_path):
    """torch.as_tensor shares a numpy buffer and torch.save writes the whole storage."""
    big = np.arange(200_000, dtype=np.float64)
    packed = pack_predictions(big[:10].astype(int), big[:10])
    path = tmp_path / "packed.pth"
    torch.save(packed, path)
    assert path.stat().st_size < 5_000


def test_unpack_accepts_numpy_as_well_as_tensors(folds):
    y_true, y_prob = folds[1]
    restored_true, restored_prob = unpack_predictions({"y_true": y_true, "y_prob": y_prob})
    assert np.array_equal(restored_true, y_true)
    assert restored_prob == pytest.approx(y_prob)


# ----------------------------------------------------------------------- ranking_metrics


def test_ranking_metrics_reports_no_threshold(folds):
    y_true, y_prob = folds[1]
    metrics = ranking_metrics(y_true, y_prob)
    assert "threshold" not in metrics
    for thresholded in ("balanced_accuracy", "f1", "sensitivity", "specificity"):
        assert thresholded not in metrics
    assert metrics["n"] == len(y_true)


def test_ranking_metrics_returns_none_for_undefined_scores(single_class_fold):
    """0.0 from average_precision_score on an all-negative split is not a score."""
    metrics = ranking_metrics(*single_class_fold)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None


# ------------------------------------------------------------------- eval-time config


@pytest.fixture
def trained_metadata():
    return {
        "run_id": "abc123",
        "config": {
            "model": {"name": "DenseNet121", "params": {"num_classes": 2}},
            "transforms": {"spatial_size": [64, 64, 64]},
            "dataloader": {"batch_size": 8, "num_workers": 0},
            "device": "cuda",
            "threshold": {"cv_strategy": "per_fold_youden"},
        },
    }


def write_config(tmp_path, config):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(config))
    return path


def test_eval_config_without_a_file_is_the_runs_own(trained_metadata):
    config, report = resolve_eval_config(trained_metadata, None)
    assert config == trained_metadata["config"]
    assert report["overridden"] == []


def test_eval_config_applies_allowlisted_keys(trained_metadata, tmp_path):
    path = write_config(tmp_path, {"threshold": {"cv_strategy": "vertical_average"}})
    config, report = resolve_eval_config(trained_metadata, path)
    assert config["threshold"]["cv_strategy"] == "vertical_average"
    assert report["overridden"] == ["threshold"]


def test_eval_config_refuses_to_change_how_the_model_computes(trained_metadata, tmp_path):
    """The failure this allowlist exists to prevent: silently rescored on other inputs."""
    path = write_config(
        tmp_path,
        {
            "transforms": {"spatial_size": [32, 32, 32]},
            "model": {"name": "Simple3DCNN", "params": {}},
        },
    )
    config, report = resolve_eval_config(trained_metadata, path)
    assert config["transforms"]["spatial_size"] == [64, 64, 64]
    assert config["model"]["name"] == "DenseNet121"
    assert sorted(report["ignored_differing"]) == ["model", "transforms"]


def test_eval_config_stays_quiet_when_ignored_keys_match(trained_metadata, tmp_path, capsys):
    """Passing the very config that trained the run must not produce noise."""
    path = write_config(tmp_path, dict(trained_metadata["config"]))
    _, report = resolve_eval_config(trained_metadata, path)
    assert report["ignored_differing"] == []
    assert capsys.readouterr().out == ""


def test_eval_config_merges_rather_than_replaces_a_block(trained_metadata, tmp_path):
    path = write_config(tmp_path, {"dataloader": {"batch_size": 64}})
    config, _ = resolve_eval_config(trained_metadata, path)
    assert config["dataloader"]["batch_size"] == 64
    assert config["dataloader"]["num_workers"] == 0


def test_eval_config_does_not_mutate_the_metadata(trained_metadata, tmp_path):
    path = write_config(tmp_path, {"device": "cpu"})
    resolve_eval_config(trained_metadata, path)
    assert trained_metadata["config"]["device"] == "cuda"


# ------------------------------------------------------- the cut a single model inherits


def write_cv_run(tmp_path, folds, thresholds=None, config=None):
    """A minimal cross-validation run directory: per-fold checkpoints plus metadata."""
    for fold, (y_true, y_prob) in folds.items():
        fold_dir = tmp_path / f"split_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        payload = {"fold": fold, "val_predictions": pack_predictions(y_true, y_prob)}
        if thresholds is not None:
            payload["threshold"] = thresholds[fold]
        torch.save(payload, fold_dir / "best_model.pth")

    torch.save(
        {
            "run_id": "donor001",
            "config": config or {"threshold": {"cv_strategy": "vertical_average"}},
            "cv": {"test_idx": [0, 1, 2], "folds": []},
        },
        tmp_path / "metadata.pth",
    )
    return tmp_path


def threshold_args(**overrides):
    defaults = {
        "threshold": None,
        "threshold_from": None,
        "threshold_strategy": None,
        "fpr_rounding": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_the_inherited_cut_is_the_ensembles_own(tmp_path, folds):
    """Benchmark 2 must score the single model at exactly benchmark 1's operating point.

    Computed here the long way -- select, then take the ensemble's cut -- and compared
    against what --threshold-from produces, so the two benchmarks cannot drift apart.
    """
    run_dir = write_cv_run(tmp_path, folds)
    # From the round-tripped predictions, not the float64 originals: pack_predictions
    # stores y_prob as float32, and the real pipeline reads it back from disk on both
    # sides. Comparing against the in-memory arrays would fail at about 1e-8 and say
    # nothing about whether the two benchmarks agree.
    round_tripped = {
        fold: unpack_predictions(
            torch.load(run_dir / f"split_{fold}" / "best_model.pth", weights_only=False)[
                "val_predictions"
            ]
        )
        for fold in folds
    }
    selection = vertical_average_operating_point(round_tripped)
    expected, phrase = ensemble_operating_point(selection, selection["fold_thresholds"])

    inherited, record = inherited_operating_point(run_dir, threshold_args())

    assert inherited == expected
    assert record["threshold_source"] == phrase


def test_the_inherited_cut_records_where_it_came_from(tmp_path, folds):
    """An operating point borrowed from another run is only defensible if it is traceable."""
    run_dir = write_cv_run(tmp_path, folds)
    _, record = inherited_operating_point(run_dir, threshold_args())

    assert record["strategy"] == "inherited_from_cv"
    assert record["source_run_id"] == "donor001"
    assert record["folds_used"] == sorted(folds)
    assert record["cv_selection"]["strategy"] == "vertical_average"
    assert "target_fpr" in record["cv_selection"]


def test_the_inherited_cut_uses_the_donors_own_strategy(tmp_path, folds):
    """The donor's config decides, so the number reproduces that run's own summary.json."""
    run_dir = write_cv_run(
        tmp_path, folds, config={"threshold": {"cv_strategy": "threshold_average"}}
    )
    _, record = inherited_operating_point(run_dir, threshold_args())
    assert record["cv_selection"]["strategy"] == "threshold_average"


def test_inheriting_from_a_directory_with_no_folds_raises(tmp_path):
    torch.save({"run_id": "x", "config": {}}, tmp_path / "metadata.pth")
    with pytest.raises(FileNotFoundError, match="threshold-from"):
        inherited_operating_point(tmp_path, threshold_args())


# ------------------------------------------------- a missing threshold fails out loud


def test_a_checkpoint_without_a_threshold_raises_rather_than_defaulting(tmp_path):
    """0.5 used to be substituted silently, reporting an untuned cut as validated."""
    path = tmp_path / "last.pth"
    with pytest.raises(ValueError) as excinfo:
        stored_threshold({"epoch": 3}, path)

    message = str(excinfo.value)
    assert str(path) in message
    for route in ("--threshold", "--threshold-from", "--threshold-strategy"):
        assert route in message, f"the error should name {route}"


def test_a_stored_threshold_is_still_honoured(tmp_path):
    assert stored_threshold({"threshold": 0.37}) == pytest.approx(0.37)


def test_per_fold_youden_lists_every_fold_missing_a_threshold(tmp_path, folds):
    """A resumed run can mix folds from before and after, so one name is not enough."""
    run_dir = write_cv_run(tmp_path, folds, thresholds={1: 0.4, 2: None, 3: None})
    fold_paths = discover_fold_checkpoints(run_dir)

    with pytest.raises(ValueError) as excinfo:
        load_stored_thresholds(fold_paths)

    message = str(excinfo.value)
    assert "split_2" in message and "split_3" in message
    assert "split_1" not in message


def test_per_fold_youden_still_works_for_a_legacy_run(tmp_path, folds):
    run_dir = write_cv_run(tmp_path, folds, thresholds={1: 0.4, 2: 0.5, 3: 0.6})
    stored = load_stored_thresholds(discover_fold_checkpoints(run_dir))
    assert stored == {1: 0.4, 2: 0.5, 3: 0.6}
