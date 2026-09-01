import csv
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from evaluate import (
    load_fold_validation_predictions,
    log_threshold_selection,
    persist_cv_threshold_selection,
    select_with_threshold_config,
    threshold_definition_from_block,
    validate_donor_refit_compatibility,
)
from metrics import (
    common_threshold_operating_point,
    compute_metrics,
    pack_predictions,
    select_cv_thresholds,
    threshold_tie_index,
    unpack_prediction_bundle,
)
from models import Simple3DCNN
from train import normalize_threshold_config


def synthetic_folds():
    # On the 0.1 grid, 0.4 is the only cut separating every positive and negative.
    return {
        1: (np.array([0, 0, 1, 1]), np.array([0.10, 0.35, 0.40, 0.90])),
        2: (np.array([0, 0, 1, 1]), np.array([0.05, 0.32, 0.40, 0.85])),
    }


def test_common_threshold_uniquely_selects_point_four():
    selected = common_threshold_operating_point(synthetic_folds(), num_thresholds=11)
    assert selected["shared_threshold"] == pytest.approx(0.4)
    assert all(
        value == pytest.approx(0.4) for value in selected["fold_thresholds"].values()
    )


def test_common_threshold_curve_has_required_diagnostics():
    selected = common_threshold_operating_point(synthetic_folds(), num_thresholds=11)
    assert len(selected["curve"]) == 11
    required = {
        "threshold",
        "mean_objective",
        "std_objective",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_sensitivity",
        "std_sensitivity",
        "mean_specificity",
        "std_specificity",
        "fold_1_balanced_accuracy",
        "fold_2_balanced_accuracy",
    }
    assert required <= set(selected["curve"][0])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("plateau_midpoint", 1),
        ("lowest", 1),
        ("highest", 5),
        ("closest_to_0_5", 5),
    ],
)
def test_all_tie_modes_are_deterministic(mode, expected):
    values = [0, 1, 1, 0, 1, 1, 0]
    thresholds = np.linspace(0, 0.6, 7)
    assert threshold_tie_index(values, thresholds, mode) == expected


def test_fixed_strategy_uses_exact_value_without_predictions():
    selected = select_cv_thresholds(
        strategy="fixed",
        fold_stored_thresholds={1: None, 2: None},
        fixed_value=0.375,
    )
    assert selected["shared_threshold"] == 0.375
    assert selected["fold_thresholds"] == {1: 0.375, 2: 0.375}
    assert "curve" not in selected


def test_final_metrics_are_inclusive_and_include_npv_and_fpr():
    metrics = compute_metrics(
        y_true=[0, 0, 1, 1], y_prob=[0.1, 0.9, 0.5, 0.2], threshold=0.5
    )
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["npv"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)


def test_npv_and_fpr_zero_denominators_are_zero():
    metrics = compute_metrics(y_true=[1, 1], y_prob=[0.8, 0.9], threshold=0.5)
    assert metrics["npv"] == 0.0
    assert metrics["fpr"] == 0.0


def test_prediction_unpacking_remains_backward_compatible():
    old = pack_predictions([0, 1], [0.2, 0.8])
    _, _, indices = unpack_prediction_bundle(old)
    assert indices is None
    new = pack_predictions([0, 1], [0.2, 0.8], indices=[4, 7])
    assert unpack_prediction_bundle(new)[2].tolist() == [4, 7]


def test_fresh_config_defaults_common_but_old_metadata_keeps_history():
    fresh = normalize_threshold_config({})["threshold"]
    old = normalize_threshold_config({}, legacy_missing_strategy=True)["threshold"]
    assert fresh == {
        "strategy": "cv_common_threshold",
        "objective": "balanced_accuracy",
        "num_thresholds": 1000,
        "tie_break": "plateau_midpoint",
    }
    assert old["strategy"] == "per_fold_youden"


def test_conflicting_canonical_and_legacy_strategies_raise():
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_threshold_config(
            {
                "threshold": {
                    "strategy": "cv_common_threshold",
                    "cv_strategy": "vertical_average",
                }
            }
        )


def write_verified_folds(tmp_path):
    items = [{"label": label} for label in [0, 1, 0, 1, 0, 1]]
    folds = [
        {"train_idx": [2, 3], "val_idx": [0, 1]},
        {"train_idx": [0, 1], "val_idx": [2, 3]},
    ]
    for fold_number, fold in enumerate(folds, 1):
        directory = tmp_path / f"split_{fold_number}"
        directory.mkdir()
        indices = fold["val_idx"]
        torch.save(
            {
                "val_predictions": pack_predictions(
                    [items[index]["label"] for index in indices],
                    [0.2, 0.8],
                    indices=indices,
                )
            },
            directory / "best_model.pth",
        )
    state = {"test_idx": [4, 5], "folds": folds}
    return items, state


def test_oof_provenance_accepts_exact_tiling_and_rejects_overlap(tmp_path):
    items, state = write_verified_folds(tmp_path)
    fold_paths = {
        fold: tmp_path / f"split_{fold}" / "best_model.pth" for fold in (1, 2)
    }
    predictions, _, provenance = load_fold_validation_predictions(
        fold_paths, cv_state=state, dataset_items=items
    )
    assert set(predictions) == {1, 2}
    assert provenance["status"] == "verified_indexed"

    state["folds"][0]["train_idx"].append(0)
    with pytest.raises(ValueError, match="overlapping train and validation"):
        load_fold_validation_predictions(
            fold_paths, cv_state=state, dataset_items=items
        )


def compatible_metadata():
    recipe = {
        "dataset": {"name": "synthetic"},
        "model": {"name": "Simple3DCNN", "params": {}},
        "transforms": {"resize": True},
        "loss": {"name": "CrossEntropyLoss", "params": {}},
        "optimizer": {"name": "AdamW", "params": {"lr": 1e-4}},
        "split": {"random_seed": 42},
        "threshold": {
            "strategy": "cv_common_threshold",
            "objective": "balanced_accuracy",
            "num_thresholds": 11,
            "tie_break": "plateau_midpoint",
        },
    }
    donor = {
        "config": {**recipe, "epochs": 60, "cv": {"enabled": True}},
        "cv": {
            "test_idx": [4, 5],
            "folds": [
                {"train_idx": [2, 3], "val_idx": [0, 1]},
                {"train_idx": [0, 1], "val_idx": [2, 3]},
            ],
        },
    }
    refit = {
        "config": {**recipe, "epochs": 7, "refit": {"enabled": True}},
        "split": {"train_idx": [0, 1, 2, 3], "val_idx": [], "test_idx": [4, 5]},
    }
    return donor, refit


def test_donor_refit_compatibility_allows_expected_workflow_differences():
    donor, refit = compatible_metadata()
    validate_donor_refit_compatibility(donor, refit)


def test_donor_refit_compatibility_rejects_recipe_and_partition_drift():
    donor, refit = compatible_metadata()
    refit["config"]["optimizer"] = {"name": "AdamW", "params": {"lr": 0.1}}
    with pytest.raises(ValueError, match="optimizer"):
        validate_donor_refit_compatibility(donor, refit)
    donor, refit = compatible_metadata()
    refit["split"]["test_idx"] = [5, 4]
    with pytest.raises(ValueError, match="test indices"):
        validate_donor_refit_compatibility(donor, refit)


def test_threshold_artifacts_round_trip_as_csv_and_json(tmp_path, monkeypatch):
    items, state = write_verified_folds(tmp_path)
    fold_paths = {
        fold: tmp_path / f"split_{fold}" / "best_model.pth" for fold in (1, 2)
    }
    predictions, _, provenance = load_fold_validation_predictions(
        fold_paths, cv_state=state, dataset_items=items
    )
    threshold_config = {
        "strategy": "cv_common_threshold",
        "objective": "balanced_accuracy",
        "num_thresholds": 11,
        "tie_break": "plateau_midpoint",
        "fpr_rounding": "at_least",
        "fpr_grid": 101,
        "threshold_grid": 0,
        "value": None,
    }
    selection = select_with_threshold_config(
        fold_paths, threshold_config, fold_predictions=predictions
    )
    stored = persist_cv_threshold_selection(
        tmp_path,
        selection,
        threshold_definition_from_block(threshold_config),
        provenance,
    )
    with (tmp_path / "threshold_selection.json").open() as handle:
        on_disk = json.load(handle)
    with (tmp_path / "threshold_curve.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert on_disk["threshold"] == selection["shared_threshold"]
    assert stored["provenance"]["status"] == "verified_indexed"
    assert len(rows) == 11

    class Run:
        def __init__(self):
            self.summary = {}
            self.logged = {}

        def log(self, values):
            self.logged.update(values)

    monkeypatch.setattr("evaluate.wandb.Table", lambda **kwargs: kwargs)
    run = Run()
    log_threshold_selection(run, selection)
    assert "CV Threshold Sweep" in run.logged


def test_simple3dcnn_calls_only_the_requested_initializers():
    with (
        patch("torch.nn.init.kaiming_normal_") as kaiming,
        patch("torch.nn.init.xavier_uniform_") as xavier,
        patch("torch.nn.init.ones_") as ones,
        patch("torch.nn.init.zeros_") as zeros,
    ):
        Simple3DCNN(channels=[4, 8], use_batch_norm=True)
    assert kaiming.call_count == 2
    assert xavier.call_count == 1
    # BatchNorm's constructor also uses ones_/zeros_ before our explicit pass.
    assert ones.call_count >= 2
    assert zeros.call_count >= 5
