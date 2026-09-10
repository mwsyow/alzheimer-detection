import numpy as np
import pytest

from metrics import per_fold_threshold_operating_point


def test_each_cut_uses_only_its_own_fold():
    folds = {1: ([0, 1], [0.1, 0.3]), 2: ([0, 1], [0.7, 0.9])}
    result = per_fold_threshold_operating_point(folds, num_thresholds=101)
    assert result["shared_threshold"] is None
    assert 0.1 < result["fold_thresholds"][1] <= 0.3
    assert 0.7 < result["fold_thresholds"][2] <= 0.9
    folds[2] = ([0, 1], [0.01, 0.02])
    changed = per_fold_threshold_operating_point(folds, num_thresholds=101)
    assert changed["fold_thresholds"][1] == result["fold_thresholds"][1]
    assert changed["fold_thresholds"][2] != result["fold_thresholds"][2]


@pytest.mark.parametrize("objective", ["balanced_accuracy", "f1"])
def test_selected_cut_maximizes_configured_objective(objective):
    result = per_fold_threshold_operating_point(
        {1: ([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.8])},
        objective=objective, num_thresholds=21, tie_break="lowest",
    )
    rows = result["curve"]
    maximum = max(row["mean_objective"] for row in rows)
    best = next(row for row in rows if np.isclose(row["mean_objective"], maximum))
    assert result["fold_thresholds"][1] == best["threshold"]
