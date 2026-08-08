"""Tests for what a cross-validation evaluation reports and to where.

The failure mode these guard against is a summary that reads as a result but is not one:
bookkeeping averaged because it happened to be numeric, or a per-fold value left sitting
in a run's summary where it looks run-level. Those are invisible in the numbers
themselves, so they are asserted rather than eyeballed.
"""

import numpy as np
import pytest

import evaluate

from evaluate import (
    COMPARISON_COLUMNS,
    NON_RESULT_METRICS,
    build_comparison_rows,
    cv_validation_aggregate,
    ensemble_operating_point,
    log_cv_to_wandb,
    prune_stale_summary_keys,
)


class FakeSummary(dict):
    """A wandb summary is a MutableMapping whose deletions reach the server."""


class FakeRun:
    def __init__(self, summary=None):
        self.summary = FakeSummary(summary or {})
        self.logged = {}
        self.finished = False

    def log(self, values):
        self.logged.update(values)

    def finish(self):
        self.finished = True


@pytest.fixture
def run(monkeypatch):
    """A stand-in for the resumed parent run, with wandb.init pointed at it."""
    fake = FakeRun()
    monkeypatch.setattr(evaluate.wandb, "init", lambda **kwargs: fake)
    return fake


@pytest.fixture
def summary():
    validation = {"roc_auc": 0.87, "accuracy": 0.85, "loss": 0.50, "threshold": 0.46}
    ensemble = {
        "roc_auc": 0.94,
        "accuracy": 0.92,
        "threshold": 0.44,
        "confusion_matrix": [[12, 2], [0, 22]],
        "note": "inherited threshold",
    }
    test_mean = {
        "roc_auc": 0.93,
        "accuracy": 0.84,
        "test_loss": 0.39,
        "threshold": 0.44,
        "fold": 3.0,
        "operating_fpr_target": 0.09,
        "operating_fpr_realised_on_val": 0.175,
    }
    return {
        "aggregate": {"mean": test_mean, "std": {key: 0.01 for key in test_mean}},
        "ensemble": ensemble,
        "cv_validation": {"mean": validation, "std": {}},
        "comparison": build_comparison_rows(
            validation_mean=validation, ensemble=ensemble, test_mean=test_mean
        ),
    }


# ------------------------------------------------------------- the comparison table


def test_comparison_rows_line_up_with_the_declared_columns(summary):
    for row in summary["comparison"]:
        assert len(row) == len(COMPARISON_COLUMNS)


def test_comparison_reads_the_test_columns_from_their_own_key_names(summary):
    rows = {row[0]: row for row in summary["comparison"]}
    # Per-fold metrics rename loss -> test_loss; validation does not. A row that silently
    # dropped the test value would still be a well-formed row, hence the explicit check.
    assert rows["Loss"][1] == 0.50
    assert rows["Loss"][3] == 0.39
    assert rows["AUC"] == ["AUC", 0.87, 0.94, 0.93]


def test_comparison_skips_metrics_no_column_reported():
    rows = build_comparison_rows(
        validation_mean={"roc_auc": 0.9}, ensemble={}, test_mean={}
    )
    assert [row[0] for row in rows] == ["AUC"]


# -------------------------------------------------------------- the ensemble's cut


def test_ensemble_takes_a_shared_threshold_when_the_strategy_produced_one():
    threshold, source = ensemble_operating_point(
        {"strategy": "threshold_average", "shared_threshold": 0.62},
        {1: 0.5, 2: 0.7},
    )
    assert threshold == pytest.approx(0.62)
    assert "threshold_average" in source


def test_ensemble_averages_the_folds_cuts_when_there_is_no_shared_one():
    # vertical_average equalises false positive rate, not probability, so it leaves
    # per-fold cuts behind and the ensemble has nothing else to inherit.
    threshold, source = ensemble_operating_point(
        {"strategy": "vertical_average", "shared_threshold": None},
        {1: 0.2, 2: 0.4, 3: 0.6},
    )
    assert threshold == pytest.approx(0.4)
    assert "mean of the per-fold" in source


# ------------------------------------------------------------------- what is logged


def test_bookkeeping_never_reaches_the_run_summary(summary, run):
    log_cv_to_wandb({"run_id": "abc"}, _config(), summary)

    for metric in NON_RESULT_METRICS:
        assert f"Test Mean {metric}" not in run.summary
        assert f"Test Std {metric}" not in run.summary
    assert run.summary["Test Mean roc_auc"] == 0.93


def test_ensemble_reports_more_than_ranking_metrics(summary, run):
    log_cv_to_wandb({"run_id": "abc"}, _config(), summary)

    assert run.summary["Test Ensemble accuracy"] == 0.92
    assert run.summary["Test Ensemble threshold"] == 0.44
    # Non-scalars stay out: a summary holds numbers, and the caveat belongs in the note.
    assert "Test Ensemble confusion_matrix" not in run.summary
    assert "Test Ensemble note" not in run.summary


def test_the_comparison_table_is_logged(summary, run):
    log_cv_to_wandb({"run_id": "abc"}, _config(), summary)

    table = run.logged["Metric Comparison"]
    assert table.columns == list(COMPARISON_COLUMNS)
    assert len(table.data) == len(summary["comparison"])


# ------------------------------------------------------- cleaning up an older run


def test_retired_keys_are_deleted_from_a_run_that_already_has_them():
    run = FakeRun(
        {
            "Fold": 5,
            "Fold AUC": 0.87,
            "Fold Best Epoch": 2,
            "Test Mean operating_fpr_target": 0.09,
            "Test Std operating_fpr_realised_on_val": 0.02,
            "Test Mean fold": 3.0,
            "CV Mean Validation AUC": 0.87,
            "Best Validation AUC": 0.87,
            "Folds Are Not Fold": 1,
        }
    )
    removed = prune_stale_summary_keys(run)

    assert set(removed) == {
        "Fold",
        "Fold AUC",
        "Fold Best Epoch",
        "Test Mean operating_fpr_target",
        "Test Std operating_fpr_realised_on_val",
        "Test Mean fold",
    }
    # A summary is not rebuilt from scratch, so anything not explicitly retired survives.
    assert run.summary["CV Mean Validation AUC"] == 0.87
    assert run.summary["Best Validation AUC"] == 0.87
    assert run.summary["Folds Are Not Fold"] == 1


# ------------------------------------------------- validation column from metadata


def test_validation_column_comes_from_the_runs_own_metadata():
    metadata = {
        "cv": {
            "fold_results": {
                1: {"metrics": {"roc_auc": 0.80, "accuracy": 0.70}},
                2: {"metrics": {"roc_auc": 0.90, "accuracy": 0.80}},
            }
        }
    }
    aggregate = cv_validation_aggregate(metadata)

    assert aggregate["mean"]["roc_auc"] == pytest.approx(0.85)
    assert aggregate["std"]["roc_auc"] == pytest.approx(np.std([0.8, 0.9], ddof=1))


def test_validation_column_tolerates_an_unfinished_or_non_cv_run():
    assert cv_validation_aggregate({})["mean"] == {}
    assert cv_validation_aggregate({"cv": {"fold_results": {1: None}}})["mean"] == {}


def _config():
    return {"wandb_entity": None, "wandb_project": "p", "wandb_mode": "disabled"}
