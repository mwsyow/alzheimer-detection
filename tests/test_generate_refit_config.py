"""Tests for generating a fresh-refit recipe from completed CV metadata."""

import json
import sys

import pytest
import torch

from datasets import resolve_split_mode
from generate_refit_config import (
    build_refit_config,
    default_output_path,
    load_fold_best_epochs,
    main,
    recommend_refit_epochs,
    write_refit_config,
)


def cv_metadata(best_epochs=(1, 20, 3, 3, 6), *, wandb_name="experiment"):
    folds = [{"train_idx": [1], "val_idx": [0]} for _ in range(len(best_epochs))]
    return {
        "run_id": "cv123456",
        "sweep_id": "sweep123",
        "config": {
            "epochs": 60,
            "device": "auto",
            "wandb_name": wandb_name,
            "model": {
                "name": "Simple3DCNN",
                "params": {"channels": [4, 8, 16], "dropout": 0.2},
            },
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 0.0003, "weight_decay": 0.0123},
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.1},
            },
            "split": {"random_seed": 42},
            "cv": {"enabled": True, "n_splits": len(best_epochs)},
            "early_stopping": {"enabled": True, "patience": 10},
            "threshold": {
                "cv_strategy": "vertical_average",
                "fpr_grid": 101,
            },
        },
        "cv": {
            "folds": folds,
            "completed_folds": list(range(1, len(best_epochs) + 1)),
            "fold_results": {
                fold: {"epoch": epoch}
                for fold, epoch in enumerate(best_epochs, start=1)
            },
        },
    }


def test_median_is_the_robust_default_and_mean_is_optional():
    median = recommend_refit_epochs([1, 20, 3, 3, 6])
    mean = recommend_refit_epochs([1, 20, 3, 3, 6], rule="mean")

    assert median["fold_epoch_counts"] == [2, 21, 4, 4, 7]
    assert median["epochs"] == 4
    assert mean["epochs"] == 8


def test_even_fold_statistics_use_conventional_half_up_rounding():
    assert recommend_refit_epochs([0, 1], rule="median")["epochs"] == 2
    assert recommend_refit_epochs([0, 1], rule="mean")["epochs"] == 2


@pytest.mark.parametrize("epochs", [[], [-1, 2], [True, 2], [1.5, 2]])
def test_invalid_fold_epochs_are_rejected(epochs):
    with pytest.raises(ValueError, match="epoch"):
        recommend_refit_epochs(epochs)


def test_generated_config_preserves_resolved_sweep_values_and_enables_refit():
    metadata = cv_metadata()
    original = json.loads(json.dumps(metadata["config"]))

    generated = build_refit_config(metadata, epochs=4)

    assert metadata["config"] == original
    assert generated["epochs"] == 4
    assert generated["optimizer"]["params"] == {
        "lr": 0.0003,
        "weight_decay": 0.0123,
    }
    assert generated["model"]["params"] == {
        "channels": [4, 8, 16],
        "dropout": 0.2,
    }
    assert generated["loss"]["params"]["label_smoothing"] == 0.1
    assert generated["cv"]["enabled"] is False
    assert generated["refit"] == {"enabled": True}
    assert generated["early_stopping"]["enabled"] is False
    assert generated["wandb_name"] == "experiment-refit"
    assert resolve_split_mode(generated) == "refit"


def test_generated_config_uses_the_current_common_threshold_definition():
    generated = build_refit_config(cv_metadata(), epochs=4)
    assert generated["threshold"] == {
        "strategy": "cv_common_threshold",
        "objective": "balanced_accuracy",
        "num_thresholds": 1000,
        "tie_break": "plateau_midpoint",
    }


def test_missing_wandb_name_falls_back_to_the_cv_run_id():
    metadata = cv_metadata(wandb_name=None)
    generated = build_refit_config(metadata, epochs=4)
    assert generated["wandb_name"] == "refit-cv123456"
    assert default_output_path(metadata).as_posix() == "configs/cv123456_refit.json"


def test_default_output_name_is_safe_and_does_not_include_runtime_timestamp():
    metadata = cv_metadata(wandb_name="My run / sweep")
    assert default_output_path(metadata).as_posix() == (
        "configs/My-run-sweep_refit.json"
    )


def test_fold_epochs_load_from_metadata_with_integer_or_json_string_keys(tmp_path):
    metadata = cv_metadata(best_epochs=(2, 4))
    metadata["cv"]["fold_results"] = {
        "1": {"epoch": 2},
        2: {"epoch": 4},
    }
    assert load_fold_best_epochs(tmp_path, metadata) == [2, 4]


def test_old_metadata_falls_back_to_each_selected_checkpoint(tmp_path):
    metadata = cv_metadata(best_epochs=(2, 4))
    metadata["cv"]["fold_results"] = {}
    for fold, epoch in enumerate((2, 4), start=1):
        directory = tmp_path / f"split_{fold}"
        directory.mkdir()
        torch.save({"epoch": epoch}, directory / "best_model.pth")

    assert load_fold_best_epochs(tmp_path, metadata) == [2, 4]


def test_incomplete_cv_run_is_rejected(tmp_path):
    metadata = cv_metadata(best_epochs=(2, 4))
    metadata["cv"]["completed_folds"] = [1]
    with pytest.raises(ValueError, match="incomplete"):
        load_fold_best_epochs(tmp_path, metadata)


def test_empty_completed_fold_list_is_not_treated_as_a_completed_old_run(tmp_path):
    metadata = cv_metadata(best_epochs=(2, 4))
    metadata["cv"]["completed_folds"] = []
    with pytest.raises(ValueError, match="incomplete"):
        load_fold_best_epochs(tmp_path, metadata)


def test_best_epoch_must_fit_inside_the_cv_budget(tmp_path):
    metadata = cv_metadata(best_epochs=(60,))
    with pytest.raises(ValueError, match="outside the configured CV budget"):
        load_fold_best_epochs(tmp_path, metadata)


@pytest.mark.parametrize("budget", [None, 0, True, 3.5])
def test_cv_budget_must_be_a_positive_integer(tmp_path, budget):
    metadata = cv_metadata(best_epochs=(1,))
    metadata["config"]["epochs"] = budget
    with pytest.raises(ValueError, match="positive integer epochs"):
        load_fold_best_epochs(tmp_path, metadata)


def test_non_cv_metadata_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a cross-validation run"):
        load_fold_best_epochs(tmp_path, {"run_id": "single", "config": {}})


def test_writing_refuses_overwrite_without_force(tmp_path):
    output = tmp_path / "refit.json"
    write_refit_config({"epochs": 3}, output)
    with pytest.raises(FileExistsError, match="--force"):
        write_refit_config({"epochs": 4}, output)

    write_refit_config({"epochs": 4}, output, force=True)
    assert json.loads(output.read_text())["epochs"] == 4
    assert output.read_text().endswith("\n")


def test_cli_generates_config_and_warns_when_a_fold_peaks_at_the_budget(
    tmp_path, monkeypatch, capsys
):
    run_dir = tmp_path / "checkpoints" / "cv123456"
    run_dir.mkdir(parents=True)
    metadata = cv_metadata(best_epochs=(2, 1))
    metadata["config"]["epochs"] = 3
    torch.save(metadata, run_dir / "metadata.pth")
    output = tmp_path / "generated_refit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_refit_config.py",
            "--cv-run",
            str(run_dir),
            "--epoch-rule",
            "mean",
            "--output",
            str(output),
        ],
    )

    main()

    generated = json.loads(output.read_text())
    stdout = capsys.readouterr().out
    assert generated["epochs"] == 3
    assert "WARNING" in stdout
    assert "Fold epoch counts: [3, 2]" in stdout
    assert f"--threshold-from {run_dir}" in stdout
