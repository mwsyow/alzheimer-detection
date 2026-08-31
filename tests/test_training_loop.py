"""Tests that drive train() itself, on a toy model small enough to run on CPU.

Two things are pinned here that nothing else in the suite could catch, because until now
no test exercised the training loop at all.

First, that training reports only threshold-free metrics. The old loop retuned a
threshold on validation every epoch and reported accuracy, F1 and the rest at it, which
scored those metrics at a cut fitted to the same samples they scored. The key set is
asserted *exactly*, not just checked for absences, so re-adding one of them fails.

Second, that a refit run -- no validation split at all -- neither crashes nor silently
fabricates the quantities it cannot measure.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

from conftest import FakeRun
from metrics import unpack_predictions
from train import train

EXPECTED_TRAIN_KEYS = {"Training Loss", "Training AUC", "Training Average Precision"}
EXPECTED_VAL_KEYS = {"Validation Loss", "Validation AUC", "Validation Average Precision"}

CHECKPOINT_CONFIG = {
    "monitor": "val_auc",
    "mode": "max",
    "min_delta": 0.0,
    "save_best": True,
    "save_last": True,
    "best_filename": "best_epoch_{epoch:03d}.pth",
    "last_filename": "last.pth",
}


def loader(n_samples: int = 24, seed: int = 0) -> DataLoader:
    """Both classes present, so roc_auc and average_precision are defined."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_samples) % 2
    # Separable enough that AUC is not degenerate, noisy enough that it is not exactly 1.
    images = torch.randn(n_samples, 1, 4, 4, 4, generator=generator)
    images += labels.view(-1, 1, 1, 1, 1).float()
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 2))


def run_epochs(tmp_path, epochs=1, with_validation=True, checkpoint_config=None, **kwargs):
    net = model()
    logger = FakeRun()
    result = train(
        epochs=epochs,
        model=net,
        optim=torch.optim.AdamW(net.parameters(), lr=1e-3),
        loss=nn.CrossEntropyLoss(),
        train_loader=loader(),
        val_loader=loader(seed=1) if with_validation else None,
        logger=logger,
        checkpoint_dir=tmp_path,
        checkpoint_config={**CHECKPOINT_CONFIG, **(checkpoint_config or {})},
        **kwargs,
    )
    return result, logger


def test_training_logs_exactly_the_threshold_free_metrics(tmp_path):
    _, logger = run_epochs(tmp_path)
    (logged,) = logger.log_calls
    assert set(logged) == {"Epoch"} | EXPECTED_TRAIN_KEYS | EXPECTED_VAL_KEYS


def test_no_thresholded_metric_is_reported(tmp_path):
    """Named individually, so a failure says which one came back.

    Matched as whole labels rather than substrings: "Average Precision" is threshold-free
    and stays, and it contains "Precision", which does not.
    """
    _, logger = run_epochs(tmp_path)
    (logged,) = logger.log_calls
    labels = {key.split(" ", 1)[1] for key in logged if key != "Epoch"}
    for banned in ("Balanced Accuracy", "Accuracy", "F1", "Precision", "Sensitivity",
                   "Specificity", "Threshold"):
        assert banned not in labels, f"{banned} is back"


def test_no_threshold_reaches_a_checkpoint(tmp_path):
    run_epochs(tmp_path)
    checkpoint = torch.load(tmp_path / "last.pth", weights_only=False)
    assert "threshold" not in checkpoint


def test_a_checkpoint_carries_the_validation_predictions(tmp_path):
    """Load-bearing: with no threshold stored, these are the only route to a cut."""
    run_epochs(tmp_path)
    checkpoint = torch.load(tmp_path / "last.pth", weights_only=False)
    y_true, y_prob = unpack_predictions(checkpoint["val_predictions"])
    assert len(y_true) == len(y_prob) == 24
    assert set(y_true.tolist()) == {0, 1}


def test_a_thresholded_monitor_is_rejected(tmp_path):
    """It used to be accepted and then never checkpoint, which looks like a training bug."""
    with pytest.raises(ValueError, match="checkpoint.monitor"):
        run_epochs(tmp_path, checkpoint_config={"monitor": "val_f1"})


# --- refit: no validation split ------------------------------------------------------


def test_a_refit_epoch_runs_and_writes_only_the_last_checkpoint(tmp_path):
    result, _ = run_epochs(tmp_path, epochs=3, with_validation=False)

    assert (tmp_path / "last.pth").exists()
    assert list(tmp_path.glob("best_epoch_*.pth")) == []
    assert result["best_result"] is None
    assert result["best_monitor_value"] is None


def test_a_refit_run_logs_no_validation_metrics(tmp_path):
    _, logger = run_epochs(tmp_path, with_validation=False)
    (logged,) = logger.log_calls
    assert set(logged) == {"Epoch"} | EXPECTED_TRAIN_KEYS


def test_a_refit_run_writes_no_best_validation_summary_key(tmp_path):
    """There is no held-out number, so an absent key is the honest report."""
    _, logger = run_epochs(tmp_path, with_validation=False)
    assert not any(key.startswith("Best Validation") for key in logger.summary)


def test_a_refit_run_never_early_stops(tmp_path):
    """The counter must be skipped, not merely left to a falsy default.

    Without a validation pass `improved` is False every epoch, so a counter that kept
    incrementing would trip `patience` and cut the refit short -- with nothing in the
    logs to say why, since a refit is expected to have no improving epochs.
    """
    result, logger = run_epochs(
        tmp_path,
        epochs=4,
        with_validation=False,
        early_stopping_config={"enabled": True, "patience": 1},
    )
    assert len(logger.log_calls) == 4
    assert result["early_stopping_counter"] == 0


def test_a_refit_checkpoint_fabricates_nothing(tmp_path):
    run_epochs(tmp_path, with_validation=False)
    checkpoint = torch.load(tmp_path / "last.pth", weights_only=False)
    assert checkpoint["val_loss"] is None
    assert checkpoint["monitor_value"] is None
    assert checkpoint["val_predictions"] is None


def test_a_refit_run_reports_its_final_training_metrics(tmp_path):
    """The only read on how the model ended up, absent a validation curve."""
    result, _ = run_epochs(tmp_path, epochs=2, with_validation=False)
    final = result["final_train_metrics"]
    assert final["roc_auc"] is not None
    assert final["average_precision"] is not None
    assert final["loss"] is not None


def test_a_refit_run_without_save_last_is_refused(tmp_path):
    """It would train to completion and write nothing at all."""
    with pytest.raises(ValueError, match="save_last"):
        run_epochs(
            tmp_path, with_validation=False, checkpoint_config={"save_last": False}
        )
