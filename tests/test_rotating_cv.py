import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from datasets import build_rotating_cv_split_indices
from metrics import regression_metrics
from models import build_model
from tasks import StandardizedHuberLoss
from conftest import FakeRun
import train
from rotating_evaluation import compare_rotating_runs, evaluate_rotating_run


def items(n=50):
    return [
        {
            "subject_id": f"subject-{index:03d}",
            "image_id": f"image-{index:03d}",
            "label": index % 2,
            "age": 50.0 + index,
        }
        for index in range(n)
    ]


def config(seed=42, n_splits=5):
    return {
        "cv": {
            "strategy": "rotating_test",
            "n_splits": n_splits,
            "shuffle": True,
            "random_seed": seed,
        }
    }


def test_every_subject_is_test_and_validation_once():
    data = items()
    state = build_rotating_cv_split_indices(data, config())
    test = [index for fold in state["folds"] for index in fold["test_idx"]]
    val = [index for fold in state["folds"] for index in fold["val_idx"]]
    assert sorted(test) == list(range(len(data)))
    assert sorted(val) == list(range(len(data)))
    for fold in state["folds"]:
        train, validation, test = map(
            set, (fold["train_idx"], fold["val_idx"], fold["test_idx"])
        )
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(range(len(data)))


def test_partition_depends_only_on_cv_seed_not_item_order():
    data = items()
    expected = build_rotating_cv_split_indices(data, config())["subject_folds"]
    reversed_data = list(reversed(data))
    actual = build_rotating_cv_split_indices(reversed_data, config())["subject_folds"]
    assert actual == expected
    changed = build_rotating_cv_split_indices(data, config(seed=7))["subject_folds"]
    assert changed != expected


def test_all_scans_of_a_subject_stay_together():
    data = items()
    data.append({**data[0], "image_id": "another-scan"})
    state = build_rotating_cv_split_indices(data, config())
    for fold in state["folds"]:
        memberships = [
            key
            for key in ("train_idx", "val_idx", "test_idx")
            if 0 in fold[key] or len(data) - 1 in fold[key]
        ]
        assert len(memberships) == 1
        assert 0 in fold[memberships[0]]
        assert len(data) - 1 in fold[memberships[0]]


def test_manifest_is_validated_and_reused(tmp_path):
    data = items()
    generated = build_rotating_cv_split_indices(data, config())
    path = tmp_path / "folds.csv"
    pd.DataFrame(
        [
            {"subject_id": subject, "label": data[int(subject[-3:])]["label"], "fold": fold}
            for subject, fold in generated["subject_folds"].items()
        ]
    ).to_csv(path, index=False)
    cfg = config(seed=999)
    cfg["cv"]["manifest_path"] = str(path)
    reused = build_rotating_cv_split_indices(data, cfg)
    assert reused["subject_folds"] == generated["subject_folds"]


def simple_config(task):
    return {
        "task": {"name": task},
        "model": {
            "name": "Simple3DCNN",
            "params": {"channels": [4, 8], "num_classes": 99},
        },
    }


@pytest.mark.parametrize(
    ("task", "width"), [("ad_classification", 2), ("age_regression", 1)]
)
def test_task_owns_head_and_features(task, width):
    model = build_model(simple_config(task), initialize_pretrained=False).eval()
    image = torch.randn(2, 1, 16, 16, 16)
    with torch.no_grad():
        result = model.forward_with_features(image)
        ordinary = model(image)
    assert ordinary.shape == (2, width)
    assert torch.equal(ordinary, result["output"])
    assert result["F"].ndim == 5
    assert result["hI"].shape == (2, 8)


def test_age_scaler_round_trips_and_metrics_are_in_years():
    loss = StandardizedHuberLoss(mean=70.0, std=10.0)
    ages = torch.tensor([60.0, 70.0, 80.0])
    standardized = loss.standardize(ages)
    assert torch.equal(loss.inverse(standardized), ages)
    metrics = regression_metrics(ages.numpy(), [61.0, 69.0, 82.0])
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx((6 / 3) ** 0.5)


class TinySource:
    def __init__(self):
        self.items = items(18)
        labels = torch.tensor([item["label"] for item in self.items])
        generator = torch.Generator().manual_seed(123)
        images = torch.randn(18, 1, 8, 8, 8, generator=generator)
        images += labels.reshape(-1, 1, 1, 1, 1).float()
        self.data = TensorDataset(images, labels)

    def loader(self, indices, mode, shuffle=False):
        return DataLoader(Subset(self.data, indices), batch_size=3, shuffle=shuffle)

    def fold_loaders(self, fold):
        return self.loader(fold["train_idx"], "train", True), self.loader(
            fold["val_idx"], "val"
        )

    def subject_id(self, index):
        return self.items[index]["subject_id"]

    def item_id(self, index):
        return self.items[index]["image_id"]

    def set_random_state(self, seed):
        pass


class TinyAgeSource(TinySource):
    def __init__(self):
        super().__init__()
        images = self.data.tensors[0]
        ages = torch.tensor([item["age"] for item in self.items], dtype=torch.float32)
        self.data = TensorDataset(images, ages)


def rotating_training_config(tmp_path):
    return {
        "epochs": 1,
        "seed": 11,
        "device": "cpu",
        "wandb_entity": None,
        "wandb_project": "test",
        "wandb_mode": "disabled",
        "task": {"name": "ad_classification"},
        "model": {
            "name": "Simple3DCNN",
            "params": {"channels": [4], "num_classes": 2},
        },
        "optimizer": {"name": "AdamW", "params": {"lr": 0.001}},
        "loss": {"name": "CrossEntropyLoss", "params": {}},
        "lr_scheduler": {"enabled": False},
        "cv": {
            "enabled": True,
            "strategy": "rotating_test",
            "n_splits": 3,
            "shuffle": True,
            "random_seed": 9,
        },
        "dataloader": {"batch_size": 3, "num_workers": 0},
        "checkpoint": {
            "dir": str(tmp_path / "checkpoints"),
            "save_best": True,
            "save_last": True,
            "monitor": "val_auc",
            "mode": "max",
            "min_delta": 0.0,
        },
        "early_stopping": {"enabled": False},
        "threshold": {
            "objective": "balanced_accuracy",
            "num_thresholds": 21,
            "tie_break": "plateau_midpoint",
        },
        "artifacts": {"dir": str(tmp_path / "artifacts")},
        "evaluation": {"output_dir": str(tmp_path / "evaluations")},
    }


def test_rotating_training_exports_and_evaluates_one_oof_row_per_subject(
    tmp_path, monkeypatch
):
    source = TinySource()
    monkeypatch.setattr(train, "build_dataset_source", lambda config: source)
    children = []

    def child(*args, **kwargs):
        run = FakeRun(run_id=f"child{len(children)}")
        children.append(run)
        return run

    monkeypatch.setattr(train, "start_fold_run", child)
    parent = FakeRun(run_id="rotate01")
    cfg = rotating_training_config(tmp_path)
    state = train.run_rotating_cross_validation(parent, cfg)
    run_dir = tmp_path / "checkpoints" / parent.id
    assert len(state["completed_folds"]) == 3
    manifest = pd.read_csv(state["artifact_manifest"])
    assert len(manifest) == 3 * len(source.items)
    assert set(manifest["split"]) == {"train", "validation", "test"}
    summary = evaluate_rotating_run(run_dir, tmp_path / "evaluations")
    predictions = pd.read_csv(
        tmp_path / "evaluations" / parent.id / "oof_predictions.csv"
    )
    assert len(predictions) == len(source.items)
    assert predictions["subject_id"].nunique() == len(source.items)
    assert summary["metrics"]["n"] == len(source.items)
    assert len(summary["thresholds"]) == 3
    assert predictions["threshold"].nunique() == 1
    assert set(predictions["threshold_source"]) == {"all_validation_folds"}
    shared = summary["threshold_selection"]["shared_threshold"]
    assert predictions["threshold"].iloc[0] == pytest.approx(shared)
    fold_thresholds = {
        details["threshold"] for details in summary["thresholds"].values()
    }
    assert len(fold_thresholds) == 1
    assert next(iter(fold_thresholds)) == pytest.approx(shared)
    selection = summary["threshold_selection"]
    assert selection["strategy"] == "cv_common_threshold"
    assert selection["validation_folds"] == [1, 2, 3]
    assert selection["validation_subjects"] == len(source.items)
    assert selection["thresholded_metrics_calibration_inclusive"] is True
    destination = tmp_path / "evaluations" / parent.id
    with (destination / "threshold_selection.json").open() as handle:
        assert json.load(handle)["threshold"] == pytest.approx(shared)
    curve = pd.read_csv(destination / "threshold_curve.csv")
    assert len(curve) == cfg["threshold"]["num_thresholds"]

    per_fold = evaluate_rotating_run(run_dir, tmp_path / "evaluations", threshold_mode="per-fold")
    per_predictions = pd.read_csv(destination / "per-fold" / "oof_predictions.csv")
    for metric in ("roc_auc", "average_precision"):
        assert per_fold["metrics"][metric] == pytest.approx(summary["metrics"][metric])
    assert not per_fold["threshold_selection"]["thresholded_metrics_calibration_inclusive"]
    for fold, rows in per_predictions.groupby("fold"):
        threshold = per_fold["thresholds"][fold]["threshold"]
        assert (rows.threshold == threshold).all()
        assert (rows.prediction == (rows.probability >= threshold).astype(int)).all()
    assert set(per_predictions.threshold_source) == {"own_validation_fold"}
    assert pd.read_csv(destination / "oof_predictions.csv").equals(predictions)

    # Test predictions affect reported metrics, never the validation-selected cut.
    for artifact in manifest.loc[manifest["split"] == "test", "artifact"]:
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        payload["image_probability"] = 1.0 - payload["image_probability"]
        torch.save(payload, artifact)
    repeated = evaluate_rotating_run(run_dir, tmp_path / "reevaluated")
    assert repeated["threshold_selection"]["shared_threshold"] == pytest.approx(shared)
    repeated_per_fold = evaluate_rotating_run(run_dir, tmp_path / "reevaluated", threshold_mode="per-fold")
    assert repeated_per_fold["thresholds"] == per_fold["thresholds"]


def test_age_rotating_training_saves_scalers_and_real_year_predictions(
    tmp_path, monkeypatch
):
    source = TinyAgeSource()
    monkeypatch.setattr(train, "build_dataset_source", lambda config: source)
    monkeypatch.setattr(
        train, "start_fold_run", lambda *args, **kwargs: FakeRun(run_id="age-child")
    )
    parent = FakeRun(run_id="rotateage")
    cfg = rotating_training_config(tmp_path)
    cfg["task"] = {"name": "age_regression"}
    state = train.run_rotating_cross_validation(parent, cfg)
    assert set(state["target_scalers"]) == {1, 2, 3}
    checkpoint = torch.load(
        tmp_path / "checkpoints" / parent.id / "split_1" / "best_model.pth",
        weights_only=False,
    )
    assert checkpoint["target_scaler"] == state["target_scalers"][1]
    summary = evaluate_rotating_run(
        tmp_path / "checkpoints" / parent.id, tmp_path / "evaluations"
    )
    assert set(("mae", "rmse", "r2", "pearson_r")) <= set(summary["metrics"])
    predictions = pd.read_csv(
        tmp_path / "evaluations" / parent.id / "oof_predictions.csv"
    )
    assert {"true_age", "predicted_age", "residual"} <= set(predictions)
    assert "threshold_selection" not in summary
    assert not (tmp_path / "evaluations" / parent.id / "threshold_curve.csv").exists()


def test_rotating_comparison_keeps_only_scalar_threshold_provenance(
    tmp_path, monkeypatch
):
    def summary(path, output_dir, **kwargs):
        threshold = 0.25 if path.name == "run-a" else 0.75
        return {
            "run_id": path.name,
            "task": "ad_classification",
            "architecture": "Simple3DCNN",
            "adaptation_mode": "scratch",
            "comparison_sha256": "same-recipe",
            "config_sha256": path.name,
            "seed": 1 if path.name == "run-a" else 2,
            "cv_random_seed": 42,
            "folds": 5,
            "best_epochs": [1, 1, 1, 1, 1],
            "thresholds": {},
            "threshold_selection": {
                "threshold": threshold,
                "curve": [{"large": "nested payload"}],
            },
            "metrics": {"roc_auc": 0.7, "n": 20},
        }

    monkeypatch.setattr("rotating_evaluation.evaluate_rotating_run", summary)
    result = compare_rotating_runs(
        [Path("run-a"), Path("run-b")], tmp_path / "evaluations"
    )
    runs = result["runs"]
    assert "threshold_selection" not in runs
    assert runs["threshold"].tolist() == [0.25, 0.75]
    assert runs["thresholded_metrics_calibration_inclusive"].all()


def test_per_fold_comparison_separates_outputs_and_forwards_logging(tmp_path, monkeypatch):
    calls = []
    def summary(path, output_dir, threshold_mode, log_wandb):
        calls.append((threshold_mode, log_wandb))
        return {"run_id": path.name, "task": "ad_classification",
                "architecture": "Simple3DCNN", "adaptation_mode": "scratch",
                "comparison_sha256": "same", "seed": 42, "cv_random_seed": 7,
                "best_epochs": [1, 2], "thresholds": {},
                "threshold_selection": {"threshold": None, "fold_thresholds": {1: 0.2, 2: 0.8},
                                        "thresholded_metrics_calibration_inclusive": False},
                "metrics": {"roc_auc": 0.8, "n": 20}}
    monkeypatch.setattr("rotating_evaluation.evaluate_rotating_run", summary)
    result = compare_rotating_runs([Path("a"), Path("b")], tmp_path,
                                   threshold_mode="per-fold", log_wandb=True)
    assert calls == [("per-fold", True)] * 2
    assert result["output_dir"].name.endswith("--per-fold")
    assert not result["runs"]["thresholded_metrics_calibration_inclusive"].any()
    assert result["runs"].threshold.isna().all()
    assert json.loads(result["runs"].fold_thresholds.iloc[0]) == {"1": 0.2, "2": 0.8}
