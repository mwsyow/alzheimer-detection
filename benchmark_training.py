"""Paired five-fold Simple3DCNN throughput pilot, with complete artifact auditing.

Run: .venv/bin/python benchmark_training.py
The baseline retains uncached/blocking I/O. Both arms use identical epoch-based
sample order so persistent workers cannot confound the comparison.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from artifacts import config_fingerprint, file_sha256
from datasets import build_dataset_source, build_rotating_cv_split_indices
from models import build_model
from rotating_evaluation import evaluate_rotating_run
from tasks import adaptation_mode
from train import build_loss, build_optimizer, seed_everything, train_step


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def preflight(config):
    """Real-shape forward/backward; never reduce only one benchmark arm."""
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU benchmark requires CUDA")
    source = build_dataset_source(config)
    fold = build_rotating_cv_split_indices(source.items, config)["folds"][0]
    images = torch.stack([source.dataset("train")[i][0] for i in fold["train_idx"][:8]])
    labels = torch.tensor([source.items[i]["label"] for i in fold["train_idx"][:8]])
    for batch in (8, 4, 2):
        model = optimizer = None
        try:
            seed_everything(config["seed"])
            model = build_model(config).cuda()
            optimizer = build_optimizer(config, model)
            train_step(
                model,
                optimizer,
                build_loss(config),
                images[:batch],
                labels[:batch],
                torch.device("cuda"),
            )
            torch.cuda.synchronize()
            return batch
        except torch.cuda.OutOfMemoryError:
            if batch == 2:
                raise
        finally:
            del model, optimizer
            torch.cuda.empty_cache()


def monitor_gpu(path, stop):
    with path.open("w") as handle:
        handle.write(
            "elapsed_seconds,gpu_utilization_percent,memory_used_mib,temperature_c,power_w\n"
        )
        start = time.perf_counter()
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                handle.write(
                    f"{time.perf_counter() - start:.3f},{result.stdout.strip()}\n"
                )
                handle.flush()
            stop.wait(2)


def audit_run(run_dir):
    """Check every payload and re-infer one subject per split and fold."""
    metadata = torch.load(
        run_dir / "metadata.pth", map_location="cpu", weights_only=False
    )
    config, cv = metadata["config"], metadata["cv"]
    assert cv["completed_folds"] == [1, 2, 3, 4, 5]
    manifest = pd.read_csv(cv["artifact_manifest"])
    source_config = copy.deepcopy(config)
    source_config.pop("performance", None)
    source = build_dataset_source(source_config)
    assert len(manifest) == len(source.items) * 5
    assert cv["fold_manifest_sha256"] == file_sha256(Path(cv["fold_manifest"]))
    for split in ("test", "validation"):
        rows = manifest[manifest.split == split]
        assert len(rows) == len(source.items)
        assert rows.subject_id.is_unique
        assert set(rows.subject_id) == set(cv["subject_folds"])
    model = build_model(config).cuda().eval()
    sample_errors = []
    predictions = []
    feature_shapes = set()
    for fold, split_indices in enumerate(cv["folds"], 1):
        rows = manifest[manifest.fold == fold]
        assert rows.dataset_index.is_unique
        checkpoint_path = Path(rows.checkpoint.iloc[0])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        checksum = file_sha256(checkpoint_path)
        checked_splits = set()
        val_predictions = checkpoint["val_predictions"]
        for row in rows.to_dict("records"):
            payload = torch.load(
                row["artifact"], map_location="cpu", weights_only=False
            )
            index = payload["dataset_index"]
            assert (
                index
                in split_indices[
                    {"train": "train_idx", "validation": "val_idx", "test": "test_idx"}[
                        payload["split"]
                    ]
                ]
            )
            assert payload["subject_id"] == source.subject_id(index)
            assert payload["image_id"] == source.item_id(index)
            assert payload["target"] == source.items[index]["label"]
            for key in (
                "subject_id",
                "dataset_index",
                "fold",
                "split",
                "seed",
                "cv_random_seed",
                "best_epoch",
                "checkpoint",
                "run_metadata",
                "architecture",
                "adaptation_mode",
                "config_sha256",
                "checkpoint_sha256",
            ):
                assert payload[key] == row[key], key
            assert payload["fold"] == fold
            assert payload["seed"] == config["seed"]
            assert payload["cv_random_seed"] == config["cv"]["random_seed"]
            assert payload["architecture"] == config["model"]["name"]
            assert payload["adaptation_mode"] == adaptation_mode(config)
            assert payload["config_sha256"] == config_fingerprint(config)
            assert payload["checkpoint_sha256"] == checksum
            assert (
                Path(payload["run_metadata"]).resolve()
                == (run_dir / "metadata.pth").resolve()
            )
            assert (
                payload["best_epoch"]
                == checkpoint["epoch"]
                == cv["fold_results"][fold]["epoch"]
            )
            for key in ("F", "hI", "output", "logits"):
                assert torch.isfinite(payload[key]).all(), key
            assert list(payload["F"].shape) == json.loads(row["F_shape"])
            assert list(payload["hI"].shape) == json.loads(row["hI_shape"])
            feature_shapes.add((tuple(payload["F"].shape), tuple(payload["hI"].shape)))
            torch.testing.assert_close(
                payload["F"].mean(dim=(-3, -2, -1)), payload["hI"]
            )
            torch.testing.assert_close(payload["output"], payload["logits"])
            logit = float(payload["logits"][1] - payload["logits"][0])
            assert abs(logit - payload["image_logit"]) < 1e-6
            assert (
                abs(
                    float(torch.sigmoid(torch.tensor(logit)))
                    - payload["image_probability"]
                )
                < 1e-6
            )
            with torch.no_grad():
                head_output = model.classifier(payload["hI"].cuda()).cpu()
            torch.testing.assert_close(
                head_output, payload["output"], atol=1e-5, rtol=1e-5
            )
            if payload["split"] == "validation":
                offset = val_predictions["indices"].tolist().index(index)
                assert (
                    abs(
                        float(val_predictions["y_prob"][offset])
                        - payload["image_probability"]
                    )
                    < 1e-5
                )
            if payload["split"] not in checked_splits:
                # Match the original batch shape AND worker-side preprocessing.
                # Single-volume cuDNN kernels and CPU reduction threading can
                # differ numerically from the exported multi-volume batch.
                split_key = {
                    "train": "train_idx",
                    "validation": "val_idx",
                    "test": "test_idx",
                }[payload["split"]]
                audit_loader = source.loader(split_indices[split_key], mode="test")
                audit_iterator = iter(audit_loader)
                images, _ = next(audit_iterator)
                with torch.no_grad():
                    actual = model.forward_with_features(images.cuda())
                del audit_iterator, audit_loader
                for key in ("F", "hI", "output"):
                    error = float((actual[key][0].cpu() - payload[key]).abs().max())
                    sample_errors.append(error)
                    torch.testing.assert_close(
                        actual[key][0].cpu(), payload[key], atol=1e-4, rtol=1e-4
                    )
                checked_splits.add(payload["split"])
            predictions.append(
                {
                    "fold": fold,
                    "split": payload["split"],
                    "subject_id": payload["subject_id"],
                    "probability": payload["image_probability"],
                }
            )
    del model
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "payloads_checked": len(manifest),
        "subjects": len(source.items),
        "reinferred_subjects": 15,
        "max_reinference_error": max(sample_errors),
        "feature_shapes": sorted(feature_shapes),
    }, pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot_rotating_simple3dcnn_5ep.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = (args.output_dir or Path("reports/training_performance") / stamp).resolve()
    report.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if (
        config["model"]["name"] != "Simple3DCNN"
        or config["cv"]["strategy"] != "rotating_test"
    ):
        raise ValueError("Pilot requires Simple3DCNN rotating_test CV")
    config["epochs"] = 5
    config["cv"]["n_splits"] = 5
    config["wandb_mode"] = "disabled"
    config["evaluation"]["log_wandb"] = False
    config["early_stopping"]["enabled"] = False
    torch.set_num_threads(4)
    config["dataloader"].update(batch_size=preflight(config), num_workers=4)
    save_json(
        report / "environment.json",
        {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "threads": 4,
            "batch_size": config["dataloader"]["batch_size"],
        },
    )
    print(f"Benchmark outputs: {report}", flush=True)
    results, frames, checkpoints = {}, {}, {}
    for arm in ("baseline", "optimized"):
        resolved = copy.deepcopy(config)
        for key in ("checkpoint", "artifacts"):
            folder = "checkpoints" if key == "checkpoint" else key
            resolved[key]["dir"] = str(
                root / folder / "training_performance" / stamp / arm
            )
        resolved["evaluation"]["output_dir"] = str(
            root / "evaluations/training_performance" / stamp / arm
        )
        resolved["wandb_name"] = f"simple3dcnn-performance-{arm}"
        resolved["performance"] = {
            "enabled": arm == "optimized",
            "profile": True,
            "cache_dir": str(report / "preprocessing_cache"),
            "persistent_workers": True,
            "prefetch_factor": 1,
        }
        path = report / f"{arm}.json"
        save_json(path, resolved)
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
        stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_gpu, args=(report / f"{arm}_gpu.csv", stop), daemon=True
        )
        monitor.start()
        start = time.perf_counter()
        try:
            with (report / f"{arm}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "train_optimized.py" if arm == "optimized" else "train.py",
                        "--config",
                        str(path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                )
        finally:
            elapsed = time.perf_counter() - start
            stop.set()
            monitor.join(timeout=12)
        run_dirs = list(Path(resolved["checkpoint"]["dir"]).glob("*/metadata.pth"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0].parent
        checkpoints[arm] = run_dir
        save_json(
            report / f"{arm}_runtime.json",
            {"total_seconds": elapsed, "checkpoint_dir": str(run_dir)},
        )
        timings = pd.concat(
            [pd.read_csv(p) for p in sorted(run_dir.glob("split_*/timings.csv"))],
            ignore_index=True,
        )
        assert len(timings) == 25
        timings.to_csv(report / f"{arm}_timings.csv", index=False)
        audit, frames[arm] = audit_run(run_dir)
        save_json(report / f"{arm}_audit.json", audit)
        evaluation = evaluate_rotating_run(
            run_dir, output_dir=Path(resolved["evaluation"]["output_dir"])
        )
        gpu = pd.read_csv(report / f"{arm}_gpu.csv")
        results[arm] = {
            "total_seconds": elapsed,
            "train_seconds": float(timings.train_seconds.sum()),
            "data_wait_seconds": float(timings.data_wait_seconds.sum()),
            "validation_seconds": float(timings.validation_seconds.sum()),
            "checkpoint_seconds": float(timings.checkpoint_seconds.sum()),
            "export_seconds": float(
                pd.read_csv(run_dir / "export_timings.csv").export_seconds.sum()
            ),
            "first_epoch_train_seconds": float(timings.iloc[0].train_seconds),
            "subsequent_epoch_mean_seconds": float(
                timings.iloc[1:].train_seconds.mean()
            ),
            "mean_gpu_utilization_percent": float(gpu.gpu_utilization_percent.mean()),
            "peak_gpu_allocated_mib": float(timings.peak_gpu_bytes.max() / 1024**2),
            "max_temperature_c": float(gpu.temperature_c.max()),
            "metrics": evaluation["metrics"],
            "best_epochs": evaluation["best_epochs"],
            "checkpoint_dir": str(run_dir),
            "artifact_dir": str(Path(resolved["artifacts"]["dir"]) / run_dir.name),
            "evaluation_dir": str(
                Path(resolved["evaluation"]["output_dir"]) / run_dir.name
            ),
            "audit": audit,
        }
        save_json(report / "results.json", results)
        print(f"{arm}: {elapsed:.1f}s; audit passed", flush=True)
    for filename in ("fold_manifest.csv", "initialization.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(checkpoints["baseline"] / filename),
            pd.read_csv(checkpoints["optimized"] / filename),
        )
    left, right = (
        pd.read_csv(report / f"{arm}_timings.csv") for arm in ("baseline", "optimized")
    )
    assert left.sample_order.equals(right.sample_order)
    merged = frames["baseline"].merge(
        frames["optimized"],
        on=["fold", "split", "subject_id"],
        validate="one_to_one",
        suffixes=("_baseline", "_optimized"),
    )
    merged.to_csv(report / "prediction_comparison.csv", index=False)
    results["comparison"] = {
        "speedup": results["baseline"]["total_seconds"]
        / results["optimized"]["total_seconds"],
        "identical_initialization_splits_and_order": True,
        "max_probability_difference": float(
            (merged.probability_baseline - merged.probability_optimized).abs().max()
        ),
        "max_train_loss_difference": float(
            (left.train_loss - right.train_loss).abs().max()
        ),
        "max_validation_loss_difference": float(
            (left.validation_loss - right.validation_loss).abs().max()
        ),
    }
    save_json(report / "results.json", results)
    lines = [
        "# Simple3DCNN rotating-CV throughput pilot",
        "",
        "Five epochs per fold, five folds, full precision. Same initialization, partitions and sample order; augmentation disabled.",
        "",
        "| Measurement | Baseline | Optimized |",
        "|---|---:|---:|",
    ]
    for key in (
        "total_seconds",
        "train_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "checkpoint_seconds",
        "export_seconds",
        "mean_gpu_utilization_percent",
        "peak_gpu_allocated_mib",
    ):
        lines.append(
            f"| {key} | {results['baseline'][key]:.3f} | {results['optimized'][key]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"End-to-end speedup: {results['comparison']['speedup']:.2f}x.",
            "",
            "## Correctness",
            "",
            json.dumps(results["comparison"], indent=2),
            "",
            "Both runs passed complete artifact audits; metrics and exact output paths are in results.json.",
            "",
            "## Limitations",
            "",
            "One sequential laptop comparison, not an HPC benchmark. OS file caches were not flushed; optimized preprocessing cache started empty. GPU utilization includes desktop activity. CUDA step events measure stream elapsed time (including scheduling gaps), not pure kernel occupancy. Data-wait and GPU times may overlap and must not be added. Full wall time includes startup, preprocessing, training and exports; audit/evaluation time is excluded. Five-epoch accuracy is a correctness check, not a converged benchmark. Unified-threshold OOF operating-point metrics retain the existing calibration-inclusive interpretation.",
        ]
    )
    (report / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Finished: {report / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
