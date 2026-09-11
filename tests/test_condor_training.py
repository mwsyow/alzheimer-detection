"""Exercise HPC wrapper selection without installs, W&B calls or submitted jobs."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from artifacts import comparison_fingerprint, config_fingerprint
from conftest import FakeRun
from training_performance import configure_optimized_training
import train


ROOT = Path(__file__).resolve().parents[1]
MOCK_LAUNCH = r"""
pip() { :; }
nvidia-smi() { :; }
uv() {
    "$TEST_PYTHON" -c 'import json, os, sys; print("MOCK_UV " + json.dumps({"argv": sys.argv[1:], "optimized": os.getenv("ALZHEIMER_TRAIN_OPTIMIZED"), "cache": os.getenv("ALZHEIMER_CACHE_DIR")}))' "$@"
}
# Intercept the final exec too: these tests must never launch a real agent.
exec() { "$@"; }
wrapper=$1
shift
source "$wrapper" "$@"
"""


def launch(tmp_path, script, args):
    (tmp_path / ".env").write_text("WANDB_API_KEY=mock-test-key\n")
    env = {
        **os.environ,
        "REPO": str(tmp_path),
        "TEST_PYTHON": sys.executable,
        "_CONDOR_SCRATCH_DIR": str(tmp_path / "job scratch"),
        # Default mode must not inherit a selector from the submitting shell.
        "ALZHEIMER_TRAIN_OPTIMIZED": "1",
        "ALZHEIMER_CACHE_DIR": "/unused/inherited/cache",
    }
    result = subprocess.run(
        ["bash", "-c", MOCK_LAUNCH, "mock", str(ROOT / "condor" / script), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    calls = [
        json.loads(line.removeprefix("MOCK_UV "))
        for line in result.stdout.splitlines()
        if line.startswith("MOCK_UV ")
    ]
    return result, calls


@pytest.mark.parametrize("optimized", [False, True])
def test_single_run_selects_script_and_preserves_arguments(tmp_path, optimized):
    args = ["--config", "configs/a config.json", "--resume", "checkpoints/a run"]
    result, calls = launch(
        tmp_path, "train.sh", args + (["--optimized"] if optimized else [])
    )
    assert result.returncode == 0, result.stderr
    assert calls[-1]["argv"] == [
        "run",
        "--no-dev",
        "python",
        "train_optimized.py" if optimized else "train.py",
        *args,
    ]
    assert calls[-1]["optimized"] == ("1" if optimized else None)
    assert calls[-1]["cache"] == (
        str(tmp_path / "job scratch/alzheimer-preprocessing") if optimized else None
    )


@pytest.mark.parametrize("prefix", [[], ["wandb", "agent"]])
@pytest.mark.parametrize("position", ["none", "before", "after"])
def test_sweep_selects_mode_without_changing_id(tmp_path, prefix, position):
    args = [*prefix, "entity/project/sweep123"]
    if position == "before":
        args.insert(0, "--optimized")
    elif position == "after":
        args.append("--optimized")
    result, calls = launch(tmp_path, "sweep_agent.sh", args)
    assert result.returncode == 0, result.stderr
    assert calls[-1]["argv"] == [
        "run",
        "--no-dev",
        "wandb",
        "agent",
        "entity/project/sweep123",
    ]
    assert calls[-1]["optimized"] == (None if position == "none" else "1")
    assert calls[-1]["cache"] == (
        None
        if position == "none"
        else str(tmp_path / "job scratch/alzheimer-preprocessing")
    )


@pytest.mark.parametrize(
    "args", [[], ["--optimized"], ["bare-id"], ["--typo", "e/p/s"], ["e/p/s", "extra"]]
)
def test_invalid_sweep_arguments_fail_before_setup(tmp_path, args):
    result, calls = launch(tmp_path, "sweep_agent.sh", args)
    assert result.returncode == 2
    assert not calls


def test_missing_training_arguments_fail_before_setup(tmp_path):
    result, calls = launch(tmp_path, "train.sh", ["--optimized"])
    assert result.returncode == 2
    assert not calls


def test_job_cache_replaces_old_resume_path_without_changing_group(monkeypatch):
    config = {"performance": {"enabled": True, "cache_dir": "/old/job"}}
    original = copy.deepcopy(config)
    monkeypatch.setenv("ALZHEIMER_CACHE_DIR", "/new/job/cache")
    configure_optimized_training(config)
    assert config["performance"]["cache_dir"] == "/new/job/cache"
    assert comparison_fingerprint(original) == comparison_fingerprint(config)
    assert config_fingerprint(original) != config_fingerprint(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_sweep_train_command_honors_wrapper_environment(monkeypatch, enabled):
    class FakeConfig(dict):
        def update(self, values, allow_val_change=False):
            super().update(values)

    config = json.loads(
        (ROOT / "configs/pilot_rotating_simple3dcnn_5ep.json").read_text()
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused.json"])
    monkeypatch.setattr(train, "load_config", lambda _: copy.deepcopy(config))
    if enabled:
        monkeypatch.setenv("ALZHEIMER_TRAIN_OPTIMIZED", "1")
    else:
        monkeypatch.delenv("ALZHEIMER_TRAIN_OPTIMIZED", raising=False)
    monkeypatch.setenv("ALZHEIMER_CACHE_DIR", "/job/local/cache")
    run = FakeRun()

    def init(**kwargs):
        run.config = FakeConfig(kwargs["config"])
        # An explicit wrapper selection wins over stale sweep performance defaults.
        run.config["performance.enabled"] = False
        run.config["performance.cache_dir"] = "/sweep/default/cache"
        return run

    captured = {}
    monkeypatch.setattr(train.wandb, "init", init)
    monkeypatch.setattr(train, "runner", lambda **kwargs: captured.update(kwargs))
    train.main()
    actual = captured["config"]["performance"]
    assert actual["enabled"] is enabled
    assert actual["cache_dir"] == (
        "/job/local/cache" if enabled else "/sweep/default/cache"
    )
    assert run.config["performance"] == actual
    assert run.finished
