"""Small, opt-in helpers for rotating-CV throughput experiments."""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Sampler


def configure_optimized_training(config):
    """Resolve an explicit trainer selection, including job-local cache on resume."""
    performance = config.setdefault("performance", {})
    performance["enabled"] = True
    if os.environ.get("ALZHEIMER_CACHE_DIR"):
        performance["cache_dir"] = os.environ["ALZHEIMER_CACHE_DIR"]
    else:
        performance.setdefault("cache_dir", ".cache/rotating_cv")


class EpochSampler(Sampler):
    """Data order independent of loader worker creation and persistence."""

    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def input_cache_hash(item):
    """Include file identity/stat information, including Analyze image headers."""
    files = []

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                candidates = [path]
                if path.suffix == ".img":
                    candidates.append(path.with_suffix(".hdr"))
                for candidate in candidates:
                    stat = candidate.stat()
                    files.append(
                        (str(candidate.resolve()), stat.st_size, stat.st_mtime_ns)
                    )

    visit(item)
    return (
        hashlib.sha256(json.dumps([item, files], sort_keys=True, default=str).encode())
        .hexdigest()
        .encode()
    )


def append_timing(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EpochTimer:
    """CUDA events are resolved at epoch end, never synchronized per step."""

    def __init__(self, device):
        self.cuda = device is not None and torch.device(device).type == "cuda"
        self.events = []
        self.data_wait = 0.0
        self.samples = 0

    def batches(self, loader):
        start = time.perf_counter()
        iterator = iter(loader)
        self.data_wait += time.perf_counter() - start
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.data_wait += time.perf_counter() - start
            self.samples += len(batch[0])
            yield batch

    def start_step(self):
        if self.cuda:
            pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            pair[0].record()
            self.events.append(pair)

    def end_step(self):
        if self.cuda:
            self.events[-1][1].record()

    def gpu_seconds(self):
        if not self.events:
            return 0.0
        self.events[-1][1].synchronize()
        return sum(start.elapsed_time(end) for start, end in self.events) / 1000
