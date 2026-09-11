# Optimized rotating-test CV

The optimized entry point shares the current training, checkpoint and feature-export
implementation. It supports only enabled `rotating_test` CV, not historical splitting
or refit workflows. No additional packages beyond the project environment are needed.

```bash
uv run python train_optimized.py --config configs/pilot_rotating_simple3dcnn_5ep.json
```

For a reproducible local baseline/optimized comparison, including five epochs on each
of five folds, run:

```bash
uv run python benchmark_training.py
```

On HPC, the wrappers also accept `--optimized` through the submit-file `args` macro:

```bash
condor_submit args="--optimized --config configs/bench_simple3dcnn.json" condor/train.sub
condor_submit args="--optimized" n_agents=4 sweep_id=ENTITY/PROJECT/SWEEP condor/sweep_agent.sub
```

The sweep selector works with existing commands naming `train.py`. It affects newly
submitted jobs only, enables the optimized mode explicitly, and places preprocessing
on job-local scratch. See `condor/README.md` for details.

The comparison requires CUDA and the preprocessed OASIS data. It uses W&B disabled,
four CPU threads/four loader workers, identical sample ordering in both arms, and
batch size 8 (reduced equally to 4 or 2 only if the GPU preflight runs out of memory).
It writes fresh timestamped directories without overwriting existing pilots.

## Configuration

`train_optimized.py` enables optimizations by default. The same options can be supplied
to the existing `train.py` entry point through JSON or W&B sweep YAML parameters, so
agent submission mechanics do not need to change:

```json
"performance": {
  "enabled": true,
  "cache_dir": ".cache/rotating_cv",
  "persistent_workers": true,
  "prefetch_factor": 1,
  "profile": false
}
```

- Deterministic preprocessing is cached on disk, only before the first random
  transform. Augmentation stays live, in its original position. Cache keys include
  transform settings, input file identity, size and modification time (including
  Analyze `.hdr` files). Clear or use a new cache after changing preprocessing code
  or package versions; do not modify source files while training.
- On HPC, point `cache_dir` to a job-local scratch path to avoid repeated shared
  filesystem traffic. Cached volumes consume disk space and are not model artifacts.
- Pinned loading and nonblocking CUDA transfers are enabled. Worker persistence and
  prefetching are used only when `dataloader.num_workers > 0`.
- Prediction tensors are copied to CPU once per pass; spatial features are copied
  once per export batch. All existing per-subject `F`, `hI`, logits and metadata remain.
- Full precision, checkpoint selection/frequency and the CV protocol are unchanged.
- Model seed and `cv.random_seed` retain their separate roles. Opt-in runs use an
  epoch-seeded sampler independent of worker creation. Worker randomness is seeded,
  but persistent workers change augmentation RNG progression versus nonpersistent
  runs; do not claim bitwise equivalence with augmentation or across resume boundaries.
  The comparison pilot disables augmentation and verifies initialization/order.

## Measurements and verification

With `profile: true`, each checkpoint split directory contains `timings.csv`.
The run directory contains initialization hashes and export timings. The comparison
adds GPU samples, resolved configurations, full artifact audits, prediction comparisons,
pooled OOF evaluations and `report.md` under `reports/training_performance/<timestamp>/`.
`results.json` lists exact checkpoint, artifact and evaluation paths.

Total runtime includes initial cache population and exports; audit/evaluation runtime
is excluded. CUDA events measure stream step elapsed time, not pure kernel occupancy.
Data waits can overlap device work. Utilization samples include desktop/other GPU
activity. One sequential laptop pilot is indicative, not a controlled HPC benchmark;
OS caches are not flushed. Five-epoch predictive metrics are correctness checks, not
converged performance estimates.
