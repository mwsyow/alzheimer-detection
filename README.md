# Alzheimer MRI Classification

This project trains 3D CNN models on OASIS MRI data for binary Alzheimer/CDR classification. The MRI volumes are loaded from Analyze/NIfTI-style image files with MONAI/Nibabel transforms, labels are read from the cleaned OASIS spreadsheet, and training metrics/checkpoints are tracked with Weights & Biases.

Six arms across four architectures are compared under one fixed protocol: Simple3DCNN,
ResNet10 from scratch, ResNet10 with MedicalNet weights, EfficientNetB0, DenseNet121
pretrained, and DenseNet121 from scratch. The `configs/bench_*.json` files are the audit
trail for that comparison — `checkpoints/` is gitignored, so they are the only tracked
record of the protocol each arm ran under — and `tests/test_benchmark_protocol.py`
asserts the shared protocol block stays identical across them.

Documentation lives in three places: this file for the workflow end to end,
[`configs/README.md`](configs/README.md) for every config key, and
[`condor/README.md`](condor/README.md) for running on the HTCondor cluster.

## Code Flow

The main entrypoint is `train.py`.

1. Load the JSON config named by `--config` (required; start from `configs/example.json`).
2. Initialize a W&B run, appending a timestamp to a configured `wandb_name`.
3. Apply W&B sweep overrides, if the run is launched by a sweep.
4. Load MRI paths and labels.
5. Pick the split mode from the config — single train/validation/test split,
   `cv` (stratified K-fold over pooled train+val, fixed test set), or
   `refit` (train+val pooled into one training set, no validation).
6. Build MONAI transforms, datasets, dataloaders, model, loss, and optimizer.
7. Train while logging metrics to W&B. Each epoch logs `Loss`, `AUC` and
   `Average Precision` only — training picks no decision threshold.
8. Save checkpoints under `checkpoints/<wandb_run_id>/`.

Single-split and refit runs write:

- `metadata.pth`: static run metadata, config, and split indices.
- `last.pth`: latest completed epoch, used for interruption recovery.
- `best_epoch_<epoch>.pth`: saved only when the monitored metric improves. A refit has
  no validation split to monitor, so it writes none — `last.pth` at `epochs` *is* the
  model.

A cross-validation run writes one directory per fold instead, with a single
best-so-far file overwritten on improvement:

```text
checkpoints/<run_id>/
├── metadata.pth          # resume state: current fold, current epoch, fold results
├── split_1/{best_model.pth, last.pth}
└── split_2/...
```

W&B gets one parent run per CV trial (aggregate metrics and the sweep objective) plus
one child per fold (per-epoch curves), sharing a group.

## Normal Training

Run training from the JSON config:

```bash
uv run python train.py --config configs/example.json
```

The config controls model params, optimizer params, transforms, dataloader settings,
splitting and cross-validation, checkpointing, early stopping, threshold selection, and
W&B settings. See [`configs/README.md`](configs/README.md) for every key.

## Resume Training

To resume an interrupted single-split or refit run, point `--resume` at `last.pth`:

```bash
uv run python train.py --resume checkpoints/<wandb_run_id>/last.pth
```

Resume automatically loads `metadata.pth` from the same directory and restores:

- original config
- train/validation/test split indices
- model weights
- optimizer state
- best validation loss
- early-stopping counter

You can also resume from a specific best checkpoint:

```bash
uv run python train.py --resume checkpoints/<wandb_run_id>/best_epoch_003.pth
```

For a cross-validation run, pass the **run directory**. It picks up the interrupted fold
at the epoch it stopped at and skips folds that already completed:

```bash
uv run python train.py --resume checkpoints/<wandb_run_id>
```

`--config` is not needed when resuming — the config comes from `metadata.pth`, so the
resumed run cannot silently change what it is training.

## Test Evaluation

Generate a refit config from the completed CV run, train it, then evaluate its
`last.pth`. Evaluation calculates the threshold from the CV run's saved best-fold
validation labels and probabilities (or reuses an identical previously calculated
selection):

```bash
uv run python train.py --config configs/<cv-config>.json
uv run python generate_refit_config.py --cv-run checkpoints/<cv_run_id>
uv run python train.py --config configs/<generated-refit-config>.json
uv run python evaluate.py \
  --checkpoint checkpoints/<refit_run_id>/last.pth \
  --threshold-from checkpoints/<cv_run_id>
```

Here `<cv_run_id>` is the parent CV trial's W&B run ID—the directory name under
`checkpoints/`—not its W&B group ID or sweep ID.

`generate_refit_config.py` copies the CV trial's resolved hyperparameters, aggregates its
per-fold best epochs into a fixed epoch budget (`--epoch-rule median`, the default, or
`mean`), and writes `configs/<wandb_name>_refit.json`. Use `--output` for a different
path and `--force` to replace an existing file.

`evaluate.py` takes `--config` for evaluation-time settings only (threshold, evaluation,
device, W&B and dataloader blocks); everything describing the trained model comes from
the run's own metadata. `--device cpu`/`--device cuda` chooses a device, `--log-wandb`
attaches the final metrics to the run, and `--threshold`, `--threshold-strategy` and
`--fpr-rounding` override the operating-point policy per invocation. Precedence is
**CLI flag > `--config` > the run's metadata > the code's default.**

Direct CV-directory test evaluation reports every fold plus an ensemble. It is
diagnostic only and requires `--allow-cv-test-evaluation`:

```bash
uv run python evaluate.py --checkpoint checkpoints/<cv_run_id> \
  --allow-cv-test-evaluation
```

Evaluation outputs are saved under:

```text
evaluations/<wandb_run_id>/
```

Files:

- `test_metrics.json`: test loss, AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, NPV, F1, FPR, threshold provenance, and confusion matrix.
- `test_predictions.csv`: image path, true label, predicted label, and class-1 probability.
- `confusion_matrix.png`.

A CV-directory evaluation nests these under `fold_<k>/` and `ensemble/`, and adds a
`summary.json` whose `comparison` block tables every metric across CV validation, test
ensemble, and test mean.

## W&B Sweep

Sweep configs live beside the JSON configs as `configs/*_sweep.yaml`;
[`configs/example_sweep.yaml`](configs/example_sweep.yaml) is the commented reference and
doubles as the sweep documentation. Each one names its own search `method` (the benchmark
arms use `grid`) and carries the `command` block that launches `train.py` with that arm's
JSON config, so the sweep file and the config it drives travel together.

Create the sweep once:

```bash
wandb sweep configs/example_sweep.yaml
```

Then start one or more agents:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

Without a count, a sweep agent can keep requesting new trials until you stop it. Use `--count` to limit how many trials an agent should run.

For local testing:

```bash
wandb agent --count 10 <entity>/<project>/<sweep_id>
```

For HPC/Condor, a common pattern is one trial per submitted job:

```bash
wandb agent --count 1 <entity>/<project>/<sweep_id>
```

Sweep hyperparameters are injected through `wandb.config` and merged into the nested JSON config using dotted keys such as:

```text
optimizer.params.lr
dataloader.batch_size
cv.random_seed
```

Each sweep trial gets its own W&B run ID, so checkpoints are isolated under:

```text
checkpoints/<wandb_run_id>/
```

Under cross-validation a trial produces K+1 W&B runs — one parent holding the sweep
objective plus one child per fold. Children never write the objective key, so a fold can
never win `sweep.best_run()`.

## Tests

```bash
uv run pytest
```

The suite covers the benchmark protocol, CV reporting and resume, refit splitting and
config generation, threshold selection, transforms, and the pretrained loaders. It needs
no GPU and no data.
