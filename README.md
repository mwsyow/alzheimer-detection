# Alzheimer MRI Classification

This project trains 3D CNN models on OASIS MRI data for binary Alzheimer/CDR classification. The MRI volumes are loaded from Analyze/NIfTI-style image files with MONAI/Nibabel transforms, labels are read from the cleaned OASIS spreadsheet, and training metrics/checkpoints are tracked with Weights & Biases.

## Code Flow

The main entrypoint is `train.py`.

1. Load a base JSON config from `configs/simple_3dcnn.json`.
2. Initialize a W&B run.
3. Apply W&B sweep overrides, if the run is launched by a sweep.
4. Load MRI paths and labels.
5. Create a fixed stratified train/validation/test split.
6. Build MONAI transforms, datasets, dataloaders, model, loss, and optimizer.
7. Train while logging metrics to W&B.
8. Save checkpoints under `checkpoints/<wandb_run_id>/`.

Important checkpoint files:

- `metadata.pth`: static run metadata, config, and split indices.
- `last.pth`: latest completed epoch, used for interruption recovery.
- `best_epoch_<epoch>.pth`: saved only when validation loss improves.

## Normal Training

Run training from the JSON config:

```bash
uv run python train.py --config configs/simple_3dcnn.json
```

The config controls model params, optimizer params, transforms, dataloader settings, checkpointing, early stopping, and W&B settings.

## Resume Training

To resume the latest interrupted run, use `last.pth`:

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

## Test Evaluation

Use `evaluate.py` after selecting a final checkpoint. It evaluates only the held-out test split saved in `metadata.pth`.

```bash
uv run python evaluate.py --checkpoint checkpoints/<wandb_run_id>/best_epoch_000.pth
```

Choose a device explicitly if needed:

```bash
uv run python evaluate.py --checkpoint checkpoints/<wandb_run_id>/best_epoch_000.pth --device cpu
uv run python evaluate.py --checkpoint checkpoints/<wandb_run_id>/best_epoch_000.pth --device cuda
```

To log final test metrics to the same W&B run:

```bash
uv run python evaluate.py --checkpoint checkpoints/<wandb_run_id>/best_epoch_000.pth --log-wandb
```

Evaluation outputs are saved under:

```text
evaluations/<wandb_run_id>/
```

Files:

- `test_metrics.json`: test loss, accuracy, balanced accuracy, precision, sensitivity, specificity, F1, ROC-AUC, and confusion matrix.
- `test_predictions.csv`: image path, true label, predicted label, and class-1 probability.

## W&B Sweep

The sweep config is in `configs/sweep.yaml`. It uses Bayesian search over selected hyperparameters such as learning rate, batch size, checkpoint min-delta, and early-stopping patience.

Create the sweep once:

```bash
wandb sweep configs/sweep.yaml
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

Each agent runs:

```bash
uv run python train.py --config configs/simple_3dcnn.json
```

Sweep hyperparameters are injected through `wandb.config` and merged into the nested JSON config using dotted keys such as:

```text
optimizer.params.lr
dataloader.batch_size
early_stopping.patience
```

Each sweep trial gets its own W&B run ID, so checkpoints are isolated under:

```text
checkpoints/<wandb_run_id>/
```
