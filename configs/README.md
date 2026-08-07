# Config reference

Copy [`example.json`](example.json) and edit. It is a valid, runnable OASIS config with
every supported key present, so anything not listed below is not read by the code.

```bash
uv run python train.py --config configs/my_config.json
```

`configs/` is gitignored apart from `example.json`, `example_sweep.yaml` and this file,
so your own configs stay local.

---

## Top level

| key | values | notes |
|---|---|---|
| `epochs` | int | Per fold when cross-validation is on. |
| `device` | `"auto"`, `"cpu"`, `"cuda"` | `auto` picks cuda when available. |
| `wandb_entity` | string or `null` | `null` uses your default entity. |
| `wandb_project` | string | Consider a separate project for smoke tests. |
| `wandb_mode` | `"online"`, `"offline"`, `"disabled"` | **Overrides the `WANDB_MODE` env var**, because it is passed to `wandb.init` explicitly. Set it here, not in the shell. |
| `wandb_name` | string or `null` | `null` lets wandb generate a name. |

## `dataset`

Chosen by `name`, the same way `model` is. Backends live in
[`datasets.py`](../datasets.py) under `DATASET_BUILDERS`.

### `"name": "oasis"`

| key | notes |
|---|---|
| `image_glob` | Glob for the `.img` volumes. |
| `label_path` | Spreadsheet with `ID` and `CDR` columns. Label is `int(bool(CDR))`, so CDR 0.5 counts as positive. |

### `"name": "medmnist"`

| key | notes |
|---|---|
| `flag` | e.g. `nodulemnist3d`, `adrenalmnist3d`, `vesselmnist3d`, `synapsemnist3d`, `organmnist3d`, `fracturemnist3d`. |
| `root` | Download cache directory. |
| `download` | `true` fetches on first use. |
| `size` | Optional. Omit for 28³; `64` uses the MedMNIST+ resolution. |
| `include_labels` | Optional list of original class ids to keep. |
| `positive_labels` | Optional list mapped to target 1; everything else becomes 0. |

`include_labels` + `positive_labels` turn a multi-class set into a binary one. The
balanced organ task in `medmnist_organ_binary.json` is
`include_labels: [0,9,10,1,2,5]`, `positive_labels: [1,2,5]` — liver/spleen/pancreas
against kidney/kidney/bladder, 600 vs 572.

Images are held in memory, so `LoadImaged` is skipped automatically. Everything
downstream is identical to OASIS.

> Configs written before this block existed used top-level `image_glob` and
> `label_path`. Those are still honoured, which is why old checkpoints remain
> resumable and evaluable.

## `model`

`params` are passed to the constructor. See [`models.py`](../models.py).

**`"name": "Simple3DCNN"`** — `num_classes`, `in_channels`, `channels` (list, one conv
block each), `kernel_size`, `padding`, `pool_kernel_size`, `use_batch_norm`, `dropout`.

**`"name": "DenseNet121"`** — `num_classes` (or `out_channels`), `in_channels`,
`spatial_dims`, `init_features`, `growth_rate`, `block_config`, `bn_size`, `act`,
`norm`, `dropout_prob`. Any other key is silently dropped by the allow-list.

### `model.pretrained` (DenseNet121)

```json
"pretrained": {
  "enabled": true,
  "pretrained_weights_path": "pretrained/DenseNet121/86_acc_model.pth",
  "freeze_backbone": false
}
```

Sits **beside** `params`, not inside it. Weights load with a shape filter, so a
mismatched classifier head is skipped and randomly initialised — that is what lets the
3-class rootstrap checkpoint feed a 2-class model.

`freeze_backbone: true` leaves only the final linear layer trainable (2,050 of 11.2M
parameters). Both `pretrained` keys must be set or `freeze_backbone` does nothing.

## `loss` / `optimizer`

Only `CrossEntropyLoss` and `AdamW` are implemented; anything else raises.

`loss.params.weight` takes a per-class list for imbalance, e.g. `[1.0, 1.343]` for a
94/70 training split. `optimizer.params` accepts `lr`, `weight_decay`, and the rest of
the AdamW signature.

## `transforms`

Applied in this order; augmentation is train-split only.

| key | notes |
|---|---|
| `resize` / `spatial_size` / `resize_mode` | `false` keeps native resolution. |
| `scale_intensity` | Min-max to [0, 1]. |
| `normalize_intensity` / `normalize_nonzero` / `normalize_channel_wise` | Z-score. |
| `rand_rotate90` / `_prob` / `_spatial_axes` | **Train only.** Ill-advised on T88 volumes: they are atlas-registered, so a 90° rotation discards the voxel correspondence a small CNN needs, with no test-time equivalent. |

## `split`

`train_size` + `val_size` + `test_size` must sum to 1.0. `random_seed` seeds the
stratified split only — model init and shuffling are unseeded.

Add `"source": "dataset"` to use a dataset's own published partition instead
(MedMNIST only; OASIS provides none). Under CV that keeps the official test set and
refolds train+val.

## `cv`

Cross-validation is enabled by **the presence of this block**; `"enabled": false`
switches it off without deleting it.

```json
"cv": { "enabled": true, "n_splits": 5, "shuffle": true, "random_seed": 42 }
```

The test set is carved out once by `split`, then the pooled train+val is
`StratifiedKFold`-ed, so test never varies across folds and stays comparable to
single-split runs.

Layout, one directory per fold:

```
checkpoints/<run_id>/
├── metadata.pth          # resume state: current fold, current epoch, fold results
├── split_1/{best_model.pth, last.pth}
└── split_2/...
```

Resume with the run directory — it picks up the interrupted fold at the right epoch and
skips completed ones:

```bash
uv run python train.py --config configs/my_config.json --resume checkpoints/<run_id>
```

wandb gets one parent run (aggregate + sweep objective) plus one child per fold
(per-epoch curves), sharing a group.

## `dataloader`

`batch_size`, `num_workers`. Note `Simple3DCNN` uses `BatchNorm3d`, whose batch
statistics are unreliable at batch size 2 and whose running stats are what `eval()`
then uses.

## `checkpoint`

| key | values | notes |
|---|---|---|
| `dir` | path | Run id is appended. |
| `save_best` / `save_last` | bool | |
| `best_filename` | `"best_epoch_{epoch:03d}.pth"` | `{epoch}` is optional. CV always uses `best_model.pth`. |
| `last_filename` | `"last.pth"` | |
| `monitor` | `val_auc`, `val_roc_auc`, `val_balanced_accuracy`, `val_accuracy`, `val_f1`, `val_loss` | |
| `mode` | `"max"` or `"min"` | Must match the monitor — `min` for `val_loss`, `max` otherwise. |
| `min_delta` | float | Improvement required to count. |

Prefer `val_auc` over `val_loss`. Validation loss converges to the class-prior entropy
(0.682 for a 135/100 split), so it rewards a model that has only learned the base rate.

The monitor also drives early stopping and the flat `Best Validation <X>` summary key a
sweep reads.

## `early_stopping`

`enabled`, `patience` — patience counts epochs without improvement on
`checkpoint.monitor`. Keep it generous; validation metrics on a small validation set
swing hard between epochs, and a real best epoch can arrive late.

---

## Thresholds

Not configurable, but worth knowing: every epoch the decision threshold is retuned on
validation to maximise balanced accuracy, and the best epoch's value is stored in the
checkpoint. `evaluate.py` applies that stored threshold at test time, so test metrics
reflect a tuned operating point rather than a bare `argmax`. Override with
`--threshold`. For CV ensembles the per-fold thresholds are averaged.
