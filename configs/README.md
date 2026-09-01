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

**`"name": "ResNet10"`** — `num_classes` (or `out_channels`), `in_channels` (accepted as
an alias for MONAI's `n_input_channels`), `spatial_dims`, `conv1_t_size`,
`conv1_t_stride`, `no_max_pool`, `shortcut_type`, `widen_factor`, `feed_forward`,
`bias_downsample`, `act`, `norm`.

**`"name": "EfficientNetB0"`** — `num_classes` (or `out_channels`), `in_channels`,
`spatial_dims`, `norm`, `adv_prop`, and `model_name` if you want a different variant
(`efficientnet-b0` … `b7`); the class name stays `EfficientNetB0` either way.

Parameter counts at `spatial_dims: 3`, 1 input channel, 2 classes: DenseNet121 11.2M,
ResNet10 14.4M, EfficientNet-B3 12.1M. These are the ResNet and EfficientNet variants
nearest DenseNet121 — ResNet-18, the next depth up, is 33.2M, and EfficientNet-B2 drops
to 8.7M. Note that EfficientNet-B3 scales depth and input resolution as well as width,
so it is slower per step than 12.1M parameters suggests.

### `model.pretrained` (all three MONAI models)

MedicalNet ResNet10 downloads directly from its public Hugging Face repository:

```json
"pretrained": {
  "enabled": true,
  "source": "medicalnet",
  "freeze_backbone": false
}
```

The Hub checkpoint is pinned to an immutable revision and verified by SHA-256. Its
72 backbone tensors must match exactly; the original release has no classifier, so the
project's two-class `fc` layer remains newly initialised.

The architecture MedicalNet requires is **not** injected for you — `model.params`
carries it, so the config and the logged run describe the same model, and every setting
is available as a sweep axis. `configs/resnet10.json` is the worked example. A
weight-bearing mismatch raises at load time rather than silently loading a partial
backbone: `widen_factor`, `shortcut_type`, `bias_downsample`, `conv1_t_size` and
`feed_forward` are all checked by name and shape.

Two settings cannot be checked, because they change only the forward pass and carry no
weights: `conv1_t_stride` and `no_max_pool`. A wrong value there loads cleanly and
trains, so they are deliberate choices rather than pins. MedicalNet's own stem is
stride 2, and `configs/resnet10.json` matches it — which also drops peak VRAM at
batch 8 from 9.31 G to 1.44 G. Setting stride 1 doubles the spatial resolution reaching
the classifier (6×8×6 rather than 3×4×3 before global pooling) at 6.5× the memory, and
applies the pretrained filters at a scale they were not trained on.

`freeze_backbone: true` trains only the new classifier and keeps the frozen backbone,
including BatchNorm, in evaluation mode. `false` fine-tunes the whole network. Resume
and evaluation reconstruct the architecture without downloading pretrained weights.

Other MONAI models retain the local-checkpoint form using
`pretrained_weights_path`. Their compatible tensors are shape-filtered and a mismatched
classifier is skipped. Pretraining options sit beside `params`, not inside it.

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

wandb gets one timestamped parent run (aggregate + sweep objective) plus one child per
fold (per-epoch curves), sharing a group. A configured name such as `experiment` becomes
`experiment-20260901-143052`; every child reuses it as
`experiment-20260901-143052-fold1`, `...-fold2`, and so on.

## `refit`

For the **final model**: one network trained on everything except the test set. Enabled
by the presence of the block, exactly like `cv`; the two are alternatives and enabling
both raises.

```json
"cv":    { "enabled": false, "n_splits": 5, "shuffle": true, "random_seed": 42 },
"refit": { "enabled": true }
```

Otherwise identical to a single-split run: the same `split` numbers produce the same test
set, and `train_size` + `val_size` are then pooled into one training set (199 volumes on
OASIS, against 164 for a single split). No other setting changes meaning.

**Do not try to get this by setting `val_size: 0`.** `split` carves off `val + test`
together and then halves it, so a two-way split at the same `test_size` and seed lands on
a *different* test set — measured at 23 of 36 volumes in common — and every number would
be incomparable with the runs already on disk. (It also just raises: the inner split
would need `test_size=1.0`.)

With nothing held back there is nothing to select on, so **`last.pth` at `epochs` is the
model** and these are ignored, with one line on stdout and a `Refit Ignored Settings`
summary key saying so:

- `checkpoint.monitor` / `mode` / `min_delta` — no validation metric to monitor
- `checkpoint.save_best` — forced off; no epoch is "best"
- `early_stopping` — no validation signal to stall against

They stay in the file so a refit config is its CV sibling plus one block. Two consequences
worth knowing: no `Best Validation *` summary key is written (deliberately — there is no
held-out number), and the checkpoint stores `val_loss`, `monitor_value` and
`val_predictions` as `null` rather than substituting anything. `Refit Final Training AUC`
and friends are the only read on how the model ended up.

**Choosing `epochs`.** There is no validation curve, so this is a plain config value that
has to be justified. Take it from the same architecture's CV run — its per-fold best
epochs are in `metadata["cv"]["fold_results"][k]["epoch"]` and in the `CV Fold Best
Epochs` summary key. First add one to turn each 0-indexed epoch into a training count,
then take the median (recommended) or mean and round halves up to an integer.
Never use the maximum: that is one fold's luckiest epoch, which is exactly the
overfitting a refit cannot detect. A benchmark protocol's budget is also wrong here — a
fixed 60 exists to give every arm equal draws at a max-over-epochs objective, and a refit
takes no max.

Generate the refit recipe directly from a completed CV run. Median is the robust default;
use `--epoch-rule mean` when that aggregation is intentional:

```bash
uv run python generate_refit_config.py --cv-run checkpoints/<cv_run_id>
uv run python generate_refit_config.py --cv-run checkpoints/<cv_run_id> \
    --epoch-rule mean --output configs/my_refit.json
```

The generator copies the resolved hyperparameters actually stored by the CV trial,
disables CV and early stopping, enables refit, appends `-refit` to `wandb_name`, and
replaces legacy threshold settings with the current common-threshold definition. By
default it writes `configs/<wandb_name>_refit.json`, refuses to overwrite an existing
file without `--force`, and prints both fold statistics plus the training and evaluation
commands. A warning means at least one fold peaked at the CV budget boundary, so its
recommended refit duration may be censored.

Read the number off the CV run of **this** architecture, not a similar one. An epoch
budget does not transfer between different networks: the two DenseNet arms are the same
network differing only in whether pretrained weights load, and the pretrained one peaks
around epoch 4-8 precisely because it starts trained, where a scratch run needs far
longer. Nothing downstream would catch the swap, since a refit has no validation split.

**Threshold.** A refit has no held-out data to choose an operating point on. During final
evaluation it derives one from its CV sibling's saved best-fold validation labels and
probabilities:

```bash
uv run python evaluate.py --checkpoint checkpoints/<refit_run>/last.pth \
    --threshold-from checkpoints/<cv_run>
```

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

## `threshold`

Read by [`evaluate.py`](../evaluate.py) — how evaluation turns a completed CV run's
stored best-fold validation labels and probabilities into one decision threshold.
Training does not calculate or require this threshold.

```json
"threshold": {
  "strategy": "cv_common_threshold",
  "objective": "balanced_accuracy",
  "num_thresholds": 1000,
  "tie_break": "plateau_midpoint"
}
```

| key | values | notes |
|---|---|---|
| `strategy` | `cv_common_threshold`, `fixed`, `vertical_average`, `threshold_average`, `per_fold_youden` | Fresh configs default to `cv_common_threshold`; old stored metadata with no strategy keeps `per_fold_youden`. `cv_strategy` is a legacy alias. |
| `objective` | `balanced_accuracy`, `f1`, `sensitivity`, `specificity`, `precision` | Metric averaged equally across folds at every common candidate. |
| `num_thresholds` | int ≥ 2 | Builds `linspace(0, 1, num_thresholds)`. `threshold_grid` is accepted as a legacy alias. |
| `tie_break` | `plateau_midpoint`, `lowest`, `highest`, `closest_to_0_5` | Deterministic resolution of equal maxima. |
| `value` | float in `[0, 1]` | Required by `fixed`; no search is performed. |
| `fpr_rounding` | `at_least`, `nearest`, `at_most` | `vertical_average` only. A fold's achievable false positive rates are multiples of 1/n_negatives, so it lands on the target or steps past it. `at_least` never falls short, which never trades away sensitivity. |
| `fpr_grid` | int ≥ 2 | `vertical_average` only. 101 gives 0.01 steps, finer than 20 negatives can resolve anyway. |
| `threshold_grid` | int, or `0` | `threshold_average` only. `0` uses the union of the folds' own ROC thresholds, which is exact; a positive value uses that many evenly spaced points. |


**`cv_common_threshold`** evaluates the same numerical candidate on every fold's
out-of-fold probabilities, averages the configured objective across folds, and chooses
one cut from that mean curve. It never optimises folds separately and never sees the
test set. The first evaluation that needs the cut writes `threshold_curve.csv` and
`threshold_selection.json` beside the donor's `metadata.pth`; later evaluations reuse the
stored cut only when the complete threshold definition matches. A fresh refit triggers
this calculation with `--threshold-from`.

**`vertical_average`** (Fawcett 2006, Alg. 3) averages the folds' TPR over a shared
false-positive-rate axis, picks the target FPR maximising `(mean_tpr + 1 - fpr) / 2`, then
solves each fold back to *its own* probability cut. Because a false positive rate counts
only ranks within a fold, this is unaffected by one fold's probabilities being on a
different scale from another's. It remains available for explicit legacy reporting.

**`threshold_average`** (Fawcett 2006, Alg. 4; what `sklearn`'s
`TunedThresholdClassifierCV` computes) averages the folds' balanced-accuracy-vs-threshold
curves and takes one `argmax`, giving every fold the same cut. It compares probabilities
across folds, so it assumes they are calibrated alike; without that it is the weaker
choice. Note that averaging the *curves* and then taking one `argmax` is not the same as
averaging each fold's own `argmax` — `argmax` is non-linear.

**`per_fold_youden`** keeps whatever threshold each fold tuned during its own training.

Common and averaging strategies use the same `y_true`/`y_prob` bundle in each fold's
`best_model.pth`; only the aggregation changes. New checkpoints also store ordered sample
indices for exact provenance checks. Older bundles without indices remain usable after
checking their lengths, ordered labels, and recorded CV partitions. A checkpoint with no
saved validation predictions cannot supply a threshold without rerunning validation
inference.

With `--threshold-from`, the threshold policy comes from the refit/evaluation config, not
from the donor's historical training metadata. This lets the same saved probabilities be
analysed with `cv_common_threshold` even when the older donor recorded
`vertical_average`.

Override per invocation with `--threshold-strategy` and `--fpr-rounding`, or bypass
selection entirely with `--threshold`.

The **ensemble** has no threshold of its own to tune. It inherits one — the shared cut
when the strategy produced one, otherwise the mean of the per-fold cuts, which is what
`vertical_average` leaves behind since it equalises false positive rate rather than
probability. Its `roc_auc` and `average_precision` need no threshold and are the honest
read; the thresholded metrics beside them are there for comparability with the per-fold
numbers, not as a validated operating point. Averaging the folds' probabilities does not
average their calibration, so the inherited cut sits at an unknown place on the
ensemble's own score scale, and every sample in the cross-validation pool was trained on
by all but one of the fold models, so no held-out data is left on which a better one
could be chosen. `ensemble/test_metrics.json` carries a `note` and `threshold_source`
saying so.

## `evaluation`

Also read by `evaluate.py` only.

| key | values | notes |
|---|---|---|
| `output_dir` | path | Run id is appended. Default `evaluations`. |
| `log_wandb` | bool | `--log-wandb` can turn this on but not off. |
| `save_predictions` | bool | `false` writes `test_metrics.json` without the per-sample CSV. |

A cross-validation run also writes `summary.json`, whose `comparison` block is a table of
every metric against three columns — CV validation (each fold's best epoch, from the run's
own `metadata.pth`), test ensemble, and test mean. Under `log_wandb` it is logged as a
wandb table named **Metric Comparison**, which is what to plot; the same numbers go to the
run summary as `Test Mean *`, `Test Std *` and `Test Ensemble *`.

Read the validation column as a reference, not a result: it is measured at a threshold
tuned on the same validation split it scores, so it is optimistic in the way described
under *Thresholds during training*. It is the number the run was selected on, which is
exactly why it belongs next to the test columns.

Bookkeeping is kept out of the run summary — `fold` (whose mean is 3.0 for any 5-fold
run) and the `operating_fpr_target` / `operating_fpr_realised_on_val` pair, which record
how the shared cut was chosen rather than how the model scored. They stay in
`summary.json` and the per-fold `test_metrics.json`. Evaluating an older run also deletes
these keys, and the per-fold `Fold ...` keys, from its wandb summary if they are still
there from a previous version.

---

## Evaluating

`evaluate.py` takes the same JSON:

```bash
uv run python evaluate.py --checkpoint checkpoints/<refit_run>/last.pth \
  --threshold-from checkpoints/<cv_run> --config configs/my_config.json
```

Evaluating a CV directory on test is a legacy diagnostic and requires explicit opt-in:

```bash
uv run python evaluate.py --checkpoint checkpoints/<cv_run> --allow-cv-test-evaluation
```

It honours **only** `threshold`, `evaluation`, `device`, the `wandb_*` keys and
`dataloader`. Everything else — model, dataset, transforms, split, cv, loss, optimizer —
comes from the run's own `metadata.pth`, because those describe how the model was built
and how its inputs were preprocessed. Passing a different `transforms` block would
silently invalidate every number reported, so a supplied value that differs from the run's
is named on stdout and ignored. `dataloader` is safe to change because evaluation runs
under `model.eval()`, where BatchNorm uses its running statistics.

Precedence is **CLI flag > `--config` > the run's metadata > the code's default.**

## Thresholds during training

**There are none.** Training picks no operating point and reports no metric that needs
one: each epoch logs `Loss`, `AUC` and `Average Precision` for both splits, and nothing
else. What it does store is the epoch's validation predictions, which is what lets
`evaluate.py` choose a cut afterwards — see the `threshold` block above.

This used to work the other way: the threshold was retuned on validation every epoch and
accuracy, balanced accuracy, F1, precision, sensitivity and specificity were reported at
it, for both splits. Those numbers were measured at a cut fitted to the same 35-odd
samples they scored, so they were optimistic by construction — see Leeflang et al. (2008)
for the magnitude, roughly 6 points at n≈40. They are gone rather than caveated. Rank runs
on `Validation AUC`.

Two consequences for older runs. `checkpoint.monitor` no longer accepts `val_accuracy`,
`val_balanced_accuracy` or `val_f1` (every config in this directory already used
`val_auc`), and runs from before this change keep their `Validation Balanced Accuracy`,
`Validation F1` and `Validation Threshold` keys in wandb while new runs simply do not
write them — so a table spanning both will have holes in those columns.
