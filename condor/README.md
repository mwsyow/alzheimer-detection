# Running on the SIC HTCondor cluster

Tooling for getting this project onto the Saarland CS HPC cluster and running jobs
there. Source for the cluster facts below: `data/CS_HPC_Docu.pdf`.

| File | Purpose |
| --- | --- |
| `sync_to_hpc.sh` | Push everything git can't carry (data, `.env`, configs, weights) |
| `check_internet.sub` | Submit file for the worker-node connectivity probe |
| `check_internet.sh` | The probe itself — runs inside the container, not directly |
| `sweep_agent.sub` | Submit file for wandb sweep agents |
| `sweep_agent.sh` | Agent wrapper — runs inside the container, not directly |
| `evaluate.sub` | Submit file for evaluating a trained checkpoint |
| `evaluate.sh` | Evaluation wrapper — runs inside the container, not directly |
| `train.sub` | Submit file for a single training run (one config, no sweep) |
| `train.sh` | Training wrapper — runs inside the container, not directly |

## Prerequisites

- **University network.** The submit nodes are only reachable from inside it; use
  the VPN from off campus.
- **SIC credentials.** Your SIC username, not your `@uni-saarland.de` email. Manage
  the account at <https://sam.sic.saarland>.
- Submit nodes: `conduit.hpc.uni-saarland.de` and `conduit2.hpc.uni-saarland.de`
  (equivalent; the scripts use `conduit2`). You can only log in to these, never to
  worker nodes.

---

## `sync_to_hpc.sh`

`.gitignore` excludes `data/`, `.env`, `configs/*` (bar the examples and the
`bench_*` benchmark arms), and `pretrained/`, so `git pull` on the cluster leaves the
project unrunnable. This script fills those gaps.

`/home` is NFS-shared with every worker node, so a full sync only has to happen
once — jobs read the files directly, with no `transfer_input_files`.

```bash
./condor/sync_to_hpc.sh <sic-user> [extra-path ...]
./condor/sync_to_hpc.sh <sic-user> --only <path> [path ...] [--delete] [--dry-run]
```

### Default set

| Path | Size |
| --- | --- |
| `data/T88_111_masked/` | 2.9 GB, 470 files (235 subjects × `.hdr`/`.img`) |
| `data/oasis_cross-sectional_cdr_cleaned.xlsx` | 20 KB |
| `configs/` | few KB |
| `pretrained/` | 44 MB |
| `.env` | if present; warns to stderr if not |

Never synced: `data/oasis_cross-sectional_disc*` (48 GB of raw discs, of which
`T88_111_masked` is already the flattened output), `checkpoints/`, `wandb/`, and
`.venv` — the venv is built against local Python and jobs run in Docker anyway.

### Options

| Option | Effect |
| --- | --- |
| `extra-path ...` | Adds files/folders to the default set. Paths are relative to the repo root and recreated remotely, so `notebooks` lands at `<remote-dir>/notebooks`. |
| `--only <path> ...` | Syncs *only* the given paths, skipping the default set. |
| `--delete` | Removes remote files no longer present locally. Requires `--only`. |
| `--dry-run` | Lists what would change without touching the cluster. |
| `REMOTE_DIR=<path>` | Env var overriding the repo path on the cluster (default `alzheimer-detection`, relative to `$HOME`). |

Flags are order-independent. Exit codes: `2` for a usage error, `1` for `--only`
with no paths.

### Examples

```bash
# first run — full migration, plus this folder so the submit files go too
./condor/sync_to_hpc.sh <sic-user> condor

# after editing a config
./condor/sync_to_hpc.sh <sic-user> --only configs

# repo lives elsewhere on the cluster
REMOTE_DIR=projects/alz ./condor/sync_to_hpc.sh <sic-user>
```

### Notes

**Re-running is cheap.** rsync transfers only files whose size or mtime changed,
so the 2.9 GB moves once; later runs stat 470 files and send nothing. That also
makes it the repair command — a killed transfer resumes from the partial file
rather than restarting.

**Deletions don't propagate by default.** Files you delete locally stay on the
cluster. This matters because the dataset is glob-based
(`data/T88_111_masked/*masked_gfc.img`): stale volumes on the cluster silently
enlarge the training set, with nothing in the logs saying so. If you regenerate
the data, prune explicitly, dry run first:

```bash
./condor/sync_to_hpc.sh <sic-user> --only data/T88_111_masked --delete --dry-run
./condor/sync_to_hpc.sh <sic-user> --only data/T88_111_masked --delete
```

`--delete` is rejected without `--only` on purpose. The script passes `-R` to
rsync so each source path is recreated remotely in one call; with the default set,
several sources share an implied `data/` parent, and deletion semantics around
implied directories are too subtle to aim safely. Requiring one explicit path
means the deletion target is always something you typed.

**No SSH config needed.** One master connection is opened into a `mktemp -d`
socket and reused by every `ssh`/`rsync`, so you authenticate once. An `EXIT` trap
closes it; nothing is written to `~/.ssh`.

---

## `check_internet.sub` — connectivity probe

Answers whether worker nodes have outbound internet, which decides if
`"wandb_mode": "online"` (and therefore sweeps) is usable.

```bash
ssh <sic-user>@conduit2.hpc.uni-saarland.de
cd alzheimer-detection/condor
condor_submit check_internet.sub
condor_q                      # idle -> running -> gone
cat check_internet.*.out
```

It reports the worker hostname, whether `/home` mounted, HTTPS reachability of
`api.wandb.ai` / `pypi.org` / `github.com`, and then — if `.env` holds a
`WANDB_API_KEY` — performs a real online `wandb.init` + `log` + `finish`, printing
the run URL. Reachability alone wouldn't prove auth survives the network path,
which is why the end-to-end run is the line that actually matters.

A `404` from `api.wandb.ai` is the healthy case: it serves nothing at its root, and
any HTTP status back means the host was reached.

**Result, 2026-08-09:** internet works. All three hosts reachable from
`lofn.hpc.uni-saarland.de`, `/home` mounted, and the wandb run was created
successfully. Online mode and sweeps are viable, and PyPI access means a stock
PyTorch image plus `uv sync` at job start works — no custom image needed.

Delete the throwaway `hpc-connectivity-check` run from the wandb project afterwards.

---

## `sweep_agent.sub` — running a wandb sweep

Create the sweep on the submit node, then queue agents against its id:

```bash
mkdir -p condor/logs                                  # once; Condor won't create it
uv run wandb sweep configs/resnet10_sweep.yaml        # prints <entity>/<project>/<id>
condor_submit sweep_id=<entity>/<project>/<id> condor/sweep_agent.sub
```

Each queued agent pulls trials until the sweep is exhausted, so `queue 4` means
four trials run concurrently. Override without editing the file:

```bash
condor_submit n_agents=2 sweep_id=<...> condor/sweep_agent.sub
```

Use the `n_agents` macro, **not** `condor_submit -queue N`. HTCondor's `-queue` takes a
whole queue *statement*, which may contain spaces (`-queue 3 item in list`), so it has
to be the last argument on the command line — `condor_submit -queue 1 sweep_id=... \
condor/sweep_agent.sub` silently swallows the macro and the submit file into the queue
statement. Macros are order-independent.

`initialdir` uses `$ENV(HOME)`, so the file needs no per-user editing.

`sweep_agent.sh` runs inside the container and: `cd`s to the repo on NFS, sources
`.env` for `WANDB_API_KEY`, installs `uv` to `/tmp/pytools` (no root, so
`--target` is required), syncs the locked environment into `/tmp/venv`, prints
whether CUDA is visible, then execs `wandb agent`. The venv and uv cache live on
node-local `/tmp` so parallel agents can't race each other over NFS. Hugging Face
uses the same policy: `HF_HOME` points to job-local scratch, so MedicalNet downloads
once per agent job and is reused across that agent's trials and folds without touching
the shared-home cache.

Each agent re-downloads torch, costing a few minutes of startup. That's the price
of not maintaining a custom image; if it becomes annoying, build one per
<https://wiki.cs.uni-saarland.de/en/HPC/dependency-management>.

### The sweep configs

`configs/resnet10_sweep.yaml` and `configs/efficientnet_b0_sweep.yaml` both target
the overfitting seen in the 5-fold baselines (train AUC 1.000/0.993 against val
0.792/0.735). 16 grid trials each; with `cv.enabled` that's 80 child runs plus 16
parents per sweep.

**`build_model` passes `model.params` straight to the constructor.** The per-model
allow-lists were removed, so a misnamed axis is now a `TypeError` at build time
rather than a silent no-op that burns a full 5-fold run. Consequences:

- **Neither model has a dropout axis.** MONAI's `ResNet` takes no dropout
  argument, and `EfficientNetBN` takes none either — it reads a fixed 0.2 from its
  per-variant params table. Since `model.params` is forwarded verbatim, `dropout`
  and `dropout_rate` are `TypeError`s, not options.
- MedicalNet fixes ResNet10 at `widen_factor: 1.0`; smaller variants have
  incompatible tensor shapes. Its sweep compares `freeze_backbone: true` against
  full fine-tuning instead.
- EfficientNetB0 has no capacity axis at all; `transforms.spatial_size` stands in.

Both sweeps are gitignored (`configs/*`), so `sync_to_hpc.sh` carries them, not git.

---

## `evaluate.sub` — evaluating a checkpoint

Whatever you put in `args` is forwarded to `evaluate.py` unchanged, so every flag
it accepts works:

```bash
# the headline number: a refit, thresholded from its CV sibling
condor_submit args="--checkpoint checkpoints/<refit_run_id>/last.pth \
    --threshold-from checkpoints/<cv_run_id> --log-wandb" condor/evaluate.sub

# one checkpoint file
condor_submit args="--checkpoint checkpoints/<run_id>/split_1/last.pth" condor/evaluate.sub

# a whole CV run on test: each fold, mean +/- std, plus the ensemble. Diagnostic
# only, hence the opt-in flag -- final reporting uses the refit above.
condor_submit args="--checkpoint checkpoints/<run_id> --allow-cv-test-evaluation \
    --config configs/resnet10.json" condor/evaluate.sub
```

Point `--checkpoint` at a single `.pth` for one model, or at a *run directory* — with
`--allow-cv-test-evaluation` — for the fold-by-fold report plus the ensemble. Without
that flag a run directory is refused, because evaluating every fold on test is a legacy
diagnostic rather than the reported result. Results are written to
`evaluations/<run_id>/` on NFS, so they outlive the job — read them from the
submit node afterwards.

For the benchmark arms, `submit_benchmark_refits.sh --evaluate` builds these invocations
and pairs each refit with its threshold donor for you.

`--config` only supplies evaluation-time settings (threshold, evaluation, device,
wandb, dataloader). Everything describing the trained model comes from the run's
own metadata, so you cannot accidentally evaluate under a different architecture
than was trained.

Requests 1 GPU and 16G. Inference is lighter than training, but a CV evaluation
still pushes the test set through K models.

---

## `train.sub` — a single training run

For one configuration, where a sweep's parent/child bookkeeping is just noise.
Whatever you put in `args` is forwarded to `train.py` unchanged:

```bash
# a 5-fold cross-validation run from a config
condor_submit args="--config configs/efficientnet_b0_bnfix.json" condor/train.sub

# pick up an interrupted CV run at the fold and epoch it stopped at
condor_submit args="--config configs/efficientnet_b0_bnfix.json --resume checkpoints/<run_id>" \
    condor/train.sub
```

Checkpoints land in `checkpoints/<run_id>/` on NFS, so they outlive the job.

Use this rather than a one-trial sweep when the question is "does this single
change work", since the answer is usually in the per-epoch validation curve of
each fold rather than in the aggregate the sweep optimises.

Requests the same 1 GPU / 24G as `sweep_agent.sub`: a CV run holds one model at a
time, so the ceiling is set by the largest configuration, not the fold count.

---

## Writing your own submit files

Every job **must** use the docker universe; anything else never matches a worker.

```
universe                = docker
docker_image            = python:3.14-slim
executable              = your_script.sh

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

request_GPUs            = 1
request_CPUs            = 1
request_memory          = 2G
request_disk            = 2G

# Both lines are required for /home to be visible inside the container.
requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true

queue
```

Things that bite:

- **`cd` to the repo inside your wrapper script.** In the docker universe the
  container starts in the job's scratch dir, *not* `initialdir`, so the configs'
  relative paths (`"image_glob": "data/T88_111_masked/*masked_gfc.img"`) resolve
  against the wrong directory and the glob silently matches zero files.
  `initialdir` still matters, but for the submit side: where a relative
  `executable` is found and where `output`/`error`/`log` land.
- **Relative vs absolute `executable`.** Relative is copied by Condor to the job
  scratch dir; absolute is interpreted as a path *inside* the container.
- **No root inside the container.** Jobs run as your user, so `pip install` needs
  `--target /tmp/pylibs` (with `PYTHONPATH`) or `--user`.
- **Resource requests.** Too small and the job is held; too large and you block
  other users. `/tmp` in the container is the job's scratch dir.
- **`request_GPUs = 1`** even for non-GPU probes, if the answer should reflect
  where training actually runs — the cluster also has CPU-only AMD nodes.

### GPU memory

`request_memory` is **host RAM** and has no effect on VRAM, so it is not the knob for
a CUDA OOM. Every submit file that runs the model — `train.sub`, `evaluate.sub` and
`sweep_agent.sub` — carries a `gpu_mem` macro that filters which GPUs may match
(`check_internet.sub` does not; it only probes connectivity):

```bash
condor_submit gpu_mem=20000 args="--config configs/resnet10.json" condor/train.sub
```

Default is `0`, which matches any GPU. Measured peak allocation at 96×128×96,
batch 8 — extrapolated from batch 1 and 2, then confirmed against the round-1 OOM
message's own "9.31 GiB is allocated by PyTorch":

| Config | Peak allocated | `gpu_mem` |
| --- | --- | --- |
| ResNet10 `widen_factor=0.25` | 2.4 G | default |
| ResNet10 `widen_factor=0.5` | 4.7 G | default |
| ResNet10 `widen_factor=1.0` | 9.3 G | `20000` |
| EfficientNetB0 | 4.5 G | default |

Add roughly 50% over the measured figure for fragmentation and the allocator's
reserve. Both wrappers also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
the round-1 OOM had only 9.31 GiB allocated but 14.03 GiB in use on a 15.89 GiB card,
so 4.37 GiB was reserved-but-unallocated fragmentation, and this is the setting
torch's own error message recommends for that gap. Between the two, the eight
`widen_factor=1.0` trials that died should now fit.

`require_gpus` needs HTCondor 9.9 or newer; the cluster runs 24.12.2 (checked
2026-08-23), so it is the supported idiom here and the older
`requirements = TARGET.CUDAGlobalMemoryMb >= N` form is not needed.

Before relying on a `gpu_mem` value, check it is satisfiable — a filter no worker can
meet leaves the job idle forever rather than failing:

```bash
condor_status -af Name GPUs_GlobalMemoryMb | sort -u -k2 -n
condor_q -better <jobid>          # if it sits idle, this says which constraint bit
```

### Storage

| Mount | Quota | Use for | Submit flag |
| --- | --- | --- | --- |
| `/home` | none (130 TiB) | Scripts, input data | `+WantGPUHomeMounted` |
| `/scratch` | 10 TiB per group | Checkpoints, I/O-heavy data (NVMe BeeGFS, ~15× faster) | `+WantScratchMounted` |
| `/tmp` | none | Job scratch, best latency, node-local | always mounted |

`/scratch` is short-term storage — copy anything you want to keep back to `/home`.
Worth moving `checkpoints/` there given the write volume.

### Managing jobs

| Command | Purpose |
| --- | --- |
| `condor_q` | State of your jobs |
| `condor_q -hold <jobid>` | Why a job is held — always start here |
| `condor_q -analyze` / `-better` | Why a job isn't matching any worker |
| `condor_qedit <jobid> RequestMemory 4096` | Adjust a requirement in place |
| `condor_release <jobid>` | Return a held job to idle after fixing it |
| `condor_rm <jobid>` | Remove one job |
| `condor_status` | Claimed vs idle worker nodes |
| `condor_submit -i <file>.sub` | Interactive job for debugging |

Interactive jobs are capped at 1 GPU / 1 CPU, limited to 4 in parallel cluster-wide,
and **killed after 1 hour**. Use one to confirm the glob finds 235 volumes before
queueing real training.

If you're on a shared team account, `condor_q` shows every member's jobs and
`condor_rm -a` would remove theirs too — always remove by explicit job ID.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Job stays `idle` | No matching worker free yet. Check with `condor_q -better`. |
| Job goes to `held` | `condor_q -hold <jobid>`. Usually memory ("gone over memory limit"), a missing input file, or a non-executable script. |
| First run slow | The worker has to pull the docker image. |
| `data/` empty inside the job | Missing `+WantGPUHomeMounted` or the `UidDomain` requirement. |
| Glob matches nothing | `initialdir` not set to the repo root. |

## Further reading

- HTCondor manual: <https://htcondor.readthedocs.io/en/latest/>
- SIC HPC FAQ: <https://wiki.cs.uni-saarland.de/en/HPC/faq>
- Machine list: <https://wiki.cs.uni-saarland.de/en/HPC/machine-list>
- Storage: <https://wiki.cs.uni-saarland.de/en/HPC/storage>
- Dependency management: <https://wiki.cs.uni-saarland.de/en/HPC/dependency-management>
- Job monitoring (ClusterCockpit): <https://hpc-monitoring.cs.uni-saarland.de>
- Tickets: <https://ticket.cs.uni-saarland.de>
