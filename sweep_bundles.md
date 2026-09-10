# Sweep bundles

Run from the repository root with the existing Python 3.14 environment. Package
only completed sweeps whose output files are no longer being written:

```bash
uv run python package_sweep.py \
  --sweep-id wmarcellius123/alzheimer-detection/1gi97aqb \
  --output bundles/sweep_1gi97aqb.tar.zst
```

The command reads W&B membership and local checkpoint metadata. It groups trials
by the same comparison hash used in evaluation (seeds excluded), stores each
run's complete resolved `config.json`, checkpoints, artifacts and evaluations,
and includes any local comparison folder with exactly the selected run IDs.
It packages existing evaluations; run `evaluate.py --sweep-id ...` first to create
them. Missing artifact/evaluation directories are printed and recorded in the
manifest. Missing checkpoints or unfinished runs fail the command.

The archive starts with `manifest.json`: sweep identity, configuration groups,
run IDs, seeds, architectures, adaptation settings, missing output categories,
original file paths, file sizes and SHA-256 checksums. Payloads are under
`configurations/<hash>/runs/<run_id>/{checkpoints,artifacts,evaluations}`.
Compression streams to disk without creating an uncompressed staging copy.
An interrupted/failed build leaves `.partial`; completed archives are never
overwritten. A separate `.sha256` file records the whole archive checksum.

Download from the local checkout:

```bash
./condor/sync_to_hpc.sh lpnn26_team005 --from-hpc --only \
  bundles/sweep_1gi97aqb.tar.zst bundles/sweep_1gi97aqb.tar.zst.sha256
```

Verify the contents, without extracting:

```bash
uv run python package_sweep.py --verify bundles/sweep_1gi97aqb.tar.zst
```

Restore the original repository-relative paths into an empty directory:

```bash
uv run python package_sweep.py --verify bundles/sweep_1gi97aqb.tar.zst \
  --restore-to bundles/restored_1gi97aqb
```

Restoration checks every payload checksum while writing. If verification fails,
the destination is incomplete and should not be used. Original files and stored
metadata are preserved byte-for-byte. Repository-relative artifact references
work when using the restored directory as the working directory. Absolute paths
recorded on HPC are preserved and may require relocation before local evaluation.
Plain tar extraction instead produces the configuration-grouped browsing layout.

For offline pilots, repeat `--run-dir` instead of supplying a W&B sweep ID:

```bash
uv run python package_sweep.py \
  --run-dir checkpoints/pilot_all_architectures_3ep/simple3dcnn/x61gaty7 \
  --run-dir checkpoints/pilot_all_architectures_3ep/resnet10/u6nqz5fk \
  --run-dir checkpoints/pilot_all_architectures_3ep/densenet121/vvu2cm72 \
  --run-dir checkpoints/pilot_all_architectures_3ep/efficientnet_b0/uyt3gjfq \
  --output bundles/pilot_all_architectures_3ep.tar.zst
```
