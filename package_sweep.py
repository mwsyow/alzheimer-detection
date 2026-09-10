"""Package completed sweeps or local pilot runs into configuration-grouped tar.zst files."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def package(root, directories, output, sweep_id=None):
    import csv
    import torch
    from artifacts import comparison_fingerprint

    root = root.resolve()
    partial = output.with_name(output.name + ".partial")
    checksum_path = output.with_name(output.name + ".sha256")
    if any(p.exists() for p in (output, partial, checksum_path)):
        raise FileExistsError(f"Output already exists: {output}")
    if not directories:
        raise ValueError("No runs selected")
    manifest = {
        "version": 1,
        "sweep_id": sweep_id,
        "source_root": str(root),
        "runs": [],
        "files": [],
    }
    generated = {}
    seen = set()

    def collect(directory, prefix):
        directory = directory.resolve()
        directory.relative_to(root)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unsupported symlink: {path}")
            if path.is_file():
                manifest["files"].append(
                    {
                        "path": f"{prefix}/{path.relative_to(directory).as_posix()}",
                        "original_path": path.relative_to(root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

    for directory in directories:
        directory = directory.resolve()
        directory.relative_to(root)
        metadata = torch.load(
            directory / "metadata.pth", map_location="cpu", weights_only=False
        )
        run_id = metadata["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or run_id in seen:
            raise ValueError(f"Invalid or duplicate run ID: {run_id}")
        seen.add(run_id)
        config = metadata["config"]
        cv = metadata.get("cv", {})
        if config.get("cv", {}).get("enabled") and len(
            cv.get("completed_folds", [])
        ) != cv.get("n_splits"):
            raise ValueError(f"Incomplete CV run: {run_id}")
        if sweep_id and metadata.get("sweep_id") not in (
            sweep_id,
            sweep_id.split("/")[-1],
        ):
            raise ValueError(f"Sweep mismatch: {run_id}")
        group = comparison_fingerprint(config)
        prefix = f"configurations/{group}/runs/{run_id}"
        generated[f"{prefix}/config.json"] = config
        record = {
            "run_id": run_id,
            "configuration_sha256": group,
            "path": prefix,
            "seed": config.get("seed"),
            "cv_random_seed": config.get("cv", {}).get("random_seed"),
            "architecture": config["model"]["name"],
            "adaptation": config["model"].get("pretrained", {}),
            "missing": [],
        }
        collect(directory, f"{prefix}/checkpoints")
        artifact = cv.get(
            "artifact_root",
            str(Path(config.get("artifacts", {}).get("dir", "artifacts")) / run_id),
        )
        evaluation = (
            Path(config.get("evaluation", {}).get("output_dir", "evaluations")) / run_id
        )
        for category, location in (
            ("artifacts", artifact),
            ("evaluations", evaluation),
        ):
            if (root / location).is_dir():
                collect(root / location, f"{prefix}/{category}")
            else:
                record["missing"].append(category)
        manifest["runs"].append(record)
        print(
            f"Indexed {run_id}: {record['architecture']}; missing={record['missing']}",
            flush=True,
        )
    for path in sorted((root / "evaluations/comparisons").glob("*/runs.csv")):
        with path.open() as stream:
            ids = {row["run_id"] for row in csv.DictReader(stream)}
        if ids == seen:
            collect(path.parent, f"comparisons/{path.parent.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(partial, "w|zst") as archive:
        for name, data in {"manifest.json": manifest, **generated}.items():
            encoded = json.dumps(data, indent=2).encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(encoded)
            archive.addfile(entry, io.BytesIO(encoded))
        for item in manifest["files"]:
            source = root / item["original_path"]
            before = source.stat()
            if before.st_size != item["size"] or sha256(source) != item["sha256"]:
                raise ValueError(f"Source changed: {source}")
            archive.add(source, arcname=item["path"], recursive=False)
            if source.stat().st_mtime_ns != before.st_mtime_ns:
                raise ValueError(f"Source changed: {source}")
    partial.rename(output)
    with checksum_path.open("x") as stream:
        stream.write(f"{sha256(output)}  {output.name}\n")
    print(f"Saved {output} ({output.stat().st_size:,} bytes)", flush=True)
    return manifest


def verify(bundle, restore_to=None):
    """Verify payload hashes; optionally restore original repository paths."""
    if restore_to is not None:
        restore_to = restore_to.resolve()
        if restore_to.exists() and any(restore_to.iterdir()):
            raise ValueError("Restore destination must be empty")
    with tarfile.open(bundle, "r|zst") as archive:
        first = archive.next()
        if first is None or first.name != "manifest.json":
            raise ValueError("Missing manifest")
        manifest = json.load(archive.extractfile(first))
        expected = {item["path"]: item for item in manifest["files"]}
        seen = set()
        if len(expected) != len(manifest["files"]):
            raise ValueError("Duplicate manifest entries")
        for member in archive:
            if member.name == "manifest.json":
                continue
            relative(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError(f"Invalid entry: {member.name}")
            seen.add(member.name)
            item = expected.get(member.name)
            if item is None:
                if not member.name.endswith("/config.json"):
                    raise ValueError(f"Unexpected entry: {member.name}")
                continue
            if member.size != item["size"]:
                raise ValueError(f"Size mismatch: {member.name}")
            original = relative(item["original_path"])
            target = None
            if restore_to is not None:
                destination = restore_to / original
                destination.parent.mkdir(parents=True, exist_ok=True)
                target = destination.open("xb")
            checksum = hashlib.sha256()
            try:
                with archive.extractfile(member) as source:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        if target:
                            target.write(chunk)
            finally:
                if target:
                    target.close()
            if checksum.hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {member.name}")
        if expected.keys() - seen:
            raise ValueError("Missing payload files")
    print(
        f"Verified {len(expected)} files across {len(manifest['runs'])} runs",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--sweep-id", help="W&B entity/project/sweep")
    choice.add_argument(
        "--run-dir", action="append", type=Path, help="Local pilot run; repeat per run"
    )
    choice.add_argument("--verify", type=Path)
    parser.add_argument(
        "--restore-to",
        type=Path,
        help="With --verify, restore original paths into an empty folder",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify, args.restore_to)
        return
    if args.restore_to:
        parser.error("--restore-to requires --verify")
    root = args.repo_root.resolve()
    if args.sweep_id:
        import wandb
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
        runs = list(wandb.Api().sweep(args.sweep_id).runs)
        unfinished = [run.id for run in runs if run.state != "finished"]
        if unfinished:
            raise ValueError(f"Unfinished sweep runs: {unfinished}")
        directories = [root / args.checkpoint_root / run.id for run in runs]
    else:
        directories = [root / path for path in args.run_dir]
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", args.sweep_id or "local_pilot")
    output = args.output or root / "bundles" / f"sweep_{slug}.tar.zst"
    package(root, directories, output, args.sweep_id)


if __name__ == "__main__":
    main()
