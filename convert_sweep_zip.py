"""Convert a sweep tar.zst bundle to a portable ZIP64 archive without staging files."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

from package_sweep import relative, sha256


def convert(source, output):
    partial = output.with_name(output.name + ".partial")
    checksum = output.with_name(output.name + ".sha256")
    if any(p.exists() for p in (output, partial, checksum)):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        tarfile.open(source, "r|zst") as archive,
        zipfile.ZipFile(
            partial,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as target,
    ):
        first = archive.next()
        if first is None or first.name != "manifest.json" or not first.isfile():
            raise ValueError("Missing manifest")
        encoded = archive.extractfile(first).read()
        manifest = json.loads(encoded)
        expected = {item["path"]: item for item in manifest["files"]}
        if len(expected) != len(manifest["files"]):
            raise ValueError("Duplicate manifest paths")
        target.writestr("manifest.json", encoded)
        seen = {"manifest.json"}
        # archive.next() avoids revisiting the first member in streaming mode.
        while member := archive.next():
            relative(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError(f"Unsafe or duplicate entry: {member.name}")
            seen.add(member.name)
            item = expected.get(member.name)
            if item is None and not member.name.endswith("/config.json"):
                raise ValueError(f"Unexpected entry: {member.name}")
            if item and member.size != item["size"]:
                raise ValueError(f"Size mismatch: {member.name}")
            digest = hashlib.sha256()
            with (
                archive.extractfile(member) as reader,
                target.open(member.name, "w", force_zip64=True) as writer,
            ):
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
            if item and digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {member.name}")
            if len(seen) % 500 == 0:
                print(f"Converted {len(seen)} entries", flush=True)
        if expected.keys() - seen:
            raise ValueError("Missing payloads")
    print("Checking ZIP integrity...", flush=True)
    with zipfile.ZipFile(partial) as archive:
        failed = archive.testzip()
        if failed:
            raise ValueError(f"ZIP integrity failure: {failed}")
    partial.rename(output)
    with checksum.open("x") as stream:
        stream.write(f"{sha256(output)}  {output.name}\n")
    print(
        f"Saved {output}: {output.stat().st_size:,} bytes; {len(expected)} verified payloads",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.with_name(
        args.source.name.removesuffix(".tar.zst") + ".zip"
    )
    convert(args.source, output)
