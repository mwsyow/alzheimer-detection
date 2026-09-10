import io
import json
import tarfile

import pytest
import torch

from package_sweep import package, verify


def pilot(root, run_id, seed):
    directory = root / "checkpoints" / run_id
    directory.mkdir(parents=True)
    config = {
        "model": {"name": "Simple3DCNN"},
        "seed": seed,
        "cv": {"enabled": True, "random_seed": seed},
        "evaluation": {"output_dir": "evaluations"},
    }
    torch.save(
        {
            "run_id": run_id,
            "config": config,
            "cv": {"n_splits": 2, "completed_folds": [1, 2]},
        },
        directory / "metadata.pth",
    )
    (directory / "weights.pth").write_bytes(b"test weights")
    return directory


def test_roundtrip_groups_seeds_and_reports_missing(tmp_path):
    source = tmp_path / "source"
    runs = [pilot(source, "one", 42), pilot(source, "two", 7)]
    bundle = tmp_path / "pilot.tar.zst"
    manifest = package(source, runs, bundle)
    assert len({r["configuration_sha256"] for r in manifest["runs"]}) == 1
    assert all(r["missing"] == ["artifacts", "evaluations"] for r in manifest["runs"])
    restored = tmp_path / "restored"
    verify(bundle, restored)
    for item in manifest["files"]:
        assert (restored / item["original_path"]).read_bytes() == (
            source / item["original_path"]
        ).read_bytes()
    with pytest.raises(ValueError, match="empty"):
        verify(bundle, restored)
    with pytest.raises(FileExistsError):
        package(source, runs, bundle)


@pytest.mark.parametrize(
    "original,checksum", [("../escape", "invalid"), ("weights.pth", "invalid")]
)
def test_rejects_traversal_and_corruption(tmp_path, original, checksum):
    bundle = tmp_path / "bad.tar.zst"
    manifest = {
        "runs": [],
        "files": [
            {
                "path": "payload",
                "original_path": original,
                "size": 3,
                "sha256": checksum,
            }
        ],
    }
    with tarfile.open(bundle, "w|zst") as archive:
        for name, data in (
            ("manifest.json", json.dumps(manifest).encode()),
            ("payload", b"bad"),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    with pytest.raises(ValueError):
        verify(bundle)


def test_refuses_incomplete_run(tmp_path):
    directory = pilot(tmp_path, "one", 42)
    metadata = torch.load(directory / "metadata.pth", weights_only=False)
    metadata["cv"]["completed_folds"] = [1]
    torch.save(metadata, directory / "metadata.pth")
    with pytest.raises(ValueError, match="Incomplete"):
        package(tmp_path, [directory], tmp_path / "out.tar.zst")
