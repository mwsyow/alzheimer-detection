"""Opt-in rotating-CV performance settings must preserve the learning task."""

import copy

import pytest
import torch
from monai.data.utils import pickle_hashing
from monai.transforms import Compose, EnsureTyped, RandGaussianNoised, ScaleIntensityd

from datasets import CachedDataset, Dataset, DatasetSource
from training_performance import EpochSampler, input_cache_hash
from metrics import collect_predictions


def data():
    return [
        {
            "image": torch.arange(64).float().reshape(1, 4, 4, 4),
            "label": i % 2,
            "age": 60 + i,
        }
        for i in range(8)
    ]


def transform():
    return Compose([ScaleIntensityd("image"), EnsureTyped("image")])


def test_cache_matches_uncached_and_reuses_files(tmp_path):
    uncached = Dataset(data(), transform())
    cached = CachedDataset(
        data(), transform(), cache_dir=tmp_path, hash_transform=pickle_hashing
    )
    for i in range(8):
        torch.testing.assert_close(uncached[i][0], cached[i][0])
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    for i in range(8):
        torch.testing.assert_close(uncached[i][0], cached[i][0])
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


def test_cache_does_not_freeze_random_augmentation(tmp_path):
    transforms = Compose(
        [ScaleIntensityd("image"), RandGaussianNoised("image", prob=1, std=0.1)]
    )
    transforms.set_random_state(42)
    cached = CachedDataset(
        data(), transforms, cache_dir=tmp_path, hash_transform=pickle_hashing
    )
    cached.transform.set_random_state(42)
    first, second = cached[0][0], cached[0][0]
    assert not torch.equal(first, second)
    transforms.set_random_state(42)
    torch.testing.assert_close(first, cached[0][0])


def test_transform_changes_invalidate_cache(tmp_path):
    first = CachedDataset(
        data(), transform(), cache_dir=tmp_path, hash_transform=pickle_hashing
    )
    second = CachedDataset(
        data(),
        Compose([ScaleIntensityd("image", minv=1, maxv=2)]),
        cache_dir=tmp_path,
        hash_transform=pickle_hashing,
    )
    assert first.transform_hash != second.transform_hash
    assert not torch.equal(first[0][0], second[0][0])


def test_file_and_header_changes_invalidate_cache(tmp_path):
    image = tmp_path / "scan.img"
    header = image.with_suffix(".hdr")
    image.write_bytes(b"image")
    header.write_bytes(b"header")
    item = {"image": str(image), "label": 0}
    first = input_cache_hash(item)
    header.write_bytes(b"changed-header")
    second = input_cache_hash(item)
    image.write_bytes(b"changed-image")
    assert len({first, second, input_cache_hash(item)}) == 3


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_order_independent_of_persistence_and_repeatable(tmp_path, workers):
    def loader(optimized):
        source = DatasetSource.__new__(DatasetSource)
        source.config = {
            "seed": 42,
            "dataloader": {"batch_size": 2, "num_workers": workers},
            "performance": {"enabled": optimized},
        }
        source._datasets = {"train": Dataset(data(), transform())}
        return source.loader(list(range(8)), "train", shuffle=True)

    left, right = loader(False), loader(True)
    for epoch in range(3):
        left.sampler.set_epoch(epoch)
        right.sampler.set_epoch(epoch)
        assert list(left.sampler) == list(right.sampler)
        left_batches = list(left)
        for a, b in zip(left_batches, right, strict=True):
            torch.testing.assert_close(a[0], b[0])
            torch.testing.assert_close(a[1], b[1])
    assert right.persistent_workers == (workers > 0)


def test_epoch_sampler_resume_order():
    first, resumed = EpochSampler(20, 42), EpochSampler(20, 42)
    initial = list(first)
    first.set_epoch(4)
    resumed.set_epoch(4)
    assert list(first) == list(resumed) != initial


def test_age_cached_target_is_float(tmp_path):
    cached = CachedDataset(data(), transform(), cache_dir=tmp_path, target_key="age")
    assert cached[0][1].dtype == torch.float32
    assert cached[0][1].item() == 60


def test_optimized_prediction_collection_matches_on_cpu():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    loader = torch.utils.data.DataLoader(Dataset(data(), transform()), batch_size=2)
    left = collect_predictions(
        model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    right = collect_predictions(
        copy.deepcopy(model),
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimized=True,
    )
    assert left["loss"] == right["loss"]
    torch.testing.assert_close(
        torch.tensor(left["y_prob"]), torch.tensor(right["y_prob"])
    )
