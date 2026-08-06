from pathlib import Path
import glob

import pandas as pd
import torch
from monai.data import DataLoader, Dataset as MonaiDataset, NibabelReader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def get_label(df: pd.DataFrame, path: Path):
    subject_id = "_".join(str(path.name).split("_")[:3])
    cdr = df[df["ID"] == subject_id]["CDR"].values[0]
    return int(bool(cdr))


def get_data(img_paths: list[Path], label_path: Path):
    df = pd.read_excel(label_path)
    dataset_items = [
        {
            "label": get_label(df, path),
            "image": str(path),
        }
        for path in img_paths
    ]
    return dataset_items


def build_dataset_items(config: dict):
    img_paths = [Path(path) for path in sorted(glob.glob(config["image_glob"]))]
    if not img_paths:
        raise FileNotFoundError(
            f"No MRI images matched image_glob={config['image_glob']!r}"
        )
    return get_data(img_paths, Path(config["label_path"]))


def stratified_three_way_split(
    dataset_items: list[dict],
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int = None,
):
    total_size = train_size + val_size + test_size
    if not abs(total_size - 1.0) < 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total_size}"
        )

    indices = list(range(len(dataset_items)))
    labels = [item["label"] for item in dataset_items]
    temp_size = val_size + test_size
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_size,
        random_state=random_seed,
        stratify=labels,
    )

    temp_labels = [dataset_items[idx]["label"] for idx in temp_idx]
    relative_test_size = test_size / temp_size
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=temp_labels,
    )

    return train_idx, val_idx, test_idx


def build_split_indices(dataset_items: list[dict], config: dict):
    split_config = config["split"]
    train_idx, val_idx, test_idx = stratified_three_way_split(
        dataset_items=dataset_items,
        train_size=split_config["train_size"],
        val_size=split_config["val_size"],
        test_size=split_config["test_size"],
        random_seed=split_config["random_seed"],
    )
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def build_transforms(config: dict, mode: str):
    transform_config = config["transforms"]
    transforms = [
        LoadImaged(
            keys=["image"],
            reader=NibabelReader(squeeze_non_spatial_dims=True),
            image_only=True,
        ),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ]
    if transform_config.get("resize", False):
        transforms.append(
            Resized(
                keys=["image"],
                spatial_size=tuple(transform_config["spatial_size"]),
                mode=transform_config.get("resize_mode", "trilinear"),
            )
        )
    if transform_config.get("scale_intensity", False):
        transforms.append(ScaleIntensityd(keys=["image"]))
    if transform_config.get("normalize_intensity", False):
        transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=transform_config.get("normalize_nonzero", True),
                channel_wise=transform_config.get("normalize_channel_wise", True),
            )
        )
    if mode == "train" and transform_config.get("rand_rotate90", False):
        transforms.append(
            RandRotate90d(
                keys=["image"],
                prob=transform_config.get("rand_rotate90_prob", 0.5),
                spatial_axes=tuple(
                    transform_config.get("rand_rotate90_spatial_axes", [0, 2])
                ),
            )
        )
    transforms.extend(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ]
    )
    return Compose(transforms)


def build_loader(
    dataset_items: list[dict],
    indices: list[int],
    config: dict,
    mode: str,
    shuffle: bool = False,
):
    mri_dataset = Dataset(data=dataset_items, transform=build_transforms(config, mode))
    dataset = Subset(mri_dataset, indices)
    dataloader_config = config["dataloader"]
    return DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=shuffle,
        num_workers=dataloader_config["num_workers"],
    )


def build_train_val_loaders(
    dataset_items: list[dict],
    split_indices: dict[str, list[int]],
    config: dict,
):
    train_loader = build_loader(
        dataset_items=dataset_items,
        indices=split_indices["train_idx"],
        config=config,
        mode="train",
        shuffle=True,
    )
    val_loader = build_loader(
        dataset_items=dataset_items,
        indices=split_indices["val_idx"],
        config=config,
        mode="val",
        shuffle=False,
    )
    return train_loader, val_loader


def build_test_loader(config: dict, test_idx: list[int]):
    dataset_items = build_dataset_items(config)
    test_loader = build_loader(
        dataset_items=dataset_items,
        indices=test_idx,
        config=config,
        mode="test",
        shuffle=False,
    )
    return test_loader, dataset_items


class Dataset(MonaiDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return item["image"], item["label"]
