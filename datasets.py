from pathlib import Path

import pandas as pd
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import train_test_split


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


class Dataset(MonaiDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return item["image"], item["label"]
