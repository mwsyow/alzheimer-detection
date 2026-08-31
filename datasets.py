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
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
            "image_id": str(path),
        }
        for path in img_paths
    ]
    return dataset_items


def resolve_dataset_config(config: dict) -> dict:
    """The "dataset" block, or the legacy top-level keys it replaced.

    Checkpoint metadata written before the block existed still carries image_glob and
    label_path at the top level, so those runs stay resumable and evaluable.
    """
    dataset_config = config.get("dataset")
    if dataset_config:
        return {"name": "oasis", **dataset_config}
    return {
        "name": "oasis",
        "image_glob": config["image_glob"],
        "label_path": config["label_path"],
    }


class DatasetBackend:
    """Where samples come from, and what it takes to turn one into a tensor.

    A backend owns three things the rest of the pipeline should not have to know
    about: how to enumerate samples, whether they still need reading from disk, and
    how to name one in a prediction dump. Everything downstream -- splitting,
    cross-validation, training, evaluation -- works the same for every backend.
    """

    name = "base"

    def __init__(self, dataset_config: dict):
        self.dataset_config = dataset_config

    def build_items(self) -> list[dict]:
        raise NotImplementedError

    def load_transforms(self) -> list:
        """Transforms that materialise "image" before the shared pipeline runs."""
        return []

    def item_id(self, item: dict) -> str:
        return str(item.get("image_id", item["image"]))


class OasisBackend(DatasetBackend):
    """OASIS MRI volumes on disk, labelled by CDR from the spreadsheet."""

    name = "oasis"

    def build_items(self) -> list[dict]:
        image_glob = self.dataset_config["image_glob"]
        img_paths = [Path(path) for path in sorted(glob.glob(image_glob))]
        if not img_paths:
            raise FileNotFoundError(f"No MRI images matched image_glob={image_glob!r}")
        return get_data(img_paths, Path(self.dataset_config["label_path"]))

    def load_transforms(self) -> list:
        return [
            LoadImaged(
                keys=["image"],
                reader=NibabelReader(squeeze_non_spatial_dims=True),
                image_only=True,
            )
        ]


class MedMNISTBackend(DatasetBackend):
    """MedMNIST v2 3D volumes, held in memory rather than read from disk.

    Every official split is pooled into one list so the project's own stratified split
    and cross-validation apply uniformly; set split.source to "dataset" to fall back to
    MedMNIST's published train/val/test partition instead.

    include_labels subsets the original classes and positive_labels maps them to a
    binary target, which is what turns a multi-class set such as organmnist3d into a
    balanced binary task.
    """

    name = "medmnist"

    def build_items(self) -> list[dict]:
        # Imported lazily so the OASIS path works without medmnist installed.
        import medmnist
        from medmnist import INFO

        config = self.dataset_config
        flag = config["flag"]
        if flag not in INFO:
            raise ValueError(f"Unknown MedMNIST flag {flag!r}. Options: {sorted(INFO)}")

        root = Path(config.get("root", "data/medmnist"))
        root.mkdir(parents=True, exist_ok=True)

        dataset_class = getattr(medmnist, INFO[flag]["python_class"])
        load_kwargs = {"root": str(root), "download": config.get("download", True)}
        size = config.get("size")
        if size is not None:
            # MedMNIST+ resolutions; 28 is the default and takes no size argument.
            load_kwargs["size"] = size

        include_labels = config.get("include_labels")
        positive_labels = config.get("positive_labels")

        items = []
        for split in ("train", "val", "test"):
            partition = dataset_class(split=split, **load_kwargs)
            for index, (image, label) in enumerate(
                zip(partition.imgs, partition.labels)
            ):
                original = int(label.ravel()[0])
                if include_labels is not None and original not in include_labels:
                    continue
                target = (
                    int(original in positive_labels)
                    if positive_labels is not None
                    else original
                )
                items.append(
                    {
                        "image": image,
                        "label": target,
                        "image_id": f"{flag}/{split}/{index}",
                        "split": split,
                        "original_label": original,
                    }
                )

        if not items:
            raise ValueError(
                f"No samples left for {flag} after include_labels={include_labels!r}."
            )
        labels = {item["label"] for item in items}
        if len(labels) < 2:
            raise ValueError(
                f"{flag} reduced to a single class {labels}; check include_labels and "
                "positive_labels."
            )
        return items


DATASET_BACKENDS = {
    OasisBackend.name: OasisBackend,
    MedMNISTBackend.name: MedMNISTBackend,
}


def build_backend(config: dict) -> DatasetBackend:
    dataset_config = resolve_dataset_config(config)
    name = dataset_config["name"]
    if name not in DATASET_BACKENDS:
        raise ValueError(
            f"Unsupported dataset: {name!r}. Options: {sorted(DATASET_BACKENDS)}"
        )
    return DATASET_BACKENDS[name](dataset_config)


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


def uses_dataset_split(config: dict) -> bool:
    """Whether to honour a dataset's own published train/val/test partition."""
    return config["split"].get("source") == "dataset"


def dataset_split_indices(dataset_items: list[dict]):
    """Group indices by the "split" each item was published under."""
    grouped = {"train": [], "val": [], "test": []}
    for index, item in enumerate(dataset_items):
        split = item.get("split")
        if split not in grouped:
            raise ValueError(
                'split.source="dataset" needs every item to carry a "split" of '
                f"train/val/test; item {index} has {split!r}. The OASIS backend does "
                "not provide one."
            )
        grouped[split].append(index)
    return grouped["train"], grouped["val"], grouped["test"]


def build_split_indices(dataset_items: list[dict], config: dict):
    split_config = config["split"]
    if uses_dataset_split(config):
        train_idx, val_idx, test_idx = dataset_split_indices(dataset_items)
        return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

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


def is_cv_enabled(config: dict) -> bool:
    """Cross-validation is switched on by the presence of a "cv" block.

    An explicit "enabled": false turns it off without having to delete the block.
    """
    cv_config = config.get("cv")
    if not cv_config:
        return False
    return bool(cv_config.get("enabled", True))


def is_refit_enabled(config: dict) -> bool:
    """Refit is switched on by the presence of a "refit" block, exactly like "cv".

    An explicit "enabled": false turns it off without having to delete the block.
    """
    refit_config = config.get("refit")
    if not refit_config:
        return False
    return bool(refit_config.get("enabled", True))


SPLIT_MODES = ("single", "cv", "refit")


def resolve_split_mode(config: dict) -> str:
    """Which of the three ways to partition the data this config asks for.

    Lives here rather than in train.py so the whole split policy is in one file, and so
    it can be tested without wandb, a model, or the images.
    """
    cv, refit = is_cv_enabled(config), is_refit_enabled(config)
    if cv and refit:
        raise ValueError(
            'cv.enabled and refit.enabled are both true, but they are alternatives: '
            "cross-validation refolds the pooled train and val indices, while refit "
            "trains one model on all of them. Enable exactly one."
        )
    if cv:
        return "cv"
    if refit:
        return "refit"
    return "single"


def pool_and_test_indices(dataset_items: list[dict], config: dict) -> dict:
    """The held-out test set, and everything else pooled into one training set.

    The single seam through which both cross-validation and refit obtain the test set,
    so it is by construction the same set the single-split path produces and results
    stay directly comparable across all three.

    Returns {"pool_idx", "pool_labels", "test_idx"}.
    """
    split_config = config["split"]

    if uses_dataset_split(config):
        # Keep the published test split intact, pool everything else.
        train_idx, val_idx, test_idx = dataset_split_indices(dataset_items)
    else:
        train_idx, val_idx, test_idx = stratified_three_way_split(
            dataset_items=dataset_items,
            train_size=split_config["train_size"],
            val_size=split_config["val_size"],
            test_size=split_config["test_size"],
            random_seed=split_config["random_seed"],
        )

    pool_idx = sorted(train_idx + val_idx)
    return {
        "pool_idx": pool_idx,
        "pool_labels": [dataset_items[idx]["label"] for idx in pool_idx],
        "test_idx": test_idx,
    }


def build_refit_split_indices(dataset_items: list[dict], config: dict):
    """One training set of everything that is not test, and no validation split.

    For the final model: nothing is held back except the test set, so there is no
    validation curve to select an epoch or an operating point from. Both come from the
    cross-validation run of the same architecture instead.

    Deliberately routed through the same three-way split as every other path rather than
    a two-way one. stratified_three_way_split carves off val+test together and then
    halves it, so a two-way train_test_split at the same test_size and seed lands on a
    *different* test set -- measured at 23 of 36 volumes in common on OASIS -- and every
    number reported against it would be incomparable with the runs already on disk.
    Setting split.val_size to 0 does not work either: relative_test_size becomes 1.0 and
    scikit-learn rejects it.
    """
    pooled = pool_and_test_indices(dataset_items, config)
    return {
        "train_idx": pooled["pool_idx"],
        "val_idx": [],
        "test_idx": pooled["test_idx"],
    }


def build_cv_split_indices(dataset_items: list[dict], config: dict):
    """Hold the test set out once, then stratified K-fold over everything else.

    The test set comes from the same stratified_three_way_split used by the
    single-split path, so it is identical to the one every non-CV run has used and
    results stay directly comparable. The train and val halves are pooled and refolded.
    """
    cv_config = config["cv"]
    split_config = config["split"]

    pooled = pool_and_test_indices(dataset_items, config)
    pool_idx = pooled["pool_idx"]
    pool_labels = pooled["pool_labels"]
    test_idx = pooled["test_idx"]

    n_splits = cv_config["n_splits"]
    if n_splits < 2:
        raise ValueError(f"cv.n_splits must be at least 2, got {n_splits}")

    shuffle = cv_config.get("shuffle", True)
    random_seed = cv_config.get("random_seed", split_config["random_seed"])
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        # scikit-learn rejects a random_state when shuffle is off.
        random_state=random_seed if shuffle else None,
    )

    folds = [
        {
            "train_idx": [pool_idx[i] for i in fold_train],
            "val_idx": [pool_idx[i] for i in fold_val],
        }
        for fold_train, fold_val in splitter.split(pool_idx, pool_labels)
    ]

    return {"test_idx": test_idx, "folds": folds}


# Every key build_transforms reads. A "transforms" block naming anything else is a
# typo, and unlike model.params -- which reaches a constructor and raises TypeError --
# an unrecognised transform key would otherwise be a silent no-op: the sweep runs to
# completion and every trial is identical, with nothing in the logs saying why.
KNOWN_TRANSFORM_KEYS = frozenset(
    {
        "resize",
        "spatial_size",
        "resize_mode",
        "scale_intensity",
        "normalize_intensity",
        "normalize_nonzero",
        "normalize_channel_wise",
        "rand_flip",
        "rand_flip_prob",
        "rand_flip_spatial_axis",
        "rand_affine",
        "rand_affine_prob",
        "rand_affine_rotate_range",
        "rand_affine_scale_range",
        "rand_affine_translate_range",
        "rand_affine_mode",
        "rand_affine_padding_mode",
        "rand_rotate90",
        "rand_rotate90_prob",
        "rand_rotate90_spatial_axes",
        "rand_bias_field",
        "rand_bias_field_prob",
        "rand_bias_field_degree",
        "rand_bias_field_coeff_range",
        "rand_gaussian_noise",
        "rand_gaussian_noise_prob",
        "rand_gaussian_noise_std",
        "rand_scale_intensity",
        "rand_scale_intensity_prob",
        "rand_scale_intensity_factors",
        "rand_shift_intensity",
        "rand_shift_intensity_prob",
        "rand_shift_intensity_offsets",
    }
)


def validate_transform_config(transform_config: dict) -> None:
    unknown = sorted(set(transform_config) - KNOWN_TRANSFORM_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown transforms key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(KNOWN_TRANSFORM_KEYS))}"
        )


def build_transforms(backend: DatasetBackend, config: dict, mode: str):
    """The transform pipeline for one mode. Augmentation is train-only.

    Ordering is not cosmetic. Spatial augmentation runs before intensity handling, so
    normalisation sees the volume the network will actually be given. RandBiasFieldd
    sits *before* normalisation because it models a multiplicative scanner
    inhomogeneity on raw intensities; the noise/scale/shift group sits *after*, because
    their magnitudes are only meaningful relative to unit variance.

    RandRotate90d moved from the end of the pipeline into the spatial group. That is a
    no-op for existing configs: a 90-degree rotation permutes voxels, so the nonzero
    set NormalizeIntensityd reduces over is unchanged, and normalisation commutes with
    it exactly. Everything else here defaults to off, so a config that predates these
    keys builds the pipeline it always did.
    """
    transform_config = config["transforms"]
    validate_transform_config(transform_config)
    augment = mode == "train"

    def enabled(key: str) -> bool:
        return augment and transform_config.get(key, False)

    def setting(key: str, default):
        return transform_config.get(key, default)

    # The backend contributes whatever it takes to materialise "image"; everything
    # after that is shared and config-driven.
    transforms = list(backend.load_transforms())
    transforms.append(EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"))
    if transform_config.get("resize", False):
        transforms.append(
            Resized(
                keys=["image"],
                spatial_size=tuple(transform_config["spatial_size"]),
                mode=transform_config.get("resize_mode", "trilinear"),
            )
        )

    # --- spatial augmentation -------------------------------------------------
    if enabled("rand_flip"):
        # Left-right on the T88 sagittal axis. Anatomically valid for a roughly
        # symmetric brain, and the cheapest way to double 159 training volumes.
        transforms.append(
            RandFlipd(
                keys=["image"],
                prob=setting("rand_flip_prob", 0.5),
                spatial_axis=setting("rand_flip_spatial_axis", 0),
            )
        )
    if enabled("rand_affine"):
        # Small rigid-plus-scale jitter, the realistic replacement for rand_rotate90.
        # rotate_range is radians per axis; 0.175 rad is 10 degrees. padding_mode
        # "zeros" pairs with NormalizeIntensityd(nonzero=True), which ignores the
        # padding rather than letting it drag the mean down.
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=setting("rand_affine_prob", 0.5),
                rotate_range=setting("rand_affine_rotate_range", [0.175, 0.175, 0.175]),
                scale_range=setting("rand_affine_scale_range", [0.1, 0.1, 0.1]),
                translate_range=setting("rand_affine_translate_range", [5, 5, 5]),
                mode=setting("rand_affine_mode", "bilinear"),
                padding_mode=setting("rand_affine_padding_mode", "zeros"),
            )
        )
    if enabled("rand_rotate90"):
        # 90-degree rotations are anatomically implausible for registered T88 brains.
        # Kept because it is what the round-1 sweeps measured, not because it is right.
        transforms.append(
            RandRotate90d(
                keys=["image"],
                prob=setting("rand_rotate90_prob", 0.5),
                spatial_axes=tuple(setting("rand_rotate90_spatial_axes", [0, 2])),
            )
        )

    # --- intensity augmentation, before normalisation -------------------------
    if enabled("rand_bias_field"):
        # Smooth multiplicative field: the MRI-specific nuisance variable, and the
        # augmentation with the most defensible prior for this modality.
        transforms.append(
            RandBiasFieldd(
                keys=["image"],
                prob=setting("rand_bias_field_prob", 0.5),
                degree=setting("rand_bias_field_degree", 3),
                coeff_range=tuple(setting("rand_bias_field_coeff_range", [0.0, 0.1])),
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

    # --- intensity augmentation, after normalisation --------------------------
    if enabled("rand_gaussian_noise"):
        transforms.append(
            RandGaussianNoised(
                keys=["image"],
                prob=setting("rand_gaussian_noise_prob", 0.5),
                std=setting("rand_gaussian_noise_std", 0.1),
            )
        )
    if enabled("rand_scale_intensity"):
        transforms.append(
            RandScaleIntensityd(
                keys=["image"],
                prob=setting("rand_scale_intensity_prob", 0.5),
                factors=setting("rand_scale_intensity_factors", 0.1),
            )
        )
    if enabled("rand_shift_intensity"):
        transforms.append(
            RandShiftIntensityd(
                keys=["image"],
                prob=setting("rand_shift_intensity_prob", 0.5),
                offsets=setting("rand_shift_intensity_offsets", 0.1),
            )
        )

    transforms.extend(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ]
    )
    return Compose(transforms)


class Dataset(MonaiDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return item["image"], item["label"]


class DatasetSource:
    """Items and per-mode Datasets, built once and then sliced by index.

    A Dataset is constructed per mode rather than per loader, so the folds of a
    cross-validation run all read through the same objects. That is also the seam for
    a future caching option: swapping Dataset for a CacheDataset in _build_dataset
    would populate the cache once and let every fold reuse it.

    Train and eval cannot share one Dataset because augmentation is train-only, so
    there is one per mode rather than one overall.
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend = build_backend(config)
        self.items = self.backend.build_items()
        self._datasets: dict[str, Dataset] = {}

    def _build_dataset(self, mode: str) -> Dataset:
        return Dataset(
            data=self.items,
            transform=build_transforms(self.backend, self.config, mode),
        )

    def dataset(self, mode: str) -> Dataset:
        # "val" and "test" differ only in name; both skip augmentation.
        if mode not in self._datasets:
            self._datasets[mode] = self._build_dataset(mode)
        return self._datasets[mode]

    def item_id(self, index: int) -> str:
        return self.backend.item_id(self.items[index])

    def loader(self, indices: list[int], mode: str, shuffle: bool = False):
        dataloader_config = self.config["dataloader"]
        return DataLoader(
            Subset(self.dataset(mode), indices),
            batch_size=dataloader_config["batch_size"],
            shuffle=shuffle,
            num_workers=dataloader_config["num_workers"],
        )

    def train_val_loaders(self, split_indices: dict[str, list[int]]):
        """Train and validation loaders; the second is None for a refit split.

        Returning None rather than an empty loader keeps the refit path identical to the
        single-split path apart from which build_*_split_indices produced the indices,
        and makes "there is no validation split" something callers must handle rather
        than something they discover as a zero-length iteration.
        """
        val_idx = split_indices["val_idx"]
        return (
            self.loader(split_indices["train_idx"], mode="train", shuffle=True),
            self.loader(val_idx, mode="val") if val_idx else None,
        )

    def fold_loaders(self, fold: dict):
        """Train/val loaders for one CV fold, over the same Datasets as every fold."""
        return (
            self.loader(fold["train_idx"], mode="train", shuffle=True),
            self.loader(fold["val_idx"], mode="val"),
        )

    def test_loader(self, test_idx: list[int]):
        return self.loader(test_idx, mode="test")


def build_dataset_source(config: dict) -> DatasetSource:
    return DatasetSource(config)
