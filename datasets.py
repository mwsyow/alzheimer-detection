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
    Orientationd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    Spacingd,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset


def oasis_scan_id(path: Path) -> str:
    return "_".join(path.name.split("_")[:3])


def oasis_subject_id(scan_id: str) -> str:
    """Return the person identifier, excluding the MR visit suffix."""
    return scan_id.rsplit("_", 1)[0]


def get_data(img_paths: list[Path], label_path: Path):
    df = pd.read_excel(label_path)
    required = {"ID", "CDR", "Age"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OASIS metadata is missing columns: {sorted(missing)}")
    rows = df.set_index("ID", verify_integrity=True)
    dataset_items = []
    for path in img_paths:
        scan_id = oasis_scan_id(path)
        if scan_id not in rows.index:
            raise ValueError(f"No OASIS metadata row for image {path} ({scan_id})")
        row = rows.loc[scan_id]
        dataset_items.append(
            {
                "label": int(bool(row["CDR"])),
                "age": float(row["Age"]),
                "subject_id": oasis_subject_id(scan_id),
                "scan_id": scan_id,
                "image": str(path),
                "image_id": str(path),
            }
        )
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


def build_rotating_cv_split_indices(dataset_items: list[dict], config: dict) -> dict:
    """Build subject-level K-fold rotations with K-2/train, 1/val and 1/test."""
    cv_config = config["cv"]
    n_splits = cv_config["n_splits"]
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 3:
        raise ValueError(f"cv.n_splits must be an integer >= 3, got {n_splits!r}")

    subjects: dict[str, dict] = {}
    for index, item in enumerate(dataset_items):
        subject_id = str(item.get("subject_id", item.get("image_id", index)))
        label = int(item["label"])
        record = subjects.setdefault(subject_id, {"label": label, "indices": []})
        if record["label"] != label:
            raise ValueError(f"Subject {subject_id!r} has conflicting AD labels")
        record["indices"].append(index)

    ordered_subjects = sorted(subjects)
    labels = [subjects[subject_id]["label"] for subject_id in ordered_subjects]
    class_counts = pd.Series(labels).value_counts().to_dict()
    if class_counts and min(class_counts.values()) < n_splits:
        raise ValueError(
            f"Each class needs at least cv.n_splits subjects; counts={class_counts}, "
            f"n_splits={n_splits}"
        )

    manifest_path = cv_config.get("manifest_path")
    if manifest_path:
        manifest = pd.read_csv(manifest_path)
        required = {"subject_id", "label", "fold"}
        if required - set(manifest.columns):
            raise ValueError(
                f"Fold manifest is missing columns: {sorted(required - set(manifest.columns))}"
            )
        if manifest["subject_id"].duplicated().any():
            raise ValueError("Fold manifest contains duplicate subject IDs")
        if "n_splits" in manifest and set(manifest["n_splits"]) != {n_splits}:
            raise ValueError("Fold manifest n_splits does not match cv.n_splits")
        if "cv_random_seed" in manifest:
            manifest_seeds = set(manifest["cv_random_seed"].dropna())
            expected_seed = cv_config.get("random_seed")
            if manifest_seeds and manifest_seeds != {expected_seed}:
                raise ValueError(
                    "Fold manifest cv_random_seed does not match cv.random_seed"
                )
        manifest_labels = dict(zip(manifest["subject_id"].astype(str), manifest["label"]))
        if set(manifest_labels) != set(ordered_subjects):
            raise ValueError("Fold manifest subjects do not exactly match the dataset")
        for subject_id in ordered_subjects:
            if int(manifest_labels[subject_id]) != subjects[subject_id]["label"]:
                raise ValueError(f"Fold manifest label mismatch for {subject_id}")
        subject_folds = {
            str(subject_id): int(fold)
            for subject_id, fold in zip(manifest["subject_id"], manifest["fold"])
        }
        if set(subject_folds.values()) != set(range(1, n_splits + 1)):
            raise ValueError("Fold manifest does not contain exactly cv.n_splits folds")
    else:
        shuffle = cv_config.get("shuffle", True)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=cv_config.get("random_seed") if shuffle else None,
        )
        subject_folds = {}
        for fold_index, (_, test_positions) in enumerate(
            splitter.split(ordered_subjects, labels), start=1
        ):
            for position in test_positions:
                subject_folds[ordered_subjects[position]] = fold_index

    rotations = []
    all_folds = set(range(1, n_splits + 1))
    for test_fold in range(1, n_splits + 1):
        val_fold = test_fold % n_splits + 1

        def indices_for(fold_numbers):
            return sorted(
                index
                for subject_id, record in subjects.items()
                if subject_folds[subject_id] in fold_numbers
                for index in record["indices"]
            )

        train_idx = indices_for(all_folds - {test_fold, val_fold})
        val_idx = indices_for({val_fold})
        test_idx = indices_for({test_fold})
        rotations.append(
            {
                "fold": test_fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "train_subject_ids": sorted(
                    {str(dataset_items[index].get("subject_id", index)) for index in train_idx}
                ),
                "val_subject_ids": sorted(
                    {str(dataset_items[index].get("subject_id", index)) for index in val_idx}
                ),
                "test_subject_ids": sorted(
                    {str(dataset_items[index].get("subject_id", index)) for index in test_idx}
                ),
            }
        )

    return {
        "strategy": "rotating_test",
        "n_splits": n_splits,
        "subject_folds": subject_folds,
        "index_to_subject_id": {
            index: str(item.get("subject_id", item.get("image_id", index)))
            for index, item in enumerate(dataset_items)
        },
        "folds": rotations,
    }


# Every key build_transforms reads. A "transforms" block naming anything else is a
# typo, and unlike model.params -- which reaches a constructor and raises TypeError --
# an unrecognised transform key would otherwise be a silent no-op: the sweep runs to
# completion and every trial is identical, with nothing in the logs saying why.
KNOWN_TRANSFORM_KEYS = frozenset(
    {
        "spacing",
        "pixdim",
        "spacing_mode",
        "orientation",
        "axcodes",
        "resize",
        "spatial_size",
        "resize_mode",
        "scale_intensity",
        "scale_channel_wise",
        "intensity_order",
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

INTENSITY_ORDERS = frozenset({"scale_then_normalize", "normalize_then_scale"})


def validate_transform_config(transform_config: dict) -> None:
    unknown = sorted(set(transform_config) - KNOWN_TRANSFORM_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown transforms key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(KNOWN_TRANSFORM_KEYS))}"
        )

    intensity_order = transform_config.get("intensity_order", "scale_then_normalize")
    if intensity_order not in INTENSITY_ORDERS:
        raise ValueError(
            "transforms.intensity_order must be one of "
            f"{', '.join(sorted(INTENSITY_ORDERS))}, got {intensity_order!r}"
        )
    if transform_config.get("spacing", False) and "pixdim" not in transform_config:
        raise ValueError("transforms.spacing=true requires transforms.pixdim")
    if transform_config.get("orientation", False) and "axcodes" not in transform_config:
        raise ValueError("transforms.orientation=true requires transforms.axcodes")


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
    if transform_config.get("spacing", False):
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=tuple(transform_config["pixdim"]),
                mode=transform_config.get("spacing_mode", "bilinear"),
            )
        )
    if transform_config.get("orientation", False):
        transforms.append(
            Orientationd(
                keys=["image"],
                axcodes=transform_config["axcodes"],
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            )
        )
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

    def append_scale_intensity():
        if transform_config.get("scale_intensity", False):
            transforms.append(
                ScaleIntensityd(
                    keys=["image"],
                    channel_wise=transform_config.get("scale_channel_wise", False),
                )
            )

    def append_normalize_intensity():
        if transform_config.get("normalize_intensity", False):
            transforms.append(
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=transform_config.get("normalize_nonzero", True),
                    channel_wise=transform_config.get("normalize_channel_wise", True),
                )
            )

    intensity_order = transform_config.get("intensity_order", "scale_then_normalize")
    if intensity_order == "scale_then_normalize":
        append_scale_intensity()
        append_normalize_intensity()
    else:
        append_normalize_intensity()
        append_scale_intensity()

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
    def __init__(self, *args, target_key: str = "label", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_key = target_key

    def __getitem__(self, index):
        item = super().__getitem__(index)
        target = item[self.target_key]
        if self.target_key == "age":
            target = torch.as_tensor(target, dtype=torch.float32)
        return item["image"], target


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
        task_name = config.get("task", {}).get("name", "ad_classification")
        if task_name not in {"ad_classification", "age_regression"}:
            raise ValueError(
                f"Unsupported task {task_name!r}; expected ad_classification or "
                "age_regression"
            )
        self.target_key = "age" if task_name == "age_regression" else "label"
        if self.target_key == "age" and any("age" not in item for item in self.items):
            raise ValueError("age_regression requires an age value for every sample")
        self._datasets: dict[str, Dataset] = {}

    def _build_dataset(self, mode: str) -> Dataset:
        return Dataset(
            data=self.items,
            transform=build_transforms(self.backend, self.config, mode),
            target_key=self.target_key,
        )

    def dataset(self, mode: str) -> Dataset:
        # "val" and "test" differ only in name; both skip augmentation.
        if mode not in self._datasets:
            self._datasets[mode] = self._build_dataset(mode)
        return self._datasets[mode]

    def item_id(self, index: int) -> str:
        return self.backend.item_id(self.items[index])

    def subject_id(self, index: int) -> str:
        item = self.items[index]
        return str(item.get("subject_id", self.backend.item_id(item)))

    def set_random_state(self, seed: int) -> None:
        """Reset already-built MONAI Randomizable transforms for fold reproducibility."""
        for dataset in self._datasets.values():
            transform = getattr(dataset, "transform", None)
            if hasattr(transform, "set_random_state"):
                transform.set_random_state(seed=seed)

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
