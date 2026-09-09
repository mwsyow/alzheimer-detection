import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.io import savemat

from scripts.preprocess_unified import (
    GateError,
    OASIS_ACQUISITION_OVERRIDES,
    Paths,
    analyze_sagittal_affine,
    clinica_command,
    approve_pilot,
    manifest_digest,
    normalize_image,
    require_approval,
    select_pilot,
    stage_bids,
    stage_bids,
    validate_inventory,
    validate_mask,
    validate_native,
    validate_registered,
    validate_transform,
)


def save_image(path: Path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), affine if affine is not None else np.eye(4)), path)


def test_validate_native_accepts_plausible_3d_volume(tmp_path):
    path = tmp_path / "native.nii.gz"
    data = np.zeros((128, 128, 128), dtype=np.int16)
    data[20:100, 20:100, 20:100] = 10
    save_image(path, data, np.diag([1.5, 1.5, 1.5, 1]))
    result = validate_native(path)
    assert result["shape"] == [128, 128, 128]


def test_analyze_sagittal_affine_encodes_asl_voxel_axes():
    affine = analyze_sagittal_affine((256, 256, 128), (1.0, 1.0, 1.25))
    assert nib.aff2axcodes(affine) == ("A", "S", "L")
    assert np.allclose(np.abs(affine[:3, :3]).sum(axis=0), (1.0, 1.0, 1.25))
    assert np.allclose(affine @ np.array([127.5, 127.5, 63.5, 1]), (0, 0, 0, 1))


def test_clinica_container_runs_as_user_in_writable_workspace():
    command = clinica_command("/host/pilot:/workspace", 2)
    assert command[command.index("--user") + 1] != "0:0"
    assert command[command.index("-w") + 1] == "/workspace"
    assert command[command.index("--working_directory") + 1] == "/workspace/work"


def test_staged_bids_description_is_recognizable_by_clinica(tmp_path):
    source = tmp_path / "native.nii.gz"
    data = np.zeros((100, 100, 100), dtype=np.int16)
    data[10:90, 10:90, 10:90] = 1
    save_image(source, data, np.diag([1.5, 1.5, 1.5, 1]))
    row = {
        "source_image": str(source),
        "bids_subject": "sub-test",
        "bids_session": "ses-baseline",
        "dataset": "test",
        "subject_id": "test",
        "group": "control",
        "age": "70",
        "sex": "f",
    }
    stage_bids([row], tmp_path / "workspace")
    description = json.loads(
        (tmp_path / "workspace/bids/dataset_description.json").read_text()
    )
    assert description["DatasetType"] == "raw"


def test_bids_description_identifies_raw_dataset_for_clinica(tmp_path):
    source = tmp_path / "native.nii.gz"
    data = np.zeros((100, 100, 100), dtype=np.int16)
    data[10:90, 10:90, 10:90] = 1
    save_image(source, data, np.diag([1.5, 1.5, 1.5, 1]))
    stage_bids(
        [
            {
                "source_image": str(source),
                "bids_subject": "sub-test",
                "bids_session": "ses-baseline",
                "dataset": "test",
                "subject_id": "test",
                "group": "control",
                "age": "70",
                "sex": "f",
            }
        ],
        tmp_path / "workspace",
    )
    description = json.loads(
        (tmp_path / "workspace/bids/dataset_description.json").read_text()
    )
    assert description["DatasetType"] == "raw"


@pytest.mark.parametrize("data", [np.zeros((128, 128, 128)), np.full((128, 128, 128), np.nan)])
def test_validate_native_rejects_empty_or_nonfinite(tmp_path, data):
    path = tmp_path / "bad.nii.gz"
    save_image(path, data, np.diag([1.5, 1.5, 1.5, 1]))
    with pytest.raises(GateError):
        validate_native(path)


def test_normalization_and_mask_gates(tmp_path):
    rng = np.random.default_rng(4)
    mask = np.zeros((100, 100, 100), dtype=np.uint8)
    mask[5:95, 5:95, 5:95] = 1
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask > 0] = rng.normal(100, 20, size=np.count_nonzero(mask))
    stripped = tmp_path / "stripped.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    normalized = tmp_path / "normalized.nii.gz"
    save_image(stripped, data)
    save_image(mask_path, mask)
    result = validate_mask(stripped, mask_path)
    assert result["dominant_component"] == 1
    normalize_image(stripped, mask_path, normalized)
    output = np.asarray(nib.load(normalized).dataobj)
    assert np.all(output[mask == 0] == 0)
    assert output[mask > 0].mean() == pytest.approx(0, abs=1e-4)
    assert output[mask > 0].std() == pytest.approx(1, abs=1e-3)


def test_mask_gate_rejects_extracranial_output(tmp_path):
    mask = np.zeros((100, 100, 100), dtype=np.uint8)
    mask[5:95, 5:95, 5:95] = 1
    data = mask.astype(np.float32)
    data[0, 0, 0] = 1
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    save_image(image_path, data)
    save_image(mask_path, mask)
    with pytest.raises(GateError, match="outside mask"):
        validate_mask(image_path, mask_path)


def test_registered_gate_checks_exact_mni_grid(tmp_path):
    path = tmp_path / "registered.nii.gz"
    data = np.zeros((169, 208, 179), dtype=np.float32)
    data[40:120, 50:150, 30:140] = 1
    save_image(path, data)
    assert validate_registered(path)["orientation"] == ["R", "A", "S"]
    bad = tmp_path / "bad-grid.nii.gz"
    save_image(bad, data[:-1])
    with pytest.raises(GateError, match="unexpected Clinica grid"):
        validate_registered(bad)


def test_transform_gate_rejects_missing_or_malformed_transform(tmp_path):
    with pytest.raises(GateError, match="missing or empty"):
        validate_transform(tmp_path / "missing.mat")
    malformed = tmp_path / "bad.mat"
    malformed.write_text("not a transform")
    with pytest.raises(GateError, match="invalid"):
        validate_transform(malformed)
    valid = tmp_path / "good.mat"
    parameters = np.concatenate([np.eye(3).reshape(-1), [1.0, 2.0, 3.0]])
    valid.write_text("Parameters: " + " ".join(str(value) for value in parameters))
    validate_transform(valid)


def test_transform_gate_accepts_ants_matlab_v4_affine(tmp_path):
    valid = tmp_path / "ants-affine.mat"
    parameters = np.concatenate([np.eye(3).reshape(-1), [2.0, -3.0, 4.0]])
    savemat(
        valid,
        {"AffineTransform_double_3_3": parameters[:, None], "fixed": np.zeros((3, 1))},
        format="4",
    )
    validate_transform(valid)


def test_transform_gate_rejects_implausible_binary_affine(tmp_path):
    invalid = tmp_path / "singular-affine.mat"
    parameters = np.zeros((12, 1))
    savemat(invalid, {"AffineTransform_double_3_3": parameters}, format="4")
    with pytest.raises(GateError, match="implausible"):
        validate_transform(invalid)


def sample_rows():
    rows = []
    for dataset in ("oasis1", "miriad"):
        for group in ("control", "ad"):
            for index in range(8):
                rows.append({"dataset": dataset, "group": group, "subject_id": f"{dataset}-{group}-{index}"})
    return rows


def test_pilot_is_balanced_and_deterministic():
    first = select_pilot(sample_rows())
    second = select_pilot(sample_rows())
    assert first == second
    counts = {}
    for row in first:
        counts[(row["dataset"], row["group"])] = counts.get((row["dataset"], row["group"]), 0) + 1
    assert set(counts.values()) == {5}


def test_oasis_visual_qc_fallback_is_pinned():
    assert OASIS_ACQUISITION_OVERRIDES == {"OAS1_0373_MR1": 2}


def test_inventory_gate_rejects_duplicates():
    row = {"dataset": "oasis1", "subject_id": "one", "bids_subject": "sub-one", "group": "control"}
    with pytest.raises(GateError, match="duplicate"):
        validate_inventory([row, row])


def test_approval_is_bound_to_provenance(tmp_path):
    paths = Paths(repo=tmp_path, output=tmp_path / "output")
    provenance = paths.output / "pilot/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"manifest_sha256": "m", "qc_sha256": "q"}))
    approve_pilot(paths, "tester")
    require_approval(paths)
    provenance.write_text(json.dumps({"manifest_sha256": "changed", "qc_sha256": "q"}))
    with pytest.raises(GateError, match="stale"):
        require_approval(paths)


def test_manifest_digest_is_order_sensitive_and_stable():
    rows = [{"subject": "a"}, {"subject": "b"}]
    assert manifest_digest(rows) == manifest_digest(rows)
    assert manifest_digest(rows) != manifest_digest(list(reversed(rows)))
