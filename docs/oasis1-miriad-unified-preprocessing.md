# OASIS-1 and MIRIAD: feasibility of unified T1 preprocessing

Date of assessment: 2026-09-07

## Decision

Native OASIS-1 and MIRIAD scans can be combined in one imaging dataset, provided
that both cohorts are processed from a single native T1 acquisition with exactly
the same offline pipeline. They should **not** be combined by mixing the existing
OASIS `*_111_t88_masked_gfc.img` derivatives with native or newly resampled
MIRIAD images. That would make preprocessing method and atlas space almost
perfect proxies for dataset membership.

The proposed primary implementation is:

1. organize one native T1 acquisition per subject as BIDS;
2. run [Clinica `t1-linear`](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Pipelines/T1_Linear/), using its standard ANTs implementation;
3. run [FreeSurfer SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) on each registered image and retain both the stripped image and mask;
4. apply robust, brain-mask-only intensity normalization in this repository;
5. apply the same fixed crop/grid and model-specific resize to both datasets;
6. perform automated and visual quality control before training.

This is a new harmonized experiment. The established OASIS T88 results should be
retained as a separate reference baseline.

## What was inspected

### OASIS-1 native input

Representative file:

`data/oasis_cross-sectional_disc2/disc2/OAS1_0076_MR1/RAW/OAS1_0076_MR1_mpr-1_anon.img`

This is the first native MP-RAGE acquisition, not the processed T88 image. The
paired Analyze `.hdr` supplies its geometry. OASIS subjects generally have three
or four same-session native MP-RAGE acquisitions (`mpr-1` through `mpr-3` or
`mpr-4`). The current project instead trains on the already combined and processed
`*_mpr_n*_anon_111_t88_masked_gfc.img` derivative.

### MIRIAD native input

Downloaded representative file:

`data/MIRIAD_sample/miriad_199_HC_F_01_MR_1.nii`

This is subject 199, healthy control, female, visit 1, acquisition 1. The XNAT
session contains one scan of type `T1` and one 16 MB NIfTI resource. It exposes no
masked, bias-corrected, or atlas-registered derivative for this acquisition.

The authenticated project inventory contained 708 `xnat:mrSessionData` records
and 268 `ext:clinicalAssessmentData` records. MIRIAD session names distinguish
visit and repeat acquisition: for example, `miriad_199_1_MR_1` and
`miriad_199_1_MR_2` are two acquisitions at the same visit, not independent
subjects.

### Measured comparison

The following values were read with NiBabel from the image headers and voxel
arrays; they were not inferred from filenames.

| Property | Native OASIS-1 sample | Native MIRIAD sample |
|---|---:|---:|
| Container | SPM/Analyze `.hdr` + `.img` | NIfTI-1 `.nii` |
| Shape | 256 × 256 × 128 | 256 × 256 × 124 |
| Voxel size | 1.0 × 1.0 × 1.25 mm | 0.9375 × 0.9375 × 1.5 mm |
| Header orientation | LAS | LSP |
| Stored datatype | signed 16-bit integer | signed 16-bit integer |
| Observed range | 0–4095 | 0–258 |
| Nonzero fraction | 43.29% | 74.73% |
| Nonzero median | 751 | 8 |
| Nonzero 99th percentile | 2802 | 141 |

For reference, a current OASIS T88 training derivative measured 176 × 208 × 176
at 1 mm isotropic, orientation LAS, and only 25.9% nonzero voxels. The large
difference in nonzero support confirms that the MIRIAD NIfTI is not equivalent to
the skull-masked OASIS derivative.

The absolute intensity ranges are not directly meaningful across MRI datasets,
but the large scale and background differences are exactly the kind of shortcuts
a CNN can exploit. A shared resize alone cannot correct them.

## Why combination is technically reasonable

Both sources contain high-resolution three-dimensional T1-weighted structural
MRI of older adults. Their native matrices and through-plane resolutions are
close enough for the same registration and resampling workflow. The differences
seen here—container, orientation, voxel size, affine, intensity scale, head
position, and background—are standard targets of structural MRI preprocessing.

Combination is therefore feasible at the **post-preprocessing image level**, not
at the native voxel level. Feasibility does not imply that scanner/site effects
will disappear. MIRIAD used a single 1.5 T GE Signa scanner and IR-FSPGR sequence,
whereas OASIS-1 used MP-RAGE acquisitions. Dataset/site must remain in the
manifest and performance must be reported separately for OASIS and MIRIAD.

There is also a label-domain issue independent of preprocessing: MIRIAD contains
clinically diagnosed mild-to-moderate probable AD and controls, while OASIS CDR
0.5 is not automatically equivalent to MIRIAD AD. The primary definite-label
analysis should use OASIS CDR 0 as control and CDR at least 1 as dementia, with
CDR 0.5 analysed separately.

## Selected established implementation

### Core: Clinica `t1-linear`

Clinica's current `t1-linear` pipeline is selected as the core because it is a
documented, reproducible pipeline created specifically as minimal preprocessing
for deep-learning classification of Alzheimer's disease from T1 MRI. It uses
ANTs and produces a standard 1 mm MNI image and transform for every input.

The official pipeline performs N4 bias-field correction, affine registration to
the symmetric `MNI152NLin2009cSym` template, and a deterministic crop to
169 × 208 × 179. It is run as:

```bash
clinica run t1-linear BIDS_DIRECTORY CAPS_DIRECTORY --n_procs N
```

Use the installed ANTs binaries for the primary reproducible run. Clinica also
offers `--use-antspy`, but its documentation describes that route as newer and
less extensively tested.

ClinicaDL calls `t1-linear` its minimal pipeline and reports that minimal and more
extensive nonlinear/skull-stripped preprocessing produced comparable AD
classification accuracy; its authors recommend the minimal pipeline for
simplicity. See the [ClinicaDL preprocessing overview](https://clinicadl.readthedocs.io/en/v0.0.3/Run/Introduction/)
and [OASIS preprocessing tutorial](https://aramislab.paris.inria.fr/clinicadl/tuto/2023/html/notebooks/preprocessing.html).

### Brain extraction: SynthStrip

Clinica `t1-linear` crops empty background but does not promise removal of all
extracranial tissue. This project currently learns from skull-masked OASIS
derivatives, and the native samples have very different background support.
SynthStrip is therefore added as one explicit, identical post-registration step.

SynthStrip is an established FreeSurfer brain-extraction method designed to be
robust across acquisition protocols, resolutions, orientations, scanners, and
pathology. Its method is described in
[Hoopes et al., 2022](https://doi.org/10.1016/j.neuroimage.2022.119474).

Run it on every Clinica cropped T1 output, without dataset-specific flags:

```bash
mri_synthstrip \
  -i INPUT_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz \
  -o OUTPUT_desc-brain_T1w.nii.gz \
  -m OUTPUT_desc-brain_mask.nii.gz
```

The mask must be retained for QC and normalization. We should not use `--no-csf`
for the primary run because excluding surrounding CSF risks removing meaningful
atrophy-related space and changes the intended whole-brain boundary.

## Proposed preprocessing, step by step

### 1. Select exactly one native acquisition per subject

- OASIS-1 primary input: `mpr-1`, provided it passes QC.
- MIRIAD primary input: visit 1, `MR_1`, provided it passes QC.

Why: using one scan per subject gives both sources the same sampling unit and
avoids treating repeated acquisitions as independent observations. Averaging
three or four OASIS acquisitions while using one MIRIAD acquisition would
introduce a dataset-specific signal-to-noise advantage. OASIS multi-acquisition
averaging and MIRIAD `MR_2` repeat scans can be studied later as a separate
robustness experiment.

If the first acquisition fails QC, use the next valid same-visit acquisition and
record that substitution in the manifest; do not choose scans based on diagnosis
or model performance.

The balanced pilot identified one such case: `OAS1_0373_MR1` `mpr-1` registered
correctly but SynthStrip removed inferior midline brain tissue. Its same-session
`mpr-2` acquisition passed the identical automated and visual checks and is the
recorded input for that subject (`selection_exception` is
`mpr-1-failed-synthstrip-qc`).

### 2. Convert both sources into a common BIDS layout

Convert the OASIS Analyze pair to NIfTI without resampling, and copy the MIRIAD
NIfTI without altering voxel values. Preserve source subject/session/acquisition
IDs in a manifest.

OASIS' legacy Analyze header uses orientation code 2 (sagittal unflipped), with
stored voxel axes anterior, superior, and left. NiBabel's generic Analyze affine
does not interpret this byte, so the implementation explicitly transfers that
axis convention into the NIfTI qform/sform. This changes only physical-space
metadata: the voxel array is checksum-equivalent and is not transposed or
interpolated.

Why: Clinica expects BIDS input, and one naming/metadata convention prevents
dataset-specific file handling from leaking into later steps. Format conversion
must not silently reorient or interpolate the data.

### 3. Validate geometry and canonicalize metadata

Check finite voxels, nonempty images, dimensions, voxel sizes, qform/sform or
Analyze affine, handedness, and plausible field of view before processing.

Why: the inspected inputs are LAS and LSP. ANTs registration can handle valid
physical-space orientation, but a bad affine can produce a plausible-looking,
incorrectly flipped result. Orientation metadata validation is not atlas
registration and must not be presented as such.

### 4. Correct intensity inhomogeneity with N4

Use the N4 step already implemented by Clinica `t1-linear`, with the same version
and parameters for both datasets.

Why: slowly varying receive-field bias changes the intensity of the same tissue
across the head. The native acquisitions come from different protocols and show
very different intensity distributions. N4 corrects spatial inhomogeneity; it
does not make scanner intensity units equivalent, so later normalization is still
required. N4 is the established ANTs method described by
[Tustison et al., 2010](https://doi.org/10.1109/TMI.2010.2046908).

### 5. Affinely register to one common template

Use Clinica's affine registration to `MNI152NLin2009cSym` and its 1 mm output.

Why: affine registration removes differences in head pose, orientation, global
scale, and shear while preserving more individual anatomy than a nonlinear warp.
Voxel locations then have approximately corresponding anatomy across both
datasets. A symmetric template also avoids building a left/right template bias
into the common space.

Do not register MIRIAD to the old OASIS T88 derivative and do not mix T88 and MNI
outputs. The common atlas must be identical for every subject.

### 6. Resample once to 1 mm isotropic and apply a fixed crop

Use Clinica's registered, cropped output (169 × 208 × 179). Avoid additional
offline interpolation before this step.

Why: the source through-plane spacings differ (1.25 versus 1.5 mm), so a common
grid is necessary for voxel-based CNN input. Each interpolation slightly blurs
the image; estimating registration in physical space and resampling once limits
that damage. A deterministic template-space crop removes irrelevant empty volume
without using diagnosis-dependent content.

### 7. Apply SynthStrip identically

Create a brain-only T1 and binary brain mask from every registered/cropped image.
Use identical model version and command-line options for OASIS and MIRIAD.

Why: skull, scalp, neck, and differing background support are strong site cues
with no direct role in the desired diagnosis. Explicit masking also gives a
consistent region over which to compute intensity statistics. Every mask must be
visually checked because brain-extraction failures can imitate focal atrophy.

### 8. Robustly normalize intensity inside the brain mask

For each image independently:

1. take voxels inside the SynthStrip mask;
2. clip them to robust limits, provisionally the 0.5th and 99.5th percentiles;
3. compute mean and standard deviation on those clipped brain voxels;
4. z-score brain voxels and keep voxels outside the mask at zero.

Why: MRI intensity has no universal absolute unit, and the inspected native
ranges differ by more than an order of magnitude. Brain-mask-only statistics
avoid the observed dataset difference in background support. Percentile clipping
limits domination by isolated extremes. Parameters must be fixed before model
evaluation and applied per image, so no validation/test distribution statistics
enter training.

The provisional percentiles should be confirmed on an image-only QC sample that
is balanced by dataset and does not use labels. They must not be tuned against
classification AUC.

### 9. Separate offline harmonization from model-specific resizing

Store the harmonized 1 mm, 169 × 208 × 179 brain image as the canonical
derivative. At training time, resize that same derivative to the architecture's
required tensor shape, using trilinear interpolation for intensities.

Why: preserving one canonical derivative makes comparisons reproducible. A
96 × 128 × 96 input and BRAT's 32 × 256 × 256 input are model recipes, not
alternative definitions of the underlying harmonized dataset. Binary masks, if
resized, require nearest-neighbour interpolation.

### 10. Perform QC and test for residual site signal

For every processed image, record:

- source dataset, subject, session, and acquisition;
- original and final geometry/orientation;
- registration similarity and transform determinant;
- brain-mask volume and bounding box;
- normalized intensity summary;
- pass/fail and exclusion reason;
- software/container versions and command line.

Visually inspect overlays on the MNI template and masks in all three planes. Flag
left/right flips, missing cerebellum or cortex, retained skull, severe motion,
registration failure, and truncated field of view.

Finally, train a simple dataset-origin classifier (OASIS versus MIRIAD) on the
harmonized images using subject-disjoint splits. Above-chance prediction is
expected because scanner effects cannot be completely removed, but very high
accuracy is a warning that the AD classifier may exploit site. Always report AD
metrics separately by source dataset as well as pooled.

## Why not use the alternatives as the primary pipeline?

### Existing OASIS T88 derivatives plus newly processed MIRIAD

Rejected because T88 registration, multi-acquisition combination, gain-field
correction, and masking would occur only on the OASIS side. A model could identify
the preprocessing source rather than disease.

### MONAI `Orientationd`, `Spacingd`, and `Resize`

These remain useful training transforms but are insufficient as harmonization.
Orientation changes axis order, spacing changes sampling resolution, and resize
changes the array shape. None estimates anatomical alignment, corrects a bias
field, or removes extracranial tissue.

### Nonlinear registration / ClinicaDL `t1-extensive`

The established extensive pipeline performs bias correction, nonlinear
registration, and skull stripping. It is a defensible sensitivity analysis, but
not the primary choice here. Nonlinear deformation can partially normalize the
atrophy morphology that the classifier is supposed to detect; it is also more
computationally complex. ClinicaDL reports comparable AD classification results
for its minimal and extensive pipelines and recommends the simpler linear route.

### ComBat as an image preprocessing step

ComBat is not selected for raw voxel images. It is primarily used on derived
features and requires estimating site-specific distributions. Applying it before
splitting would leak information, while fitting it inside every training fold is
substantially more complex. First establish the common physical/intensity
pipeline and evaluate residual site effects.

## Experimental safeguards

- Split by subject, never by scan.
- For the primary pooled experiment use one image per subject.
- Stratify folds jointly by diagnosis and source dataset when counts permit.
- Keep an untouched cross-dataset evaluation: train OASIS, test MIRIAD, and vice
  versa.
- Report pooled and per-dataset AUC, average precision, sensitivity, specificity,
  balanced accuracy, and confidence intervals.
- Do not allow dataset-specific preprocessing parameters.
- Fit any learned normalization or harmonization component on training folds
  only. The proposed N4, registration, SynthStrip, and per-image normalization do
  not require cohort-level fitting.
- Keep OASIS CDR 0.5 separate in the first definite AD/control experiment.

## Limits of this assessment

The direct numeric comparison uses one native scan from each source; it proves
the formats are structurally compatible and demonstrates concrete preprocessing
differences, but it does not characterize the full distribution of either
cohort. Before bulk preprocessing, run the proposed workflow on a balanced pilot
of at least five controls and five AD subjects from each dataset, inspect all QC
outputs without reference to model performance, then freeze the pipeline.

MIRIAD's data-use agreement prohibits unauthorized redistribution. Native scans,
clinical exports, and derived subject-level images should remain ignored by Git
and be transferred only to authorized storage.

## Implemented workflow

The executable implementation is `scripts/preprocess_unified.py`. It pins
Clinica 0.11.3 with ANTs 2.6.5 and SynthStrip 1.8, writes all generated data
below the ignored `data/unified_preprocessing/` directory, and refuses to start
the full cohort until the balanced pilot has been explicitly approved.

Run the stages from the repository root:

```bash
uv sync
uv run python scripts/preprocess_unified.py inventory
uv run python scripts/preprocess_unified.py build-images
uv run python scripts/preprocess_unified.py pilot --n-procs 2
# Inspect every montage in data/unified_preprocessing/pilot/qc/montages/.
uv run python scripts/preprocess_unified.py approve-pilot --reviewer YOUR_NAME
uv run python scripts/preprocess_unified.py run --n-procs 2
uv run python scripts/preprocess_unified.py verify --cohort full
```

`inventory` verifies the expected 235 OASIS-1 and 69 MIRIAD subjects and hashes
the native inputs. The pilot is deterministic and balanced: five controls and
five definite AD cases from each source. Approval is tied to hashes of the pilot
manifest, QC table, and provenance; changing or rerunning those artifacts makes
the approval stale. Failed subject-level gates are recorded in `failures.csv`,
and processing can be rerun without replacing already valid outputs.

Each brain-extraction derivative has a sidecar containing the SHA-256 of its
registered input and the pipeline version. Reuse is content-based rather than
timestamp-based. The wrapper runs Clinica under the invoking user's UID/GID and
repairs ownership of products from older container runs before performing a
strict, subject-bounded invalidation of stale derivatives.

Correctness gates cover native geometry and finite intensities, lossless BIDS
staging, the exact shared MNI grid and orientation, plausible affine transforms,
binary and spatially plausible brain masks, zero-valued background, and
brain-mask-only normalized mean and variance. Provenance records commands,
parameters, source hashes, and resolved container image IDs. Unit tests for the
gates are in `tests/test_unified_preprocessing.py`.

Each N4 node uses approximately 1.5 GiB on the inspected local machine. Two
Clinica workers are therefore the safe default for a 16 GiB workstation. Higher
values are appropriate only when the available RAM is sufficient; ANTs and OMP
threads are fixed to one per worker to prevent nested CPU oversubscription.

## Sources

- [Malone et al. (2013), MIRIAD public release](https://pmc.ncbi.nlm.nih.gov/articles/PMC3809512/)
- [Cash et al. (2015), MIRIAD atrophy challenge](https://pmc.ncbi.nlm.nih.gov/articles/PMC4634338/)
- [Official UCL MIRIAD description and data-use terms](https://www.ucl.ac.uk/brain-sciences/ion/research/research-centres/dementia-research-centre/research-clinical-trials/minimal-interval-resonance-imaging-alzheimers-disease-miriad)
- [Clinica `t1-linear` documentation](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Pipelines/T1_Linear/)
- [ClinicaDL preprocessing tutorial](https://aramislab.paris.inria.fr/clinicadl/tuto/2023/html/notebooks/preprocessing.html)
- [ClinicaDL preprocessing comparison](https://clinicadl.readthedocs.io/en/v0.0.3/Run/Introduction/)
- [ANTs project](https://github.com/ANTsX/ANTs)
- [N4 bias correction paper](https://doi.org/10.1109/TMI.2010.2046908)
- [SynthStrip paper](https://doi.org/10.1016/j.neuroimage.2022.119474)
- [XNAT scan download API](https://wiki.xnat.org/xnat-api/how-to-download-files-via-the-xnat-rest-api)
