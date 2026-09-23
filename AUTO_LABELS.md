# Automatic esophagus/aorta pseudo-labels

`auto_split_class1.py` runs pretrained TotalSegmentator on each original CT,
then partitions the existing merged class-1 mask into esophagus (1) and aorta (4).
This creates **pseudo-labels**, not verified anatomical ground truth. No manual
seeds are required. Anatomical correctness still needs assessment; a successful
run only confirms technical checks. Validation Dice against these labels measures
agreement with the automatic labelling pipeline.

## Install and run

Run from the repository root. The optional environment keeps TotalSegmentator's
dependencies separate from the current training environment:

```bash
python -m venv .venv-pseudolabels
.venv-pseudolabels/bin/python -m pip install -r requirements-pseudolabels.txt

# Pilot on one patient. First inference downloads pretrained weights.
.venv-pseudolabels/bin/python auto_split_class1.py \
  --patients Patient_01 --device gpu \
  --output-dir data/segthor_pseudo_pilot

# Process every Patient_* folder using a NEW destination.
.venv-pseudolabels/bin/python auto_split_class1.py \
  --device gpu --output-dir data/segthor_pseudo_masks
```

Use `--device cpu` without a GPU (slower). The default source is
`data/segthor_part1/train`, containing `Patient_XX/Patient_XX.nii.gz` and
`Patient_XX/GT.nii.gz`. All patients, including those later used for validation,
are processed with the same fixed algorithm. No training or parameter fitting
is performed. Original files are never overwritten.

Outputs per patient:

- `GT_pseudo.nii.gz`: separate labels on the original grid.
- `pseudo_label_report.json`: file hashes, seed counts, overlap coverage,
  parameters, model version, and explicit pseudo-label provenance.
- `predictions/esophagus.nii.gz` and `predictions/aorta.nii.gz`: model outputs.

The root `summary.json` records successful and failed patients. Failure exits
with code 1; the dataset preparation step rejects missing patients. A failed
patient may leave predictions for inspection but no completed pseudo-mask.
Use a fresh output directory for retries; existing output directories are refused.

## Reuse predictions without running the model

Arrange separate **binary** model masks as
`PREDICTIONS/Patient_XX/esophagus.nii.gz` and `aorta.nii.gz`. Shapes and affines
must match the original CT/mask. This mode does not import TotalSegmentator or
download weights, and records the prediction file hashes (model version unknown).

```bash
ai4mi/bin/python auto_split_class1.py \
  --predictions-dir /path/to/PREDICTIONS \
  --output-dir data/segthor_pseudo_masks
```

## Build the training dataset

After a successful run for all patients:

```bash
ai4mi/bin/python prepare_separate_labels.py \
  --labels-dir data/segthor_pseudo_masks --mask-name GT_pseudo.nii.gz \
  --label-source pseudo --output data/SEGTHOR_PSEUDO --check-only

ai4mi/bin/python prepare_separate_labels.py \
  --labels-dir data/segthor_pseudo_masks --mask-name GT_pseudo.nii.gz \
  --label-source pseudo --output data/SEGTHOR_PSEUDO

ai4mi/bin/python main.py --dataset SEGTHOR_PSEUDO --model enet \
  --experiment B0 --mode full --epochs 30 --seed 1000 \
  --dest results/segthor_pseudo/enet/B0 --gpu
```

Use `--model unet` with a different destination for U-Net. CT PNGs and existing
patient splits are preserved; masks use the existing nearest-neighbour resize
and 0,63,126,189,252 encoding. Dataset dimensions, HU windowing, networks, loss,
and augmentation settings are unchanged. Provenance is carried into training
results. The class-comparison plot identifies pseudo-label agreement explicitly.

If you created the dataset as `data/SEGTHOR_FIXED`, use `--dataset SEGTHOR_FIXED`
instead. It is also registered with five classes and requires separate organ
labels. The dataset argument must match the directory name under `data/`.

## Method and limits

1. TotalSegmentator `total` task, `roi_subset=['esophagus', 'aorta']`, `fast=False`.
   Input is the original CT, not the resized/windowed training PNGs.
2. Intersect each prediction with `original_mask == 1`.
3. Retain its interior using a physical-distance erosion of **1 mm**. These are
   geometric seed regions, not calibrated confidence estimates. Require at least
   **8 seed voxels per organ**. Both values are configurable via
   `--seed-margin-mm` and `--min-seed-voxels`; defaults are starting heuristics.
4. Watershed on the negative 3D Euclidean distance to the merged-mask boundary,
   using voxel spacing, 6-connectivity and the two named seed labels. Every merged
   component must contain a seed. No watershed/background gap is inserted.
5. Assert exact merged-region coverage and preservation of labels 0, 2, 3.

Model predictions must be binary, disjoint within class 1, and geometrically
aligned. Missing seeds, unseeded components, invalid labels, and sheared grids
are rejected. Unknown NIfTI spatial units are assumed to be mm and recorded.
Prediction coverage below 50% is flagged as a review warning, not an accuracy
score. Incorrect model predictions can still pass all these checks. The algorithm
preserves the existing outer annotation, including any existing annotation errors.
Keep training and validation patient splits fixed; do not tune these parameters
to improve downstream validation scores against the generated labels.

Implementation: [TotalSegmentator API](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/python_api.py)
and [watershed documentation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html).

Focused tests (no model or GPU required):

```bash
ai4mi/bin/python -m unittest discover -s tests
```
