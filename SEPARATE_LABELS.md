# Separate esophagus and aorta labels

For the automatic pseudo-label workflow when separate annotations are unavailable,
see [AUTO_LABELS.md](AUTO_LABELS.md). Those labels carry explicit machine-generated
provenance and are evaluated as pseudo-label agreement, not verified ground truth.

The course `GT.nii.gz` masks currently merge esophagus and aorta into class 1.
An empty class 4 cannot be restored by changing class names or reallocating IDs.
For the verified-annotation workflow below, original annotations are required. Only Patient_07 currently has a usable
`GT2.nii.gz`; the other 19 patients still need separate annotations.

Run these commands from `/home/mili/Downloads/ai4mi` with the project environment
(`source ai4mi/bin/activate`, or your cluster environment).

The comparison plot now always shows four separate organs. Historical merged-label
results cannot be used with it; the `--label-mode` option has been removed.

## Prepare separate masks

Arrange original annotations as `ORIGINAL_MASKS/Patient_01/GT.nii.gz`, etc.
They must use 0=background, 1=esophagus, 2=heart, 3=trachea, 4=aorta, and match
the shape and affine of the corresponding existing source volume.
Heuristic divisions must not be presented as original anatomical annotations.

```bash
# First audit every patient. Replace /path/to/original_masks with your directory.
python prepare_separate_labels.py \
  --labels-dir /path/to/original_masks --mask-name GT.nii.gz --check-only

# Once the audit passes, create a fresh dataset.
python prepare_separate_labels.py \
  --labels-dir /path/to/original_masks --mask-name GT.nii.gz
```

The output is `data/SEGTHOR_SEPARATE`. CT PNGs are copied unchanged, train/validation
membership is preserved, and masks are resized with nearest-neighbor interpolation
then encoded as 0,63,126,189,252. Mask-source details and counts are recorded in
`label_schema.json`. The script checks every source before writing any output.
With no `--labels-dir`, it audits the current source folders, trying GT2 then GT;
this currently fails for the 19 patients lacking separate masks.

## Retrain and draw the four-organ plot

The saved five-channel models were trained to predict merged class 1 and empty
class 4. Their saved Dice arrays cannot be converted into esophagus/aorta scores.
Use fresh runs, with matched seeds and the same settings across models:

```bash
for model in enet unet; do
  for experiment in B0 B1 B2 B3 B4 B5 B6; do
    python main.py --dataset SEGTHOR_SEPARATE --model "$model" \
      --experiment "$experiment" --epochs 20 --seed 1000 --mode full \
      --dest "results/segthor_separate/$model/$experiment" --gpu || exit 1
  done
done

python plot_class_comparison.py \
  --results results/segthor_separate --experiments B0 B1 B2 B3 B4 B5 B6 \
  --headless
```

Training checks actual PNG labels and records their counts and content hash in
each result directory. The plot refuses legacy results without that record,
empty classes, or comparisons across different masks/splits. Use new result
directories: training refuses to overwrite a nonempty run.

The metric remains your original mean Dice over all validation slices at the
saved best epoch, including Dice=1 for individual empty/empty slices. This is
not 3D per-patient Dice or Dice restricted to slices containing the organ.
