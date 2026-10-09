To get the datasets for the preprocessing variants

```bash

# Float32 control: existing HU window and direct 256×256 whole-slice resize.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --output data/CT_FULL_P0

# Orientation alone: the same control with CT and GT reoriented to RAS.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --canonical --output data/CT_FULL_orientation

# Body crop alone: CT-only crop with a 20 mm margin, then resize to 256×256.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --crop-body --output data/CT_FULL_crop

# Native-spacing patch control: no XY resampling; consistent orientation.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --canonical --geometry physical --output data/CT_FULL_native_patches

# Spacing experiment: same patches/orientation, but training-median XY spacing.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --canonical --geometry physical --spacing train-median --output data/CT_FULL_spacing

# Geometry combination: spacing + body crop.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --canonical --geometry physical --spacing train-median --crop-body --output data/CT_FULL_spacing_crop

# Second window alone, compared with CT_FULL_P0.
python preprocess_ct.py --dataset-id SEGTHOR_FULL --windows dual --output data/CT_FULL_dual

python preprocess_ct.py --dataset-id SEGTHOR_FULL --reference-data data/SEGTHOR_FULL \
  --source-dir data/segthor_train_full/train \
  --mask-name GT.nii.gz --output data/CT_FULL_P0
```

To train the preprocessing versions:

```bash
# 1. Float32 control.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_P0 \
  --model enet --experiment B0 --mode full --epochs 50 --seed 1000 \
  --dest results/segthor_full/enet/P0_float --gpu

# 2. Intensity normalization alone; reuse CT_FULL_P0.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_P0 --normalization zscore --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_zscore --gpu

# 3. Body crop alone; compare with P0_float.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_crop --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_crop --gpu

# 4. Native-spacing patch control; needed to isolate the effect of spacing.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_native_patches --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_native_patches --gpu

# 5. Standardized spacing; compare with P_native_patches.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_spacing --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_spacing --gpu

# Organ-aware slice sampling alone.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_P0 --slice-sampling organ --organ-sampling-prob 0.5 --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_sampling --gpu

# 2.5D alone: three adjacent slices, center-slice target.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_P0 --context 3 --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_context --gpu

# Second CT window alone: the extra channel is stored in CT_FULL_dual.
python -O main.py --dataset SEGTHOR_FULL --data-root data/CT_FULL_dual --model enet --experiment B0 --mode full --epochs 50 --seed 1000 --dest results/segthor_full/enet/P_dual --gpu

```