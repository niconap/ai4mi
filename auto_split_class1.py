"""Create esophagus/aorta pseudo-labels using TotalSegmentator and 3D watershed.

Only original class 1 is partitioned; 0, 2 and 3 remain byte-for-byte unchanged.
Outputs are automatic pseudo-labels, not independently verified annotations.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_geometry(image, reference, name):
    if (image.shape != reference.shape or len(image.shape) != 3 or
            not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)):
        raise ValueError(f'{name}: shape/affine differs from the source mask. '
                         'Use predictions on the original CT grid.')


def split_class1(labels, esophagus, aorta, spacing, seed_margin_mm=1.0,
                 min_seed_voxels=8):
    """Partition merged labels with eroded model predictions as named markers.

    Distance/erosion use physical voxel spacing. Watershed is restricted to the
    merged mask, with 6-connectivity and no background line between organs.
    Reject missing/overlapping seeds and any disconnected region without seeds.
    """
    if labels.ndim != 3 or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Expected a merged 3D mask containing only labels 0,1,2,3.')
    if esophagus.shape != labels.shape or aorta.shape != labels.shape:
        raise ValueError('Prediction shapes must match the merged mask.')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Expected three positive finite voxel spacings in mm.')
    if not np.isfinite(seed_margin_mm) or seed_margin_mm < 0 or min_seed_voxels < 1:
        raise ValueError('Seed margin must be nonnegative; minimum seed count must be positive.')
    merged = labels == 1
    if not merged.any():
        raise ValueError('Source mask has no class 1.')
    # Tight bounding box with an explicit zero border avoids whole-volume EDTs
    # and correctly handles masks that touch a volume edge.
    bounds = ndi.find_objects(merged.astype(np.uint8))[0]
    region = np.pad(merged[bounds], 1)
    predictions = [np.pad(np.asarray(p[bounds], dtype=bool), 1) & region
                   for p in (esophagus, aorta)]
    if (predictions[0] & predictions[1]).any():
        raise ValueError('Esophagus/aorta predictions overlap inside class 1.')
    markers = np.zeros(region.shape, dtype=np.uint8)
    seed_counts = {}
    for class_id, name, prediction in zip((1, 4), ('esophagus', 'aorta'), predictions):
        interior = ndi.distance_transform_edt(prediction, sampling=spacing)
        seeds = prediction & (interior > seed_margin_mm)
        count = int(seeds.sum())
        if count < min_seed_voxels:
            raise ValueError(f'{name}: only {count} interior seed voxels; '
                             'prediction missing or too thin after erosion. No split written.')
        markers[seeds] = class_id
        seed_counts[name] = count
    connectivity = ndi.generate_binary_structure(3, 1)
    components, count = ndi.label(region, structure=connectivity)
    seeded_components = np.unique(components[markers > 0])
    missing = np.setdiff1d(np.arange(1, count + 1), seeded_components)
    if len(missing):
        raise ValueError(f'{len(missing)} disconnected class-1 component(s) have no seeds; '
                         'cannot assign an organ safely. No split written.')
    distance = ndi.distance_transform_edt(region, sampling=spacing)
    separated = watershed(-distance, markers, mask=region,
                          connectivity=connectivity, watershed_line=False)[1:-1, 1:-1, 1:-1]
    result = labels.astype(np.uint8, copy=True)
    view = result[bounds]
    view[merged[bounds]] = separated[merged[bounds]]
    if (not np.array_equal((result == 1) | (result == 4), merged) or
            not np.array_equal(result[~merged], labels[~merged])):
        raise RuntimeError('Split violated mask preservation constraints.')
    coverage = float((predictions[0] | predictions[1]).sum() / merged.sum())
    return result, {
        'seed_voxels': seed_counts, 'prediction_coverage_of_class1': coverage,
        'class1_components': int(count),
        'output_voxels': {'esophagus': int((result == 1).sum()),
                          'aorta': int((result == 4).sum())},
        'review_recommended': True,
        'warnings': ['Model predictions cover less than half of merged class 1.']
                    if coverage < 0.5 else [],
    }


def generate_predictions(ct_path, output, device):
    """Lazy import: cached-prediction mode needs no TotalSegmentator install."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as error:
        raise RuntimeError('Install requirements-pseudolabels.txt, or supply '
                           '--predictions-dir with existing binary predictions.') from error
    totalsegmentator(ct_path, output, task='total',
                    roi_subset=['esophagus', 'aorta'], fast=False, ml=False,
                    device=device, nr_thr_saving=1)
    return version('TotalSegmentator')


def process_patient(source_dir, patient, output_dir, predictions_dir=None,
                    device='gpu', seed_margin_mm=1.0, min_seed_voxels=8):
    """Write one new mask and its provenance; never replace a source mask."""
    source = Path(source_dir) / patient
    destination = Path(output_dir) / patient
    if destination.exists():
        raise ValueError(f'{destination} already exists. Choose a fresh output directory.')
    ct_path, mask_path = source / f'{patient}.nii.gz', source / 'GT.nii.gz'
    ct, original = nib.load(ct_path), nib.load(mask_path)
    check_geometry(ct, original, 'CT')
    labels = np.asanyarray(original.dataobj)
    if labels.ndim != 3 or not np.isin(labels, [0, 1, 2, 3]).all() or not (labels == 1).any():
        raise ValueError('Expected merged source labels 0,1,2,3 with nonempty class 1.')
    # EDT with axis spacing assumes orthogonal axes; do not silently ignore shear.
    axes = original.affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if not np.allclose((axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-4):
        raise ValueError('Sheared voxel grid is unsupported.')
    unit = original.header.get_xyzt_units()[0]
    if unit in ('meter', 'micron'):
        spacing *= {'meter': 1000, 'micron': .001}[unit]
    destination.mkdir(parents=True)
    prediction_folder = (Path(predictions_dir) / patient if predictions_dir is not None
                         else destination / 'predictions')
    model_version = None
    if predictions_dir is None:
        model_version = generate_predictions(ct_path, prediction_folder, device)
    predictions, prediction_hashes = [], {}
    for organ in ('esophagus', 'aorta'):
        path = prediction_folder / f'{organ}.nii.gz'
        prediction = nib.load(path)
        check_geometry(prediction, original, organ)
        array = np.asanyarray(prediction.dataobj)
        if not np.isin(array, [0, 1]).all():
            raise ValueError(f'{path}: expected a binary prediction, not a multilabel map.')
        predictions.append(array.astype(bool))
        prediction_hashes[organ] = file_sha256(path)
    result, report = split_class1(labels, *predictions, spacing,
                                  seed_margin_mm, min_seed_voxels)
    output_path = destination / 'GT_pseudo.nii.gz'
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(result, original.affine, header)
    # Retain reference qform/sform, including their codes.
    for form in ('qform', 'sform'):
        matrix, code = getattr(original, f'get_{form}')(coded=True)
        getattr(output_image, f'set_{form}')(matrix, int(code))
    report.update({
        'status': 'generated', 'label_source': 'pseudo', 'patient': patient,
        'method': 'TotalSegmentator interior seeds + 3D distance watershed',
        'totalsegmentator_version': model_version,
        'predictions_source': str(prediction_folder.resolve()),
        'prediction_sha256': prediction_hashes,
        'source_ct_sha256': file_sha256(ct_path), 'source_mask_sha256': file_sha256(mask_path),
        'seed_margin_mm': seed_margin_mm, 'min_seed_voxels': min_seed_voxels,
        'spacing_mm': spacing.tolist(),
        'spacing_unit_assumption': 'mm' if unit == 'unknown' else None,
    })
    nib.save(output_image, output_path)
    report['output_mask_sha256'] = file_sha256(output_path)
    (destination / 'pseudo_label_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('data/segthor_part1/train'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/segthor_pseudo_masks'))
    parser.add_argument('--patients', nargs='+', help='Default: every Patient_* directory.')
    parser.add_argument('--predictions-dir', type=Path,
                        help='Reuse Patient_XX/{esophagus,aorta}.nii.gz; skips model inference.')
    parser.add_argument('--device', default='gpu', help='gpu, gpu:0, cpu, or mps.')
    parser.add_argument('--seed-margin-mm', type=float, default=1.0)
    parser.add_argument('--min-seed-voxels', type=int, default=8)
    args = parser.parse_args()
    patients = args.patients or sorted(p.name for p in args.source_dir.glob('Patient_*') if p.is_dir())
    if not patients or any(Path(p).name != p or not p.startswith('Patient_') for p in patients):
        parser.error('Supply valid Patient_* directory names.')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a fresh destination.')
    if not np.isfinite(args.seed_margin_mm) or args.seed_margin_mm < 0 or args.min_seed_voxels < 1:
        parser.error('Invalid seed parameters.')
    args.output_dir.mkdir(parents=True)
    summary = {'label_source': 'pseudo', 'patients': {}}
    for patient in patients:
        print(f'{patient}: generating pseudo-labels...', flush=True)
        try:
            report = process_patient(args.source_dir, patient, args.output_dir,
                                     args.predictions_dir, args.device,
                                     args.seed_margin_mm, args.min_seed_voxels)
            summary['patients'][patient] = report
            print(f'  Saved GT_pseudo.nii.gz; coverage '
                  f'{report["prediction_coverage_of_class1"]:.1%}', flush=True)
        except (ValueError, OSError, RuntimeError) as error:
            summary['patients'][patient] = {'status': 'failed', 'error': str(error)}
            print(f'  FAILED: {error}', flush=True)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if any(r['status'] == 'failed' for r in summary['patients'].values()):
        parser.exit(1, 'Some patients failed. See summary.json; do not build a partial dataset.\n')


if __name__ == '__main__':
    main()
