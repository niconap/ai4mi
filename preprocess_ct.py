import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.transform import resize

from segthor_labels import decode_png, inspect_dataset

WINDOWS = {'soft': [(-160., 240.)], 'dual': [(-160., 240.), (-1000., 400.)]}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def patient_slices(root):
    result, owners = {}, {}
    for split in ('train', 'val', 'test'):
        for image in sorted((Path(root) / split / 'img').glob('*.png')):
            patient, z = image.stem.rsplit('_', 1)
            if patient in owners and owners[patient] != split:
                raise ValueError(f'Patient leakage: {patient}')
            owners[patient] = split
            result.setdefault(patient, {'split': split, 'indices': []})['indices'].append(int(z))
            if split != 'test' and not (Path(root) / split / 'gt' / image.name).is_file():
                raise ValueError(f'Missing reference mask for {image.name}')
    return result


def geometry(image, canonical):
    axes = image.affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    directions = np.abs(axes / spacing)
    unit = image.header.get_xyzt_units()[0]
    factor = {'unknown': 1., 'mm': 1., 'meter': 1000., 'micron': .001}[unit]
    source = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(('R', 'A', 'S')) if canonical else source
    transform = nib.orientations.ornt_transform(source, target)
    inverse = nib.orientations.ornt_transform(target, source)
    order = np.argsort(transform[:, 0].astype(int))
    return spacing[order] * factor, transform, inverse


def body_bounds(ct, spacing, margin_mm):
    footprint = np.zeros(ct.shape[:2], bool)
    for z in range(ct.shape[2]):
        components, n = ndi.label(ct[:, :, z] > -500)
        if n:
            counts = np.bincount(components.ravel()); counts[0] = 0
            footprint |= ndi.binary_fill_holes(components == counts.argmax())
    box = ndi.find_objects(footprint.astype(np.uint8))[0]
    margin = np.ceil(margin_mm / np.asarray(spacing[:2])).astype(int)
    return [[max(0, s.start - int(m)), min(ct.shape[i], s.stop + int(m))]
            for i, (s, m) in enumerate(zip(box, margin))]


def map_plane(array, record, order):
    box = tuple(slice(*b) for b in record['crop'])
    array = array[box]
    if record['geometry'] == 'resize':
        return resize(array, (256, 256), order=order, mode='constant',
                      preserve_range=True, anti_aliasing=False)
    ratio = np.asarray(record['output_spacing'][:2]) / record['spacing'][:2]
    return ndi.affine_transform(array, np.diag(ratio), output_shape=record['output_shape'],
                                order=order, mode='nearest', prefilter=False)


def restore_probabilities(probabilities, record, reference_shape):
    if record['geometry'] == 'resize' and record['crop'] == [[0, record['canonical_shape'][0]], [0, record['canonical_shape'][1]]]:
        output = []
        for plane in probabilities:
            native = nib.orientations.apply_orientation(plane[:, :, None],
                        np.asarray(record['inverse_orientation']))[:, :, 0]
            output.append(resize(native, reference_shape, order=1, mode='edge',
                                 preserve_range=True, anti_aliasing=False))
        output = np.clip(np.stack(output), 0, 1).astype(np.float32)
        return output / np.maximum(output.sum(0, keepdims=True), 1e-8)
    crop_shape = [b - a for a, b in record['crop']]
    output = []
    for k, plane in enumerate(probabilities):
        if record['geometry'] == 'resize':
            cropped = resize(plane, crop_shape, order=1, mode='edge',
                             preserve_range=True, anti_aliasing=False)
        else:
            ratio = np.asarray(record['spacing'][:2]) / record['output_spacing'][:2]
            cropped = ndi.affine_transform(plane, np.diag(ratio), output_shape=crop_shape,
                                           order=1, mode='nearest', prefilter=False)
        canonical = np.full(record['canonical_shape'][:2], float(k == 0), np.float32)
        canonical[tuple(slice(*b) for b in record['crop'])] = cropped
        native = nib.orientations.apply_orientation(canonical[:, :, None],
                                                     np.asarray(record['inverse_orientation']))[:, :, 0]
        output.append(resize(native, reference_shape, order=1, mode='edge',
                             preserve_range=True, anti_aliasing=False))
    output = np.clip(np.stack(output), 0, 1).astype(np.float32)
    return output / np.maximum(output.sum(0, keepdims=True), 1e-8)


def prepare(reference_data, source_dir, output, *, labels_dir=None, mask_name='GT.nii.gz',
            geometry_mode='resize', canonical=False, spacing_mode='native', spacing_mm=None,
            crop_body=False, margin_mm=20., windows='soft', fitted_plan=None, dataset_id=None):
    reference_data, source_dir, output = map(Path, (reference_data, source_dir, output))
    options = dict(geometry=geometry_mode, canonical=canonical, spacing_mode=spacing_mode,
                   spacing_mm=spacing_mm, crop_body=crop_body, margin_mm=margin_mm, windows=windows)
    frozen = json.loads(Path(fitted_plan).read_text()) if fitted_plan else None
    if frozen:
        options = frozen['options']
    groups = patient_slices(reference_data)
    if frozen:
        leaked = set(frozen['fitted_on']) & {p for p, r in groups.items() if r['split'] != 'train'}
        if leaked:
            raise ValueError(f'Patient leakage from fitted training statistics: {sorted(leaked)}')
    train = sorted(p for p, r in groups.items() if r['split'] == 'train')
    report = inspect_dataset(reference_data, require_separate=True) if train else None
    headers, records = {}, {}
    for patient, group in groups.items():
        ct_path = source_dir / patient / f'{patient}.nii.gz'
        if not ct_path.exists():
            ct_path = source_dir / f'{patient}.nii.gz'
        ct = nib.load(ct_path)
        spacing, transform, inverse = geometry(ct, options['canonical'])
        label_path = None
        if group['split'] != 'test':
            label_path = (Path(labels_dir) if labels_dir else source_dir) / patient / mask_name
            gt = nib.load(label_path)
        headers[patient] = (ct, ct_path, label_path)
        records[patient] = dict(split=group['split'], indices=group['indices'], spacing=spacing.tolist(),
                                orientation=transform.tolist(), inverse_orientation=inverse.tolist(),
                                original_shape=list(ct.shape), source_orientation=list(nib.aff2axcodes(ct.affine)),
                                source_ct=str(ct_path.resolve()),
                                source_mask=str(label_path.resolve()) if label_path else None)
    orientations = {tuple(r['source_orientation']) for r in records.values()}
    if frozen:
        orientations |= {tuple(r['source_orientation']) for r in frozen['patients'].values()}
    fitted_spacing = frozen['fitted_spacing'] if frozen else np.median([records[p]['spacing'][:2] for p in train], axis=0).tolist()
    target = options['spacing_mm'] or (fitted_spacing if options['spacing_mode'] == 'train-median' else None)
    output.mkdir(parents=True)
    sums = np.zeros(len(WINDOWS[options['windows']]), np.float64)
    squares = sums.copy(); count = 0
    samples = {'train': [], 'val': [], 'test': []}
    for patient, record in records.items():
        print(f'{patient}: {record["split"]}', flush=True)
        ct_image, ct_path, label_path = headers[patient]
        raw = ct_image.get_fdata(dtype=np.float32)
        labels = np.asanyarray(nib.load(label_path).dataobj) if label_path else None
        if labels is not None:
            for z in record['indices']:
                path = reference_data / record['split'] / 'gt' / f'{patient}_{z:04d}.png'
                with Image.open(path) as im:
                    reference = decode_png(np.asarray(im))
                expected = resize(labels[:, :, z], reference.shape, order=0, mode='constant',
                                  preserve_range=True, anti_aliasing=False).astype(np.uint8)
                if not np.array_equal(reference, expected):
                    raise ValueError(f'{path}: does not match the supplied 3D labels.')
        transform = np.asarray(record['orientation'])
        ct = nib.orientations.apply_orientation(raw, transform)
        gt = nib.orientations.apply_orientation(labels, transform) if labels is not None else None
        record['canonical_shape'] = list(ct.shape)
        record['crop'] = body_bounds(ct, record['spacing'], options['margin_mm']) if options['crop_body'] else [[0, ct.shape[0]], [0, ct.shape[1]]]
        record['geometry'] = options['geometry']
        record['output_spacing'] = list(target or record['spacing'][:2]) + [record['spacing'][2]]
        extent = np.array([b - a for a, b in record['crop']])
        record['output_shape'] = ([256, 256] if options['geometry'] == 'resize' else
                                  (np.ceil((extent - 1) * np.array(record['spacing'][:2]) /
                                           record['output_spacing'][:2]).astype(int) + 1).tolist())
        if options['geometry'] == 'resize':
            record['output_spacing'][:2] = (extent * np.array(record['spacing'][:2]) / 256).tolist()
        record['ct_sha256'] = file_sha256(ct_path)
        record['mask_sha256'] = file_sha256(label_path) if label_path else None
        split = record['split']
        for folder in ('img', 'gt', 'reference_gt'):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        record['slice_order'] = sorted(record['indices'], reverse=bool(transform[2, 1] < 0))
        for original_z in record['slice_order']:
            z = original_z if transform[2, 1] > 0 else ct.shape[2] - 1 - original_z
            stem = f'{patient}_{original_z:04d}'
            plane = ct[:, :, z]
            channels = []
            for low, high in WINDOWS[options['windows']]:
                windowed = (np.clip(plane, low, high) - low) / (high - low)
                channels.append(map_plane(windowed, record, 1))
            image = np.clip(np.stack(channels), 0, 1).astype(np.float32)
            np.save(output / split / 'img' / f'{stem}.npy', image, allow_pickle=False)
            present = []
            if gt is not None:
                source_gt = gt[:, :, z]
                cropped = source_gt[tuple(slice(*b) for b in record['crop'])]
                mapped = map_plane(source_gt, record, 0).astype(np.uint8)
                Image.fromarray(mapped * 63).save(output / split / 'gt' / f'{stem}.png')
                shutil.copy2(reference_data / split / 'gt' / f'{stem}.png', output / split / 'reference_gt' / f'{stem}.png')
                present = np.unique(mapped).tolist()
            samples[split].append({'stem': stem, 'patient': patient, 'classes': present})
            if split == 'train' and not frozen:
                support = map_plane((plane > -500).astype(np.uint8), record, 0).astype(bool)
                values = image[:, support].astype(np.float64)
                sums += values.sum(1); squares += (values ** 2).sum(1); count += values.shape[1]
        del raw, ct, labels, gt
    if frozen:
        stats = frozen['intensity_statistics']
    else:
        if count == 0:
            raise ValueError('No training body voxels for intensity statistics.')
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
        stats = {'mean': mean.tolist(), 'std': std.tolist(), 'voxels': count,
                 'selection': 'CT > -500 HU, training slices only; windowed [0,1] values',
                 'patients': train}
    if dataset_id:
        report.update(dataset_id=dataset_id, label_source='annotated',
                      annotation_origin='Supplied GT.nii.gz; verified against reference PNG masks',
                      patient_splits={split: sorted(p for p, r in records.items() if r['split'] == split)
                                      for split in ('train', 'val', 'test')},
                      source_masks={p: {'path': r['source_mask'], 'sha256': r['mask_sha256'],
                                        'label_source': 'annotated'}
                                    for p, r in records.items() if r['source_mask']})
    plan = {'version': 1, 'options': options, 'fitted_spacing': fitted_spacing,
            'fitted_on': frozen['fitted_on'] if frozen else train,
            'intensity_statistics': stats, 'patients': records, 'samples': samples,
            'reference_report': report, 'reference_data': str(reference_data.resolve())}
    plan['signature'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    (output / 'preprocessing.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {output}; training median XY spacing: {fitted_spacing} mm')
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-data', type=Path, default=Path('data/SEGTHOR_FULL'))
    p.add_argument('--source-dir', type=Path, default=Path('data/segthor_train_full/train'))
    p.add_argument('--labels-dir', type=Path, help='Mask root; defaults to --source-dir.')
    p.add_argument('--mask-name', default='GT.nii.gz')
    p.add_argument('--dataset-id', choices=['SEGTHOR_FULL'],
                   help='Declare supplied full-data annotations; verify GT and record training provenance.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry', choices=['resize', 'physical'], default='resize')
    p.add_argument('--canonical', action='store_true', help='Reorient axial CT/mask together to RAS.')
    p.add_argument('--spacing', choices=['native', 'train-median'], default='native')
    p.add_argument('--spacing-mm', type=float, nargs=2, metavar=('X', 'Y'))
    p.add_argument('--crop-body', action='store_true')
    p.add_argument('--margin-mm', type=float, default=20.)
    p.add_argument('--windows', choices=WINDOWS, default='soft')
    p.add_argument('--fitted-plan', type=Path, help='Reuse its frozen geometry options and training statistics, including for test-only data.')
    a = p.parse_args()
    try:
        prepare(a.reference_data, a.source_dir, a.output, labels_dir=a.labels_dir,
                mask_name=a.mask_name, geometry_mode=a.geometry, canonical=a.canonical,
                spacing_mode=a.spacing, spacing_mm=a.spacing_mm, crop_body=a.crop_body,
                margin_mm=a.margin_mm, windows=a.windows, fitted_plan=a.fitted_plan,
                dataset_id=a.dataset_id)
    except (ValueError, OSError) as error:
        p.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
