"""Rebuild PNG masks from separate labels, preserving existing CTs and splits.

No split is inferred here. Explicitly mark machine-generated inputs as pseudo-labels.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize

from segthor_labels import inspect_dataset, require_separate_volume
from auto_split_class1 import file_sha256


def prepare(existing_data, source_dir, labels_dir, output, mask_name='auto', check_only=False,
            label_source='annotated'):
    existing_data, source_dir, labels_dir, output = map(
        Path, (existing_data, source_dir, labels_dir, output))
    if output.exists() and not check_only:
        raise ValueError(f'Output already exists: {output}. Use a fresh directory.')
    if label_source not in ('annotated', 'pseudo'):
        raise ValueError('label_source must be annotated or pseudo.')
    inspect_dataset(existing_data)
    groups = defaultdict(list)
    for split in ('train', 'val'):
        for path in sorted((existing_data / split / 'gt').glob('*.png')):
            patient, index = path.stem.rsplit('_', 1)
            groups[patient].append((split, path, int(index)))
    chosen, errors = {}, []
    names = ['GT2.nii.gz', 'GT.nii.gz'] if mask_name == 'auto' else [mask_name]
    # Complete the entire source-label audit before creating any output.
    for patient, slices in sorted(groups.items()):
        reference = nib.load(source_dir / patient / 'GT.nii.gz')
        problems = []
        for name in names:
            path = labels_dir / patient / name
            if not path.is_file():
                problems.append(f'{name}: missing file')
                continue
            try:
                provenance = None
                report_path = labels_dir / patient / 'pseudo_label_report.json'
                if label_source == 'pseudo':
                    provenance = json.loads(report_path.read_text())
                    if (provenance.get('status') != 'generated' or
                            provenance.get('label_source') != 'pseudo' or
                            provenance.get('output_mask_sha256') != file_sha256(path) or
                            provenance.get('source_mask_sha256') != file_sha256(
                                source_dir / patient / 'GT.nii.gz')):
                        raise ValueError('Pseudo-label provenance does not match these masks.')
                elif report_path.exists() or name == 'GT_pseudo.nii.gz':
                    raise ValueError('Machine-generated labels require --label-source pseudo.')
                volume = nib.load(path)
                if (volume.shape != reference.shape or
                        not np.allclose(volume.affine, reference.affine, rtol=0, atol=1e-4)):
                    raise ValueError('Shape/affine differs from the existing reference mask.')
                if any(index < 0 or index >= volume.shape[2] for _, _, index in slices):
                    raise ValueError('Existing slice index is outside the source volume.')
                counts = require_separate_volume(np.asanyarray(volume.dataobj))
            except (ValueError, OSError) as error:
                problems.append(f'{name}: {error}')
                continue
            chosen[patient] = {'path': str(path.resolve()), 'voxel_counts': counts,
                               'label_source': label_source}
            if provenance is not None:
                chosen[patient]['pseudo_label_report'] = provenance
            print(f'{patient}: accepted {name}', flush=True)
            break
        else:
            errors.append(f'{patient}: ' + '; '.join(problems))
    if errors:
        raise ValueError('Cannot prepare separate labels:\n' + '\n'.join(errors))
    if check_only:
        print(f'All {len(chosen)} patients have separate annotations; no files written.')
        return
    output.mkdir(parents=True)
    for patient, slices in sorted(groups.items()):
        labels = np.asanyarray(nib.load(chosen[patient]['path']).dataobj).astype(np.uint8)
        for split, old_mask, index in slices:
            with Image.open(old_mask) as image:
                shape = np.asarray(image).shape
            mask = resize(labels[:, :, index], shape, order=0, mode='constant',
                          preserve_range=True, anti_aliasing=False).astype(np.uint8)
            for subfolder in ('img', 'gt'):
                (output / split / subfolder).mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask * 63).save(output / split / 'gt' / old_mask.name)
            shutil.copy2(existing_data / split / 'img' / old_mask.name,
                         output / split / 'img' / old_mask.name)
    if (existing_data / 'spacing.pkl').is_file():
        shutil.copy2(existing_data / 'spacing.pkl', output / 'spacing.pkl')
    report = inspect_dataset(output, require_separate=True)
    report['label_source'] = label_source
    report['source_masks'] = chosen
    (output / 'label_schema.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {output}: class 1 = esophagus, class 4 = aorta.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-data', type=Path, default=Path('data/SEGTHOR'))
    parser.add_argument('--source-dir', type=Path, default=Path('data/segthor_part1/train'),
                        help='Existing source masks, arranged as Patient_XX/GT.nii.gz.')
    parser.add_argument('--labels-dir', type=Path,
                        help='Separate masks under Patient_XX/; defaults to source-dir.')
    parser.add_argument('--mask-name', default='auto',
                        help='Filename in each patient folder; auto checks GT2.nii.gz then GT.nii.gz.')
    parser.add_argument('--output', type=Path, default=Path('data/SEGTHOR_SEPARATE'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--label-source', choices=['annotated', 'pseudo'], default='annotated',
                        help='Use pseudo for outputs of auto_split_class1.py.')
    args = parser.parse_args()
    try:
        prepare(args.existing_data, args.source_dir, args.labels_dir or args.source_dir,
                args.output, args.mask_name, args.check_only, args.label_source)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
