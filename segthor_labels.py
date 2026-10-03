import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
PNG_VALUES = np.array([0, 63, 126, 189, 252], dtype=np.uint8)


def decode_png(array):
    array = np.asarray(array)
    if array.ndim != 2 or not np.isin(array, PNG_VALUES).all():
        raise ValueError('Incorrect values')
    return (array // 63).astype(np.uint8)


def require_separate_volume(array):
    array = np.asarray(array)
    if array.ndim != 3 or not np.isin(array, range(5)).all():
        raise ValueError('Wrong mask')
    counts = np.bincount(array.astype(np.uint8).ravel(), minlength=5)
    missing = [CLASS_NAMES[i] for i in range(1, 5) if counts[i] == 0]
    if missing:
        raise ValueError('Missing annotated organ(s): ' + ', '.join(missing)
                         + 'Original separate-organ mask; '
                         'merged class cant be split')
    return counts.tolist()


def inspect_dataset(root, require_separate=False):
    root = Path(root)
    splits = {}
    digest = hashlib.sha256()
    patient_sets = []
    for split in ('train', 'val'):
        files = sorted((root / split / 'gt').glob('*.png'))
        images = sorted((root / split / 'img').glob('*.png'))
        if not files or [p.stem for p in files] != [p.stem for p in images]:
            raise ValueError(f'{root / split}: images and masks must match by filename.')
        patient_sets.append({p.stem.rsplit('_', 1)[0] for p in files})
        counts = np.zeros(5, dtype=np.int64)
        positive = np.zeros(5, dtype=np.int64)
        for path in files:
            with Image.open(path) as image:
                labels = decode_png(np.asarray(image))
            per_slice = np.bincount(labels.ravel(), minlength=5)
            counts += per_slice
            positive += per_slice > 0
            digest.update(str(path.relative_to(root)).encode() + b'\0')
            digest.update(labels.tobytes())
        splits[split] = {'slices': len(files), 'pixels': counts.tolist(),
                         'positive_slices': positive.tolist()}
    if patient_sets[0] & patient_sets[1]:
        raise ValueError('Leakage between train and val splits.')
    separate = all(all(s['pixels'][i] > 0 for i in range(1, 5))
                   for s in splits.values())
    merged = all(s['pixels'][4] == 0 and all(s['pixels'][i] > 0 for i in (1, 2, 3))
                 for s in splits.values())
    if require_separate and not separate:
        raise ValueError('Separate esophagus/aorta labels required'
                         'prepare_separate_labels.py with original masks first')
    if not separate and not merged:
        raise ValueError('wrong label')
    report = {'schema': 'separate' if separate else 'merged',
            'class_names': CLASS_NAMES if separate else
            ['Background', 'Esophagus + aorta', 'Heart', 'Trachea', 'Empty class'],
            'mask_sha256': digest.hexdigest(), 'splits': splits}
    provenance_path = root / 'label_schema.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for key in ('label_source', 'source_masks'):
            if key in provenance:
                report[key] = provenance[key]
    return report
