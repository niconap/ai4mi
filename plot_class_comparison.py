"""Compare ENet/U-Net results using explicit, verified organ-label meanings."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np


def read_scores(folder):
    """Legacy merged results cannot be presented as separate-organ measurements."""
    folder = Path(folder)
    schema_path = folder / 'label_schema.json'
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else None
    if schema is None:
        raise ValueError(f'{folder}: no label_schema.json. Retrain on SEGTHOR_FIXED; '
                         'old merged-label scores cannot be split.')
    if schema['schema'] != 'separate':
        raise ValueError(f'{folder}: recorded labels are {schema["schema"]}, not separate.')
    for split in ('train', 'val'):
        if any(n <= 0 for n in schema['splits'][split]['pixels'][1:5]):
            raise ValueError(f'{folder}: an organ is absent from {split} ground truth.')
    match = re.search(r'epoch (\d+):', (folder / 'best_epoch.txt').read_text())
    if match is None:
        raise ValueError(f'Cannot read best epoch from {folder}')
    epoch = int(match.group(1))
    dice = np.load(folder / 'dice_val.npy', mmap_mode='r')
    if dice.ndim != 3 or dice.shape[2] != 5 or epoch >= len(dice) or dice.shape[1] == 0:
        raise ValueError(f'Unexpected Dice shape or epoch in {folder}: {dice.shape}')
    if dice.shape[1] != schema['splits']['val']['slices']:
        raise ValueError(f'{folder}: Dice sample count does not match the recorded dataset.')
    per_class = dice[epoch].mean(axis=0)[1:]
    if not np.isfinite(per_class).all() or ((per_class < 0) | (per_class > 1)).any():
        raise ValueError(f'Invalid Dice in {folder}')
    return epoch, per_class, schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/segthor'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--experiments', nargs='+',
                        default=[f'not_cumulative{i}' for i in range(7)])
    args = parser.parse_args()
    if args.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    experiment_labels = []
    for experiment in args.experiments:
        match = re.fullmatch(r'(?:not_cumulative|B)([0-6])', experiment)
        if match is None:
            parser.error(f'Expected experiment B0-B6 or not_cumulative0-6: {experiment}')
        experiment_labels.append(f'aug {match.group(1)}')
    organs = ['Esophagus', 'Heart', 'Trachea', 'Aorta']
    scores, rows = {}, []
    signature = None
    pseudo_labels = False
    for model in ['enet', 'unet']:
        scores[model] = []
        for experiment in args.experiments:
            folder = args.results / model / experiment
            epoch, per_class, schema = read_scores(folder)
            pseudo_labels |= schema.get('label_source') == 'pseudo'
            current = schema['mask_sha256']
            if signature is not None and signature != current:
                raise ValueError('Runs use different masks or patient splits; comparison aborted.')
            signature = current
            scores[model].append(per_class)
            for class_id, organ in enumerate(organs, 1):
                rows.append([model, experiment, epoch, class_id, organ, per_class[class_id - 1]])
            print(f'{model.upper()} {experiment}, epoch {epoch}: {per_class[:len(organs)].round(4)}')
        scores[model] = np.stack(scores[model])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for class_index, (axis, organ) in enumerate(zip(axes.flat, organs)):
        for model, color, marker in [('enet', '#0072B2', 'o'), ('unet', '#D55E00', 's')]:
            axis.plot(experiment_labels, scores[model][:, class_index],
                      label=model.upper(), color=color, marker=marker, linewidth=2)
        axis.set_title(organ)
        axis.set_ylim(0, 1)
        axis.set_ylabel('Mean validation slice Dice')
        axis.tick_params(axis='x', labelbottom=True)
        axis.grid(alpha=0.25)
        axis.legend(loc='lower right')
    fig.suptitle('ENet vs UNet across augmentation experiments', fontsize=16)
    fig.text(.5, .065, 'aug 0: baseline   |   aug 1: affine   |   aug 2: contrast/gamma   |   '
             'aug 3: noise\naug 4: blur/resolution   |   aug 5: bilateral denoising   |   aug 6: elastic',
             ha='center', fontsize=10)
    note = 'Separate organ annotations'
    if pseudo_labels:
        note = 'Esophagus/aorta: agreement with automatic pseudo-labels, not verified annotations'
    fig.text(.5, .025, note + '\nSaved best epochs; mean over all validation slices.',
             ha='center', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, .12, 1, .95))
    output = args.output or Path('results/enet_unet_separate.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'experiment', 'epoch_zero_based', 'class_id', 'label', 'dice'])
        writer.writerows(rows)
    print(f'Saved {output} and {output.with_suffix(".csv")}')
    if not args.headless:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
