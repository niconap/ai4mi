import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from segthor_labels import decode_png, inspect_dataset

METRICS = ('loss', 'dice', 'iou', 'hd95')
COLORS = ('#2563a6', '#d46a24', '#29805a', '#8b59a3')


def load_run(folder):
    config = json.loads((folder / 'config.json').read_text())
    schema = json.loads((folder / 'label_schema.json').read_text())
    if config['model'] not in ('vnet', 'vnet25d') or config.get('debug'):
        raise ValueError(f'{folder}: expected a full VNet run.')
    arrays = {f'{metric}_{split}': np.load(folder / f'{metric}_{split}.npy')
              for metric in METRICS for split in ('tra', 'val')}
    dice = arrays['dice_val']
    recorded = np.any(dice != 0, axis=(1, 2))
    epochs = int(recorded.sum())
    match = re.search(r'epoch (\d+):', (folder / 'best_epoch.txt').read_text())
    curves = {}
    for split, subset in (('tra', 'train'), ('val', 'val')):
        expected = (len(dice), schema['splits'][subset]['slices'], 5)
        written = np.any(arrays[f'dice_{split}'][:epochs] != 0, axis=2)
        counts = written.sum(1)
        for metric in METRICS:
            array = arrays[f'{metric}_{split}']
            valid_shape = (array.ndim == 2 and len(array) == len(dice)) if metric == 'loss' else array.shape == expected
            curves[f'{metric}_{split}'] = (array[:epochs].mean(1) if metric == 'loss' else
                np.array([row[:n, 1:].mean() for row, n in zip(array[:epochs], counts)]))
    return dict(name=folder.name, config=config, schema=schema, best=int(match[1]),
                epochs=epochs, curves=curves, dice=dice[:epochs])


def style_axis(ax, ylabel, bounded=False):
    ax.set_xlabel('Epoch (1-based)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    if bounded:
        ax.set_ylim(-.02, 1.02)
    else:
        ax.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/segthor_full/vnet'))
    parser.add_argument('--experiments', nargs='+', default=['B0', 'B1'])
    parser.add_argument('--reference-data', type=Path, default=Path('data/SEGTHOR_FULL'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = [load_run(args.results / name) for name in args.experiments]
    reference = inspect_dataset(args.reference_data, require_separate=True)
    masks = sorted((args.reference_data / 'val' / 'gt').glob('*.png'))
    present = []
    for path in masks:
        with Image.open(path) as image:
            labels = decode_png(np.asarray(image))
        present.append(np.bincount(labels.ravel(), minlength=5) > 0)
    present = np.asarray(present)
    output = args.output or args.results / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ('Loss', 'Foreground Dice', 'Foreground IoU', 'Foreground HD95')
    units = ('Mean batch loss', 'Mean slice Dice', 'Mean slice IoU', 'Mean slice HD95 (pixels)')
    handles = []
    summary = {}
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        x = np.arange(1, run['epochs'] + 1)
        for ax, metric, title, unit in zip(axes.flat, METRICS, titles, units):
            for split, line in (('tra', '--'), ('val', '-')):
                ax.plot(x, run['curves'][f'{metric}_{split}'], color=color,
                        linestyle=line, linewidth=1.7, alpha=.7 if split == 'tra' else 1)
            best = run['best']
            ax.plot(best + 1, run['curves'][f'{metric}_val'][best], 'o', color=color, ms=6)
            ax.set_title(title)
            style_axis(ax, unit, bounded=metric in ('dice', 'iou'))
        handles.extend([Line2D([], [], color=color, linestyle=line, label=f'{run["name"]} {label}')
                        for line, label in (('--', 'training'), ('-', 'validation'))])
        summary[run['name']] = dict(completed_epochs=run['epochs'], best_epoch_1based=run['best'] + 1,
            saved_best_metrics={key: float(values[run['best']]) for key, values in run['curves'].items()})
    fig.suptitle('VNet learning curves', fontsize=18, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .87))
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    handles = []
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        x = np.arange(1, run['epochs'] + 1)
        scores = {}
        for k, ax in enumerate(axes.flat, start=1):
            all_slices = run['dice'][:, :, k].mean(1)
            gt_present = run['dice'][:, present[:, k], k].mean(1)
            ax.plot(x, all_slices, color=color, linestyle='--', linewidth=1.5, alpha=.6)
            ax.plot(x, gt_present, color=color, linewidth=2)
            ax.plot(run['best'] + 1, gt_present[run['best']], 'o', color=color, ms=6)
            name = reference['class_names'][k]
            ax.set_title(f'{name} · {present[:, k].sum():,} / {len(masks):,} GT-positive slices')
            style_axis(ax, 'Mean validation slice Dice', bounded=True)
            scores[name] = dict(all_slices=float(all_slices[run['best']]),
                               gt_positive_slices=float(gt_present[run['best']]))
        summary[run['name']]['saved_best_organ_dice'] = scores
        handles.extend([Line2D([], [], color=color, linestyle=line, label=f'{run["name"]} {label}')
                        for line, label in (('-', 'GT-positive slices'), ('--', 'all slices'))])
    fig.suptitle('VNet validation Dice by organ', fontsize=18, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .85))
    fig.savefig(output / 'organ_dice.png', dpi=180)
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved plots to {output}')


if __name__ == '__main__':
    main()
