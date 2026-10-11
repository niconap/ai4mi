
import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from plot_class_comparison import read_scores

FAMILIES = ('P', 'P1', 'P2')
EXPERIMENTS = {'P': 'B0', 'P1': 'B1', 'P2': 'B2'}
LABELS = {'P': 'P / B0 — no augmentation', 'P1': 'P1 / B1 — affine',
          'P2': 'P2 / B2 — contrast + gamma'}
COLORS = {'P': '#2766a8', 'P1': '#d16b25', 'P2': '#21826c'}
VARIANTS = {'float': 'Float32 control', 'zscore': 'Z-score', 'crop': 'Body crop',
            'sampling': 'Organ sampling', 'context': '2.5D context',
            'native_patches': 'Native patches', 'spacing': 'Standardized spacing',
            'dual': 'Dual window', 'orientation': 'Orientation',
            'spacing_crop': 'Spacing + crop', 'spacing_crop_zscore': 'Spacing + crop + z-score'}
BASE_METRICS = ('loss_tra', 'loss_val', 'dice_tra', 'dice_val')
EXTRA_METRICS = ('iou_tra', 'iou_val', 'hd95_tra', 'hd95_val')
METRICS = BASE_METRICS + EXTRA_METRICS


def portable_plan(plan):
    result = dict(plan)
    result.pop('signature', None)
    result.pop('reference_data', None)
    result['patients'] = {
        name: {k: v for k, v in patient.items() if k not in ('source_ct', 'source_mask')}
        for name, patient in plan.get('patients', {}).items()}
    if 'reference_report' in plan:
        report = dict(plan['reference_report'])
        if 'source_masks' in report:
            report['source_masks'] = {
                name: {k: v for k, v in source.items() if k != 'path'}
                for name, source in report['source_masks'].items()}
        result['reference_report'] = report
    return result


def model_label(grouped):
    model = next(iter(next(iter(grouped.values())).values())).config['model']
    return {
    'enet': 'ENet',
    'resunetpp': 'ResUNet++',
    'unet': 'U-Net',
    'unet25d': '2.5D U-Net',
    'unet3d': '3D U-Net',}[model]


@dataclass
class Run:
    folder: Path
    family: str
    variant: str
    config: dict
    plan: dict
    schema: dict
    best: int
    completed: int
    planned: int
    curves: dict
    organ_dice: np.ndarray
    train_samples: np.ndarray


def read_run(folder):
    folder = Path(folder)
    match = re.fullmatch(r'P([012])?_(.+)', folder.name)
    if match is None:
        raise ValueError(f'Unrecognized P/P1/P2 run: {folder}')
    family = 'P' if match[1] in (None, '0') else f'P{match[1]}'
    variant = match[2]
    config = json.loads((folder / 'config.json').read_text())
    suffix = re.fullmatch(r'(.+)_B([012])', variant)
    if suffix:
        variant = suffix[1]
        experiment = 'B' + suffix[2]
        if config.get('experiment') != experiment:
            raise ValueError(f'{folder}: name and recorded augmentation disagree.')
        family = {'B0': 'P', 'B1': 'P1', 'B2': 'P2'}[experiment]
    best, organs, schema = read_scores(folder)
    plan = json.loads((folder / 'preprocessing.json').read_text())
    arrays = {key: np.load(folder / f'{key}.npy', allow_pickle=False) for key in BASE_METRICS}
    planned = int(config['epochs'])
    for key, array in arrays.items():
        ndim = 3 if key.startswith('dice') else 2
    recorded = np.any(arrays['dice_val'] != 0, axis=(1, 2))
    completed = int(recorded.sum())
    curves = {key: arrays[key][:completed].mean(axis=1, dtype=np.float64)
              for key in ('loss_tra', 'loss_val')}
    curves['dice_val'] = arrays['dice_val'][:completed, :, 1:].mean(axis=(1, 2), dtype=np.float64)
    train = arrays['dice_tra'][:completed]
    written = np.any(train != 0, axis=2)
    counts = written.sum(axis=1)
    curves['dice_tra'] = np.array([epoch[:n, 1:].mean(dtype=np.float64)
                                   for epoch, n in zip(train, counts)])
    for key in EXTRA_METRICS:
        path = folder / f'{key}.npy'
        if not path.is_file() and key.startswith('hd95'):
            path = folder / f'{key.replace("hd95", "hausdorff95")}.npy'
        if not path.is_file():
            curves[key] = np.full(completed, np.nan)
            continue
        values = np.load(path, allow_pickle=False)
        expected = arrays['dice_tra' if key.endswith('tra') else 'dice_val'].shape
        if values.shape != expected:
            raise ValueError(f'{path}: expected shape {expected}, got {values.shape}.')
        values = values[:completed]
        if not np.isfinite(values).all() or (values < 0).any() or (key.startswith('iou') and (values > 1).any()):
            raise ValueError(f'{path}: invalid metric values.')
        if key.endswith('tra'):
            curves[key] = np.array([epoch[:n, 1:].mean(dtype=np.float64)
                                    for epoch, n in zip(values, counts)])
        else:
            curves[key] = values[:, :, 1:].mean(axis=(1, 2), dtype=np.float64)
    return Run(folder, family, variant, config, plan, schema, best, completed, planned,
               curves, organs, counts)


def collect_runs(root, exclude=()):
    folders = sorted(p for p in Path(root).iterdir()
                     if p.is_dir() and re.fullmatch(r'P([012])?_.+', p.name))
    if not folders:
        raise ValueError(f'No P/P1/P2 result folders in {root}.')
    excluded = set(exclude)
    runs = []
    for folder in folders:
        config = json.loads((folder / 'config.json').read_text())
        if config.get('experiment') not in excluded:
            runs.append(read_run(folder))
    
    grouped = {}
    for run in runs:
        cell = grouped.setdefault(run.variant, {})
        cell[run.family] = run
    keys = ('dataset', 'model', 'mode', 'seed', 'normalization', 'context', 'slice_sampling',
            'organ_sampling_prob', 'debug')
    for variant, family_runs in grouped.items():
        fingerprints = [json.dumps({'config': {k: r.config.get(k) for k in keys},
                                   'preprocessing': portable_plan(r.plan)}, sort_keys=True)
                        for r in family_runs.values()]
    order = sorted(grouped, key=lambda name: (list(VARIANTS).index(name) if name in VARIANTS else len(VARIANTS), name))
    return {name: grouped[name] for name in order}


def selected_families(grouped):
    return tuple(family for family in FAMILIES if any(family in row for row in grouped.values()))


def save_table(grouped, output):
    families = selected_families(grouped)
    rows, row_names = [], list(grouped)
    for variant in row_names:
        row = [VARIANTS.get(variant, variant)]
        for family in families:
            run = grouped[variant].get(family)
            if run is None:
                row += ['—'] * 5
            else:
                marker = '*' if run.completed < run.planned else ''
                row += [f'{run.best + 1}{marker}', f'{run.curves["dice_val"][run.best]:.4f}',
                        f'{run.curves["loss_val"][run.best]:.5f}',
                        ('—' if np.isnan(run.curves['iou_val'][run.best]) else f"{run.curves['iou_val'][run.best]:.4f}"),
                        ('—' if np.isnan(run.curves['hd95_val'][run.best]) else f"{run.curves['hd95_val'][run.best]:.2f}")]
        rows.append(row)
    fig, ax = plt.subplots(figsize=(max(16, 7*len(families)), max(4, 2 + .43*len(rows))))
    fig.patch.set_facecolor('white'); ax.axis('off')
    fig.suptitle(model_label(grouped) + ' · preprocessing and augmentation comparison', fontsize=21, fontweight='bold', y=.965)
    group_width = .78 / len(families)
    widths = [.22] + [group_width*x for x in (.12, .23, .23, .21, .21)]*len(families)
    bbox = [.015, .03, .97, .81]
    table = ax.table(cellText=rows, colLabels=['Preprocessing'] + ['Epoch', 'Val Dice ↑', 'Val loss ↓', 'Val IoU ↑', 'Val HD95 ↓']*len(families),
                     colWidths=widths, cellLoc='center', bbox=bbox)
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dbe3ec'); cell.set_linewidth(.6)
        cell.set_facecolor('#e8edf3' if row == 0 else ('#f4f7fa' if row % 2 == 0 else 'white'))
        if row == 0:
            cell.set_text_props(fontweight='bold', color='#1e293b')
        elif col == 0:
            cell.set_text_props(ha='left', fontweight='normal', color='#1e293b')
    for index, variant in enumerate(row_names, 1):
        best_value = max(r.curves['dice_val'][r.best] for r in grouped[variant].values())
        for j, family in enumerate(families):
            run = grouped[variant].get(family)
            if run and np.isclose(run.curves['dice_val'][run.best], best_value, rtol=0, atol=1e-8):
                cell = table[index, 2+5*j]
                cell.set_facecolor('#d5ede2'); cell.set_text_props(fontweight='bold', color='#14583c')
    for j, family in enumerate(families):
        x = bbox[0] + bbox[2]*(widths[0] + group_width*j)
        ax.add_patch(Rectangle((x, .854), bbox[2]*group_width, .066, transform=ax.transAxes,
                               color=COLORS[family], clip_on=False))
        ax.text(x + bbox[2]*group_width/2, .887, LABELS[family], transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    fig.subplots_adjust(left=.025, right=.985, top=.865, bottom=.02)
    fig.savefig(output, dpi=190, facecolor='white')
    plt.close(fig)


def save_curves(grouped, output):
    families = selected_families(grouped)
    rows = len(grouped)
    fig, axes = plt.subplots(rows, len(METRICS), figsize=(32, 2.4*rows+2.1), squeeze=False)
    titles = ('Training loss', 'Validation loss', 'Training Dice', 'Validation Dice',
              'Training IoU', 'Validation IoU', 'Training HD95', 'Validation HD95')
    max_epoch = max(r.planned for row in grouped.values() for r in row.values())
    for i, (variant, family_runs) in enumerate(grouped.items()):
        for j, (key, title) in enumerate(zip(METRICS, titles)):
            ax = axes[i, j]
            for family, run in family_runs.items():
                epochs = np.arange(1, run.completed+1)
                ax.plot(epochs, run.curves[key], color=COLORS[family], lw=1.35)
                if key.endswith('val') and not key.startswith('loss'):
                    ax.scatter(run.best+1, run.curves[key][run.best], color=COLORS[family], s=23, zorder=3,
                               edgecolors='white', linewidths=.4)
            if key.startswith('loss'):
                positive = all((r.curves[key] > 0).all() for r in family_runs.values())
                if positive:
                    ax.set_yscale('log')
                ax.set_ylabel('Loss (log scale)' if positive else 'Loss', fontsize=9)
            elif key.startswith('hd95'):
                ax.set_ylim(bottom=0); ax.set_ylabel('Foreground HD95 (pixels)', fontsize=9)
            else:
                ax.set_ylim(0, 1); ax.set_ylabel('Foreground IoU' if key.startswith('iou') else 'Foreground Dice', fontsize=9)
            ax.set_xlim(1, max_epoch); ax.set_xlabel('Epoch', fontsize=9)
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{VARIANTS.get(variant, variant)} · {title}', fontsize=10, loc='left')
    handles = [Line2D([0], [0], color=COLORS[f], lw=2, label=LABELS[f]) for f in families]
    fig.suptitle(model_label(grouped) + ' · ' + ' / '.join(families) + ' learning curves', fontsize=20, y=.986, fontweight='bold')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .968), ncol=len(families), frameon=False, fontsize=11)
    fig.subplots_adjust(left=.065, right=.98, top=.935, bottom=.074, hspace=.65, wspace=.34)
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)


def save_csv(grouped, output):
    fields = ['preprocessing', 'version', 'augmentation', 'run', 'seed', 'completed_epochs',
              'planned_epochs', 'best_epoch_1based', 'train_loss_at_best', 'val_loss_at_best',
              'train_dice_at_best', 'val_dice_at_best', 'val_esophagus_dice', 'val_heart_dice',
              'val_trachea_dice', 'val_aorta_dice',
              'train_iou_at_best', 'val_iou_at_best', 'train_hd95_at_best', 'val_hd95_at_best', 'evaluated_training_samples', 'label_source', 'mask_sha256']
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for variant, family_runs in grouped.items():
            for family in FAMILIES:
                run = family_runs.get(family)
                if run:
                    writer.writerow([variant, family, run.config['experiment'], str(run.folder), run.config.get('seed'),
                                     run.completed, run.planned, run.best+1,
                                     *[run.curves[key][run.best] for key in BASE_METRICS], *run.organ_dice,
                                     *[('' if np.isnan(run.curves[key][run.best]) else run.curves[key][run.best]) for key in EXTRA_METRICS],
                                     run.train_samples[run.best], run.schema.get('label_source', 'unknown'),
                                     run.schema['mask_sha256']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/segthor_full/enet'))
    parser.add_argument('--output-dir', type=Path, help='Default: <results>/preprocessing_comparison')
    parser.add_argument('--exclude', nargs='+', choices=['B0', 'B1', 'B2'], default=[],
                        help='Augmentations to omit from the table, curves and CSV, e.g. --exclude B1.')
    args = parser.parse_args()
    try:
        grouped = collect_runs(args.results, exclude=args.exclude)
        output = args.output_dir or args.results / 'preprocessing_comparison'
        output.mkdir(parents=True, exist_ok=True)
        save_table(grouped, output / 'summary_table.png')
        save_curves(grouped, output / 'learning_curves.png')
        save_csv(grouped, output / 'summary.csv')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Saved {sum(map(len, grouped.values()))} runs across {len(grouped)} preprocessing variants to {output}')


if __name__ == '__main__':
    main()
