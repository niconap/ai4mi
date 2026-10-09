
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
METRICS = ('loss_tra', 'loss_val', 'dice_tra', 'dice_val')


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
    return {'enet': 'ENet', 'resunetpp': 'ResUNet++'}[model]


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
    best, organs, schema = read_scores(folder)
    plan = json.loads((folder / 'preprocessing.json').read_text())
    arrays = {key: np.load(folder / f'{key}.npy', allow_pickle=False) for key in METRICS}
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
    if not runs:
        raise ValueError('No runs remain after excluding the selected augmentations.')
    if len({r.config['model'] for r in runs}) != 1:
        raise ValueError('Runs have different models; compare one model directory at a time.')
    if len({r.schema['mask_sha256'] for r in runs}) != 1:
        raise ValueError('Runs have different ground-truth masks or patient splits.')
    grouped = {}
    for run in runs:
        cell = grouped.setdefault(run.variant, {})
        if run.family in cell:
            raise ValueError(f'Duplicate {run.family}/{run.variant} runs.')
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


def comparison_notes(grouped):
    runs = [r for row in grouped.values() for r in row.values()]
    incomplete = [f'{r.folder.name}: {r.completed}/{r.planned} epochs' for r in runs if r.completed < r.planned]
    notes = ['Loss is reported at the saved best-Dice checkpoint']
    if incomplete:
        notes.append('* Fewer completed epochs: ' + '; '.join(incomplete) + '.')
    seeds = sorted({str(r.config.get('seed')) for r in runs})
    label_source = 'Existing pseudo-label ground truth' if any(r.schema.get('label_source') == 'pseudo' for r in runs) else 'Separate-organ ground truth'
    notes.append(label_source + '; seed(s): ' + ', '.join(seeds) + '. Epoch numbers are 1-based.')
    return notes


def save_table(grouped, output):
    families = selected_families(grouped)
    rows, row_names = [], list(grouped)
    for variant in row_names:
        row = [VARIANTS.get(variant, variant)]
        for family in families:
            run = grouped[variant].get(family)
            if run is None:
                row += ['—', '—', '—']
            else:
                marker = '*' if run.completed < run.planned else ''
                row += [f'{run.best + 1}{marker}', f'{run.curves["dice_val"][run.best]:.4f}',
                        f'{run.curves["loss_val"][run.best]:.5f}']
        rows.append(row)
    fig, ax = plt.subplots(figsize=(16, max(7, 3.5 + .43*len(rows))))
    fig.patch.set_facecolor('white'); ax.axis('off')
    fig.suptitle(model_label(grouped) + ' · preprocessing and augmentation comparison', fontsize=21, fontweight='bold', y=.965)
    group_width = .78 / len(families)
    widths = [.22] + [group_width*x/.26 for x in (.075, .095, .09)]*len(families)
    bbox = [.015, .24, .97, .60]
    table = ax.table(cellText=rows, colLabels=['Preprocessing'] + ['Epoch', 'Val Dice ↑', 'Val loss ↓']*len(families),
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
                cell = table[index, 2+3*j]
                cell.set_facecolor('#d5ede2'); cell.set_text_props(fontweight='bold', color='#14583c')
    for j, family in enumerate(families):
        x = bbox[0] + bbox[2]*(widths[0] + group_width*j)
        ax.add_patch(Rectangle((x, .854), bbox[2]*group_width, .066, transform=ax.transAxes,
                               color=COLORS[family], clip_on=False))
        ax.text(x + bbox[2]*group_width/2, .887, LABELS[family], transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(.018, .195, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=10,
            color='#475569', linespacing=1.5)
    fig.subplots_adjust(left=.025, right=.985, top=.865, bottom=.02)
    fig.savefig(output, dpi=190, facecolor='white')
    plt.close(fig)


def save_curves(grouped, output):
    families = selected_families(grouped)
    rows = len(grouped)
    fig, axes = plt.subplots(rows, 4, figsize=(17, 2.4*rows+2.1), squeeze=False)
    titles = ('Training loss', 'Validation loss', 'Training Dice', 'Validation Dice')
    max_epoch = max(r.planned for row in grouped.values() for r in row.values())
    for i, (variant, family_runs) in enumerate(grouped.items()):
        for j, (key, title) in enumerate(zip(METRICS, titles)):
            ax = axes[i, j]
            for family, run in family_runs.items():
                epochs = np.arange(1, run.completed+1)
                ax.plot(epochs, run.curves[key], color=COLORS[family], lw=1.35)
                if key == 'dice_val':
                    ax.scatter(run.best+1, run.curves[key][run.best], color=COLORS[family], s=23, zorder=3,
                               edgecolors='white', linewidths=.4)
            if key.startswith('loss'):
                positive = all((r.curves[key] > 0).all() for r in family_runs.values())
                if positive:
                    ax.set_yscale('log')
                ax.set_ylabel('Loss (log scale)' if positive else 'Loss', fontsize=9)
            else:
                ax.set_ylim(0, 1); ax.set_ylabel('Foreground Dice', fontsize=9)
            ax.set_xlim(1, max_epoch); ax.set_xlabel('Epoch', fontsize=9)
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{VARIANTS.get(variant, variant)} · {title}', fontsize=10, loc='left')
    handles = [Line2D([0], [0], color=COLORS[f], lw=2, label=LABELS[f]) for f in families]
    fig.suptitle(model_label(grouped) + ' · ' + ' / '.join(families) + ' learning curves', fontsize=20, y=.986, fontweight='bold')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .968), ncol=len(families), frameon=False, fontsize=11)
    foot = ['Validation dots mark saved best checkpoints. No smoothing; unwritten epochs are omitted.',
            'Training Dice excludes unused drop_last rows and reflects sampled/augmented training inputs.']
    foot += comparison_notes(grouped)[2:]
    fig.text(.07, .009, '\n'.join(foot), fontsize=9, va='bottom', color='#475569', linespacing=1.35)
    fig.subplots_adjust(left=.065, right=.98, top=.935, bottom=.074, hspace=.65, wspace=.34)
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)


def save_csv(grouped, output):
    fields = ['preprocessing', 'version', 'augmentation', 'run', 'seed', 'completed_epochs',
              'planned_epochs', 'best_epoch_1based', 'train_loss_at_best', 'val_loss_at_best',
              'train_dice_at_best', 'val_dice_at_best', 'val_esophagus_dice', 'val_heart_dice',
              'val_trachea_dice', 'val_aorta_dice', 'evaluated_training_samples', 'label_source', 'mask_sha256']
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for variant, family_runs in grouped.items():
            for family in FAMILIES:
                run = family_runs.get(family)
                if run:
                    writer.writerow([variant, family, run.config['experiment'], str(run.folder), run.config.get('seed'),
                                     run.completed, run.planned, run.best+1,
                                     *[run.curves[key][run.best] for key in METRICS], *run.organ_dice,
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
        (output / 'README.txt').write_text('Included versions: ' + ', '.join(selected_families(grouped)) + '\n' +
            '\n'.join(comparison_notes(grouped)) +
            '\nCSV metrics are all taken at each saved best-Dice checkpoint.\n'
            'Learning curves include training and validation; training Dice excludes unused zero-filled rows.\n')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Saved {sum(map(len, grouped.values()))} runs across {len(grouped)} preprocessing variants to {output}')
    for note in comparison_notes(grouped):
        print(note)


if __name__ == '__main__':
    main()
