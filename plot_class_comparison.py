"""Compare saved ENet and UNet validation Dice across augmentation experiments."""

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/segthor'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/enet_unet_per_class_full.png'))
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    experiments = ['not_cumulative0', 'not_cumulative1', 'not_cumulative2', 'not_cumulative3', 'not_cumulative4', 'not_cumulative5', 'not_cumulative6']
    scores = {}
    for model in ['enet', 'unet']:
        scores[model] = []
        for experiment in experiments:
            folder = args.results / model / experiment
            match = re.search(r'epoch (\d+):', (folder / 'best_epoch.txt').read_text())
            if match is None:
                raise ValueError(f'Cannot read best epoch from {folder}')
            epoch = int(match.group(1))
            dice = np.load(folder / 'dice_val.npy')
            if dice.ndim != 3 or dice.shape[2] != 5 or epoch >= len(dice):
                raise ValueError(f'Unexpected Dice shape or epoch in {folder}: {dice.shape}')
            per_class = dice[epoch].mean(axis=0)[1:]
            if not np.isfinite(per_class).all():
                raise ValueError(f'Non-finite Dice in {folder}')
            scores[model].append(per_class)
            print(f'{model.upper()} {experiment}, epoch {epoch}: {per_class.round(4)}')
        scores[model] = np.stack(scores[model])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for class_index, (axis, organ) in enumerate(zip(
            axes.flat, ['Esophagus', 'Heart', 'Trachea', 'Aorta'])):
        for model, color, marker in [('enet', '#0072B2', 'o'), ('unet', '#D55E00', 's')]:
            axis.plot(experiments, scores[model][:, class_index],
                      label=model.upper(), color=color, marker=marker, linewidth=2)
        axis.set_title(organ)
        axis.set_ylim(0, 1)
        axis.set_ylabel('Mean validation slice Dice')
        axis.tick_params(axis='x', labelbottom=True)
        axis.grid(alpha=0.25)
        axis.legend(loc='lower right')
    fig.suptitle('ENet vs UNet per organ across augmentation experiments', fontsize=16)
    fig.text(0.5, 0.055,
             'NC0: baseline   |   NC1: affine   |   NC2: contrast/gamma   |   '
             'NC3: noise   |   NC4: blur/resolution   |  NC5: bilateral denoising'
             'NC6: elastic ', ha='center', fontsize=10)
    fig.text(0.5, 0.025,
             'Each run uses its saved best foreground-Dice epoch. ',
             ha='center', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f'Saved {args.output}')
    if not args.headless:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
