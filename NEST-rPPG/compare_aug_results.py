#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Compare baseline vs augmentation CV results across all three datasets.

Usage:
    python compare_aug_results.py                    # baseline vs aug (default)
    python compare_aug_results.py --tag aug          # compare against a specific run tag
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DATASETS = ['PURE_my_in', 'UBFC_my_in', 'BUAA_my_in']
METRICS  = ['MAE', 'RMSE', 'r', 'ME', 'Std']


def load_results(dataset, weight_cl, tag=''):
    tag_str = f'_{tag}' if tag else ''
    path = os.path.join(
        config.RESULT_LOG_DIR,
        f'intratest_cv_{dataset}_wl{weight_cl}{tag_str}_results.csv'
    )
    if not os.path.isfile(path):
        return None, path
    df = pd.read_csv(path)
    df['fold_str'] = df['fold'].astype(str)
    return df, path


def get_summary_row(df, label):
    row = df[df['fold_str'] == label]
    return row.iloc[0] if not row.empty else None


def get_fold_rows(df):
    mask = df['fold_str'].apply(lambda x: x.replace('.0', '').isdigit())
    return df[mask].copy()


def print_comparison(dataset, baseline_df, aug_df, tag):
    print(f"\n{'='*70}")
    print(f"  {dataset}")
    print(f"{'='*70}")
    print(f"  {'':6}  {'Baseline':>30}  {'Aug (' + tag + ')':>30}")
    print(f"  {'fold':6}  {'MAE':>6} {'RMSE':>6} {'r':>6} {'ME':>6}  "
          f"{'MAE':>6} {'RMSE':>6} {'r':>6} {'ME':>6}  "
          f"{'ΔMAE':>7}")
    print(f"  {'-'*68}")

    b_folds = get_fold_rows(baseline_df).sort_values('fold_str')
    a_folds = get_fold_rows(aug_df).sort_values('fold_str')

    for (_, br), (_, ar) in zip(b_folds.iterrows(), a_folds.iterrows()):
        delta = ar['MAE'] - br['MAE']
        arrow = '▼' if delta < -0.5 else ('▲' if delta > 0.5 else '~')
        print(f"  {str(br['fold_str']).replace('.0',''):6}  "
              f"{br['MAE']:6.2f} {br['RMSE']:6.2f} {br['r']:6.3f} {br['ME']:6.2f}  "
              f"{ar['MAE']:6.2f} {ar['RMSE']:6.2f} {ar['r']:6.3f} {ar['ME']:6.2f}  "
              f"{delta:+6.2f} {arrow}")

    print(f"  {'-'*68}")
    for label in ['mean', 'fold_std']:
        br = get_summary_row(baseline_df, label)
        ar = get_summary_row(aug_df, label)
        if br is None or ar is None:
            continue
        delta = ar['MAE'] - br['MAE']
        tag_str = 'mean    ' if label == 'mean' else 'fold_std'
        print(f"  {tag_str:6}  "
              f"{br['MAE']:6.2f} {br['RMSE']:6.2f} {br['r']:6.3f} {br['ME']:6.2f}  "
              f"{ar['MAE']:6.2f} {ar['RMSE']:6.2f} {ar['r']:6.3f} {ar['ME']:6.2f}  "
              f"{delta:+6.2f}")


def make_comparison_plot(results, tag, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (dataset, baseline_df, aug_df) in zip(axes, results):
        b_folds = get_fold_rows(baseline_df).sort_values('fold_str')
        a_folds = get_fold_rows(aug_df).sort_values('fold_str')

        folds = [str(f).replace('.0', '') for f in b_folds['fold_str']]
        x = np.arange(len(folds))
        w = 0.35

        b_mae = b_folds['MAE'].values
        a_mae = a_folds['MAE'].values

        bars_b = ax.bar(x - w/2, b_mae, w, label='Baseline', color='steelblue', alpha=0.8)
        bars_a = ax.bar(x + w/2, a_mae, w, label=f'Aug ({tag})', color='coral', alpha=0.8)

        b_mean = get_summary_row(baseline_df, 'mean')
        a_mean = get_summary_row(aug_df, 'mean')
        if b_mean is not None:
            ax.axhline(b_mean['MAE'], color='steelblue', ls='--', lw=1.2, alpha=0.7,
                       label=f'Baseline mean={b_mean["MAE"]:.2f}')
        if a_mean is not None:
            ax.axhline(a_mean['MAE'], color='coral', ls='--', lw=1.2, alpha=0.7,
                       label=f'Aug mean={a_mean["MAE"]:.2f}')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.set_ylabel('MAE (BPM)')
        ax.set_title(dataset.replace('_my_in', ''))
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Intratest CV: Baseline vs Augmentation ({tag}) — MAE per fold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='aug',
                        help='Run tag used in the aug CSV filename (default: aug)')
    parser.add_argument('--weight_cl', type=float, default=0.0)
    parser.add_argument('--datasets', type=str, default='',
                        help='Comma-separated list of datasets to compare '
                             '(default: PURE_my_in,UBFC_my_in,BUAA_my_in)')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')] if args.datasets else DATASETS
    print(datasets)

    results = []
    for dataset in datasets:
        baseline_df, b_path = load_results(dataset, args.weight_cl, tag='')
        aug_df,      a_path = load_results(dataset, args.weight_cl, tag=args.tag)

        if baseline_df is None:
            print(f"[SKIP] Baseline not found: {b_path}")
            continue
        if aug_df is None:
            print(f"[SKIP] Aug results not found: {a_path}")
            continue

        results.append((dataset, baseline_df, aug_df))
        print_comparison(dataset, baseline_df, aug_df, args.tag)

    if results:
        suffix = '_' + args.datasets.replace(',', '_') if args.datasets else ''
        out_path = os.path.join(config.RESULT_LOG_DIR,
                                f'comparison_baseline_vs_{args.tag}{suffix}.png')
        make_comparison_plot(results, args.tag, out_path)


if __name__ == '__main__':
    main()
