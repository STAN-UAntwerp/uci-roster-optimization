"""Manuscript workload figures from public team-level aggregates only.

Run from any directory: python figures/workload_analysis.py
No source-level inputs, model fitting or Gurobi license are needed.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['original', 'ranker', 'regressor']
LABELS = ['Original', 'Ranker', 'Regressor']
# Match the original/optimized blue and purple, and Figure 9's orange.
COLORS = ['#4EA8DE', '#59267C', '#C85200']


def distribution_panel(ax, frame, metric, ylabel, multiplier=1, compact=False):
    values = frame.pivot(index=['year', 'team'], columns='allocation', values=metric)[ORDER]
    assert values.shape == (36, 3) and values.notna().all().all()
    values = values * multiplier
    boxes = ax.boxplot([values[k] for k in ORDER], positions=[1, 2, 3], widths=.48,
                       patch_artist=True, showfliers=False, whis=(0, 100),
                       boxprops={'edgecolor': 'black', 'linewidth': .8},
                       whiskerprops={'color': 'black', 'linewidth': .8},
                       capprops={'color': 'black', 'linewidth': .8},
                       medianprops={'color': 'black', 'linewidth': 1.2})
    for patch, color in zip(boxes['boxes'], COLORS):
        patch.set_facecolor(color)
    labels = ['Orig.', 'Rank.', 'Regr.'] if compact else LABELS
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)
    ax.set_xlim(.5, 3.5)


def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif',
                         'font.serif': ['Computer Modern'], 'font.size': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10})
    out = ROOT / 'reproduced'
    out.mkdir(exist_ok=True)
    workload = pd.read_csv(ROOT / 'results/workload_team_summary.csv')
    points = pd.read_csv(ROOT / 'results/concentration_team_summary.csv')
    workload.drop(columns=['year']).groupby('allocation').mean(numeric_only=True).to_csv(
        out / 'workload_overall_summary.csv')
    points.drop(columns=['year']).groupby(['scoring', 'evaluator', 'allocation']).mean(numeric_only=True).to_csv(
        out / 'concentration_overall_summary.csv')
    points = points[(points.evaluator == 'ranker') & (points.scoring == 'predicted')]
    fig, axes = plt.subplots(1, 2, figsize=(5.15, 3.27))
    distribution_panel(axes[0], workload, 'mean_days', 'Mean assigned race days')
    distribution_panel(axes[1], workload, 'share_at_cap', r'Riders with at least 85 days (\%)', 100)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 100)
    fig.tight_layout(w_pad=1)
    fig.savefig(out / 'rider_workload_season.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(5.15, 3.27))
    for ax, w in zip(axes, (30, 60, 90)):
        distribution_panel(ax, workload, f'max_peak_{w}', 'Peak racing days', compact=True)
        ax.set_title(f'{w}-day window', fontsize=10)
        ax.set_ylim(0, w)
    fig.tight_layout(w_pad=.6)
    fig.savefig(out / 'rider_workload_peaks.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(5.15, 3.27))
    distribution_panel(axes[0], points, 'gini_points', 'Gini of predicted points')
    distribution_panel(axes[1], points, 'top5_share', r'Top-five share of predicted points (\%)', 100)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 100)
    fig.tight_layout(w_pad=1)
    fig.savefig(out / 'rider_points_concentration.pdf')
    plt.close(fig)
    print('Saved three workload/concentration PDFs in', out)


if __name__ == '__main__':
    main()
