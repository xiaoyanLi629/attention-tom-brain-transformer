#!/usr/bin/env python3
"""
=============================================================================
Stage 7: Publication-Quality Visualization (PDF output)
=============================================================================

Generates 6 figures + summary tables for IEEE BIBM 2026 paper.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
import seaborn as sns
import json
import logging
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})

SOCIAL_COLOR = '#E74C3C'
PHYSICAL_COLOR = '#3498DB'
MODEL_COLORS = {
    'GPT-2-Medium': '#2ECC71',
    'GPT-2-XL': '#E67E22',
    'Mistral-7B': '#9B59B6',
    'Qwen2-7B': '#E74C3C',
}
MODEL_ORDER = ['GPT-2-Medium', 'GPT-2-XL', 'Mistral-7B', 'Qwen2-7B']
ROI_NAMES = ['rTPJ', 'lTPJ', 'mPFC', 'PC', 'rSTS', 'lSTS']


def save_fig(fig, name, figures_dir):
    path = figures_dir / f"{name}.pdf"
    fig.savefig(str(path), bbox_inches='tight', format='pdf')
    logger.info(f"  Saved: {name}.pdf")
    plt.close(fig)


from contextlib import contextmanager

@contextmanager
def font_boost(boost=2):
    """Temporarily bump all rcParams font sizes by `boost` points."""
    keys = ['font.size', 'axes.titlesize', 'axes.labelsize',
            'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']
    old = {k: plt.rcParams[k] for k in keys}
    for k in keys:
        plt.rcParams[k] = old[k] + boost
    try:
        yield
    finally:
        for k in keys:
            plt.rcParams[k] = old[k]


# =============================================================================
# Probing bars + line (manuscript Fig 1)
# =============================================================================

def plot_probing_fig(figures_dir, results_dir):
    probing_path = results_dir / "probing_results.csv"
    if not probing_path.exists():
        return
    df = pd.read_csv(str(probing_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    with font_boost(2):
        fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))

        for model in models:
            sub = df[df['model'] == model].sort_values('layer')
            n_layers = sub['layer'].max() + 1
            xs = sub['layer'].values / n_layers
            ys = sub['accuracy'].values
            ax.plot(xs, ys, '-', linewidth=2.8,
                    color=MODEL_COLORS.get(model, 'gray'), label=model, alpha=0.9)
            peak_idx = int(np.argmax(ys))
            ax.scatter([xs[peak_idx]], [ys[peak_idx]], s=80, zorder=5,
                       color=MODEL_COLORS.get(model, 'gray'), edgecolor='black', linewidth=1)
        ax.axhline(y=0.5, color='gray', ls='--', lw=1.5, label='Chance (50%)')
        ax.set_xlabel('Relative Layer Position')
        ax.set_ylabel('Classification Accuracy')
        ax.set_ylim(0.45, 1.05)
        ax.legend(loc='upper right', ncol=2, frameon=True,
                  facecolor='white', edgecolor='lightgray', framealpha=0.9)

        fig.tight_layout()
        save_fig(fig, 'fig04_probing', figures_dir)


# =============================================================================
# Layer-wise encoding r (manuscript Fig 2)
# =============================================================================

def _plot_encoding_layers_body(figures_dir, results_dir):
    enc_path = results_dir / "encoding_results.csv"
    if not enc_path.exists():
        return
    df = pd.read_csv(str(enc_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panel_iter = iter([
        ('rTPJ', 'shapessocial',   'A. rTPJ: Social',   (0, 0)),
        ('rTPJ', 'shapesphysical', 'B. rTPJ: Physical', (0, 1)),
        ('mPFC', 'shapessocial',   'C. mPFC: Social',   (1, 0)),
        ('mPFC', 'shapesphysical', 'D. mPFC: Physical', (1, 1)),
    ])

    for roi, task, title, pos in panel_iter:
        ax = axes[pos]
        for model in models:
            sub = df[(df['model'] == model) & (df['task'] == task) & (df['roi'] == roi)]
            if len(sub) == 0:
                continue
            layer_mean = sub.groupby('layer')['encoding_r'].mean()
            n_layers = layer_mean.index.max() + 1
            ax.plot(layer_mean.index / n_layers, layer_mean.values,
                    '-', linewidth=2, color=MODEL_COLORS.get(model, 'gray'),
                    label=model, alpha=0.85)
        ax.axhline(y=0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Relative Layer Position')
        ax.set_ylabel('Encoding r')
        ax.set_title(title, fontweight='bold', fontsize=17, loc='left')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, 'fig02_encoding_layers', figures_dir)


def plot_encoding_layers_fig(figures_dir, results_dir):
    with font_boost(2):
        _plot_encoding_layers_body(figures_dir, results_dir)


# =============================================================================
# Fig 1: Glass Brain + Noise Ceiling
# =============================================================================

def _plot_tmap_glass_brain(tmap_path, ax=None, title=None, vmax=None, threshold=1.5):
    """Render voxel-wise t-map on a 4-view glass brain (lyrz)."""
    from nilearn import plotting
    import nibabel as nib

    t_img = nib.load(str(tmap_path))
    t_data = t_img.get_fdata()
    finite = np.isfinite(t_data)
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(t_data[finite]), 99)) if finite.any() else 5.0
    vmax = max(vmax, threshold + 0.5)

    plotting.plot_glass_brain(
        t_img,
        display_mode='lyrz',
        colorbar=True,
        cmap='RdBu_r',
        plot_abs=False,
        vmax=vmax,
        threshold=threshold,
        axes=ax,
        title=title if ax is None else None,
    )


def _plot_fig1_body(figures_dir, results_dir):
    import nibabel as nib

    # Locate whole-brain t-map produced by s01b_wholebrain_contrast
    wb_dir = config.BRAIN_ATTENTION_DIR / "wholebrain"
    tmap_masked = wb_dir / "social_vs_physical_tmap_masked.nii.gz"
    tmap_path = tmap_masked if tmap_masked.exists() else wb_dir / "social_vs_physical_tmap.nii.gz"

    # --- Standalone fig01a: glass brain of social > physical t-map ---
    if tmap_path.exists():
        fig_a = plt.figure(figsize=(12, 4.5))
        fig_a.suptitle('Social > Physical: Theory of Mind Activation Contrast',
                       fontsize=17, fontweight='bold', y=0.98)
        ax_a = fig_a.add_subplot(111)
        _plot_tmap_glass_brain(tmap_path, ax=ax_a, threshold=2.0)
        save_fig(fig_a, 'fig01a_glass_brain', figures_dir)
    else:
        logger.warning(f"  Whole-brain t-map not found: {tmap_path}")
        logger.warning("  Run scripts/s01b_wholebrain_contrast.py first; skipping fig01a")

    # --- Composite fig01: glass brain + noise ceiling + best encoding r ---
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3, height_ratios=[1.1, 1])

    ax_brain = fig.add_subplot(gs[0, :])
    ax_brain.set_title('A. Social > Physical t-map (N=58, p<.05 uncorrected)',
                       fontweight='bold', fontsize=18, loc='left')
    if tmap_path.exists():
        _plot_tmap_glass_brain(tmap_path, ax=ax_brain, threshold=2.0)
    else:
        ax_brain.axis('off')
        ax_brain.text(0.5, 0.5, 'Whole-brain t-map not available\n(run s01b_wholebrain_contrast.py)',
                      ha='center', va='center', transform=ax_brain.transAxes, color='gray')

    # Panel B: Noise ceiling
    ceiling_path = results_dir / "noise_ceiling.csv"
    ceil_df = pd.read_csv(str(ceiling_path)) if ceiling_path.exists() else None

    ax = fig.add_subplot(gs[1, 0])
    if ceil_df is not None:
        x = np.arange(len(ROI_NAMES))
        w = 0.35
        for task, color, off, label in [('shapessocial', SOCIAL_COLOR, -w/2, 'Social'),
                                         ('shapesphysical', PHYSICAL_COLOR, w/2, 'Physical')]:
            vals = []
            for roi in ROI_NAMES:
                row = ceil_df[(ceil_df['roi'] == roi) & (ceil_df['task'] == task)]
                vals.append(row['ceiling_upper'].values[0] if len(row) > 0 else 0)
            ax.bar(x + off, vals, w, color=color, alpha=0.8, label=label, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(ROI_NAMES)
        ax.set_ylabel('ISC (noise ceiling)')
        ax.set_title('B. Noise Ceiling', fontweight='bold', loc='left')
        ax.legend()

    # Panel C: Best encoding r
    ax = fig.add_subplot(gs[1, 1])
    enc_path = results_dir / "encoding_results.csv"
    if enc_path.exists():
        df = pd.read_csv(str(enc_path))
        x = np.arange(len(ROI_NAMES))
        w = 0.35
        for task, color, off, label in [('shapessocial', SOCIAL_COLOR, -w/2, 'Social'),
                                         ('shapesphysical', PHYSICAL_COLOR, w/2, 'Physical')]:
            vals = []
            for roi in ROI_NAMES:
                sub = df[(df['task'] == task) & (df['roi'] == roi)]
                vals.append(sub.groupby('layer')['encoding_r'].mean().max() if len(sub) > 0 else 0)
            ax.bar(x + off, vals, w, color=color, alpha=0.8, label=label, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(ROI_NAMES)
        ax.set_ylabel('Best encoding r')
        ax.set_title('C. Best Encoding r', fontweight='bold', loc='left')
        ax.legend()
        ax.axhline(y=0, color='gray', lw=0.5)

    save_fig(fig, 'fig01_brain_encoding', figures_dir)


def plot_fig1(figures_dir, results_dir):
    with font_boost(2):
        _plot_fig1_body(figures_dir, results_dir)


# =============================================================================
# Fig 2: Encoding Heatmap — Models × ROIs
# =============================================================================

def plot_fig2(figures_dir, results_dir):
    enc_path = results_dir / "encoding_results.csv"
    if not enc_path.exists():
        return
    df = pd.read_csv(str(enc_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    with font_boost(4):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

        for ti, (task, title_label) in enumerate([('shapessocial', 'A. Social Condition'),
                                                    ('shapesphysical', 'B. Physical Condition')]):
            ax = axes[ti]
            matrix = np.zeros((len(models), len(ROI_NAMES)))
            for mi, model in enumerate(models):
                for ri, roi in enumerate(ROI_NAMES):
                    sub = df[(df['model'] == model) & (df['task'] == task) & (df['roi'] == roi)]
                    if len(sub) > 0:
                        matrix[mi, ri] = sub.groupby('layer')['encoding_r'].mean().max()

            sns.heatmap(matrix, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, vmin=-0.02, vmax=0.04,
                        xticklabels=ROI_NAMES, yticklabels=models,
                        cbar_kws={'label': 'Encoding r', 'shrink': 0.8},
                        linewidths=0.5, linecolor='white',
                        annot_kws={'fontsize': 18})
            ax.set_title(title_label, fontweight='bold', fontsize=22, loc='left')
            ax.tick_params(axis='both', labelsize=17)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.yaxis.label.set_size(18)

        fig.tight_layout()
        save_fig(fig, 'fig02_encoding_heatmap', figures_dir)


# =============================================================================
# Fig 3: Violin plots — Subject-level encoding r distribution
# =============================================================================

def _plot_fig3_body(figures_dir, results_dir):
    enc_path = results_dir / "encoding_results.csv"
    if not enc_path.exists():
        return
    df = pd.read_csv(str(enc_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    # For each model, get the best layer for rTPJ and mPFC, then show subject distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ri, (roi, ax) in enumerate(zip(['rTPJ', 'mPFC'], axes)):
        plot_data = []
        for model in models:
            for task in ['shapessocial', 'shapesphysical']:
                sub = df[(df['model'] == model) & (df['task'] == task) & (df['roi'] == roi)]
                if len(sub) == 0:
                    continue
                best_layer = sub.groupby('layer')['encoding_r'].mean().idxmax()
                subject_rs = sub[sub['layer'] == best_layer]['encoding_r'].values
                for r in subject_rs:
                    plot_data.append({
                        'Model': model,
                        'Condition': 'Social' if task == 'shapessocial' else 'Physical',
                        'Encoding r': r
                    })

        plot_df = pd.DataFrame(plot_data)
        if len(plot_df) == 0:
            continue

        sns.violinplot(data=plot_df, x='Model', y='Encoding r', hue='Condition',
                       split=True, inner='quart', ax=ax,
                       palette={'Social': SOCIAL_COLOR, 'Physical': PHYSICAL_COLOR},
                       alpha=0.8, cut=0)
        ax.axhline(y=0, color='gray', ls='--', lw=0.8)
        ax.set_title(f'{"A" if ri == 0 else "B"}. {roi}: subject distribution (best layer)',
                     fontweight='bold', fontsize=17, loc='left')
        ax.set_xlabel('')
        ax.set_xticklabels(models, rotation=15, ha='right')
        if ax.legend_ is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, 'fig03_violin_encoding', figures_dir)


def plot_fig3(figures_dir, results_dir):
    with font_boost(3):
        _plot_fig3_body(figures_dir, results_dir)


# =============================================================================
# Fig 4: Social vs Physical Difference — t-value heatmap
# =============================================================================

def plot_fig4(figures_dir, results_dir):
    contrast_path = results_dir / "social_vs_physical.csv"
    if not contrast_path.exists():
        return
    df = pd.read_csv(str(contrast_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for mi, model in enumerate(models):
        ax = axes[mi]
        sub = df[df['model'] == model].copy()
        if len(sub) == 0:
            continue

        # Pivot: layer × ROI → t_statistic
        pivot = sub.pivot_table(values='t_statistic', index='layer', columns='roi')
        pivot = pivot.reindex(columns=ROI_NAMES)

        sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                    cbar=mi == len(models) - 1,
                    cbar_kws={'label': 't-statistic', 'shrink': 0.7},
                    yticklabels=5, linewidths=0)
        ax.set_title(model, fontweight='bold', fontsize=16)
        ax.set_xlabel('ROI')
        if mi == 0:
            ax.set_ylabel('Layer')

    fig.suptitle('', y=1.0)
    fig.tight_layout()
    save_fig(fig, 'fig04_social_vs_physical_tmap', figures_dir)


# =============================================================================
# Fig 5: Attention Heatmaps (layers × heads)
# =============================================================================

def _plot_fig5_body(figures_dir, results_dir):
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_attention"
    if not attention_dir.exists():
        return

    models = [m for m in MODEL_ORDER if (attention_dir / m).exists()]
    cmap = LinearSegmentedColormap.from_list('sp', ['#D6EAF8', '#2E86C1', '#1B4F72'])

    fig, axes = plt.subplots(len(models), 2, figsize=(14, 6.0 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for mi, model in enumerate(models):
        for ti, (task, label) in enumerate([('shapessocial', 'Social'),
                                              ('shapesphysical', 'Physical')]):
            ax = axes[mi][ti]
            csv_path = attention_dir / model / f"{task}_sparsity.csv"
            if not csv_path.exists():
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            sp_df = pd.read_csv(str(csv_path))
            heatmap = sp_df.pivot_table(values='gini', index='layer', columns='head', aggfunc='mean')

            sns.heatmap(heatmap, ax=ax, cmap=cmap, vmin=0.3, vmax=1.0,
                        cbar_kws={'label': 'Gini', 'shrink': 0.8},
                        xticklabels=5, yticklabels=5, linewidths=0)
            ax.set_xlabel('Head')
            ax.set_ylabel('Layer')
            ax.set_title(f'{model}: {label}', fontweight='bold', fontsize=16)

    fig.tight_layout()
    save_fig(fig, 'fig05_attention_heatmaps', figures_dir)


def plot_fig5(figures_dir, results_dir):
    with font_boost(3):
        _plot_fig5_body(figures_dir, results_dir)


# =============================================================================
# Fig 6: Probing + Radar Summary
# =============================================================================

def plot_fig6(figures_dir, results_dir):
    probing_path = results_dir / "probing_results.csv"
    if not probing_path.exists():
        return
    df = pd.read_csv(str(probing_path))
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[1.2, 0.8, 1])

    # Panel A: Probing accuracy by layer
    ax = fig.add_subplot(gs[0, 0])
    for model in models:
        sub = df[df['model'] == model].sort_values('layer')
        n_layers = sub['layer'].max() + 1
        ax.plot(sub['layer'] / n_layers, sub['accuracy'], '-o', markersize=3,
                color=MODEL_COLORS.get(model, 'gray'), linewidth=2.5, label=model, alpha=0.85)
    ax.axhline(y=0.5, color='gray', ls='--', lw=1.5)
    ax.set_xlabel('Relative Layer Position')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('A. Probing by Layer', fontweight='bold', loc='left')
    ax.legend(fontsize=12)
    ax.set_ylim(0.4, 1.0)

    # Panel B: Peak accuracy bars
    ax = fig.add_subplot(gs[0, 1])
    best_accs = []
    best_layers = []
    colors = []
    for m in models:
        sub = df[df['model'] == m]
        best_accs.append(sub['accuracy'].max())
        best_layers.append(int(sub.loc[sub['accuracy'].idxmax(), 'layer']))
        colors.append(MODEL_COLORS.get(m, 'gray'))
    ax.bar(range(len(models)), best_accs, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    ax.axhline(y=0.5, color='gray', ls='--')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('-', '-\n') for m in models], fontsize=12)
    ax.set_ylabel('Peak Accuracy')
    ax.set_title('B. Peak Accuracy', fontweight='bold', loc='left')
    ax.set_ylim(0.4, 1.0)
    for i, (acc, layer) in enumerate(zip(best_accs, best_layers)):
        ax.text(i, acc + 0.015, f'L{layer}\n{acc:.1%}', ha='center', fontsize=13, fontweight='bold')

    # Panel C: Radar chart — multi-metric comparison
    ax = fig.add_subplot(gs[0, 2], polar=True)

    # Metrics: probing acc, mean encoding r (social rTPJ), sparsity (social gini), encoding r (mPFC)
    enc_path = results_dir / "encoding_results.csv"
    spec_path = results_dir / "head_specialization_summary.csv"

    categories = ['Probing\nAccuracy', 'Encoding r\n(rTPJ)', 'Encoding r\n(mPFC)',
                   'Attention\nSparsity', 'Sparse\nHead %']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    if enc_path.exists() and spec_path.exists():
        enc_df = pd.read_csv(str(enc_path))
        spec_df = pd.read_csv(str(spec_path))

        for model in models:
            values = []
            # Probing acc (normalize to 0-1)
            sub = df[df['model'] == model]
            values.append(sub['accuracy'].max())

            # Encoding r rTPJ (normalize: map [-0.01, 0.04] to [0, 1])
            esub = enc_df[(enc_df['model'] == model) & (enc_df['task'] == 'shapessocial') & (enc_df['roi'] == 'rTPJ')]
            best_r = esub.groupby('layer')['encoding_r'].mean().max() if len(esub) > 0 else 0
            values.append(max(0, best_r) * 10)  # scale up for visibility

            # Encoding r mPFC
            esub = enc_df[(enc_df['model'] == model) & (enc_df['task'] == 'shapessocial') & (enc_df['roi'] == 'mPFC')]
            best_r = esub.groupby('layer')['encoding_r'].mean().max() if len(esub) > 0 else 0
            values.append(max(0, best_r) * 10)

            # Sparsity
            srow = spec_df[(spec_df['model'] == model) & (spec_df['task'] == 'shapessocial')]
            values.append(srow['mean_gini'].values[0] if len(srow) > 0 else 0)

            # Sparse head %
            values.append(srow['pct_sparse_gini'].values[0] if len(srow) > 0 else 0)

            values += [values[0]]  # close the polygon
            ax.plot(angles, values, 'o-', linewidth=2,
                    color=MODEL_COLORS.get(model, 'gray'), label=model, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS.get(model, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_title('C. Multi-Metric Comparison', fontweight='bold', fontsize=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    save_fig(fig, 'fig06_probing_radar', figures_dir)


# =============================================================================
# Summary Tables (CSV for LaTeX)
# =============================================================================

def generate_tables(figures_dir, results_dir):
    """Generate summary tables as CSV files."""

    # Table 1: Core results summary
    enc_path = results_dir / "encoding_results.csv"
    probing_path = results_dir / "probing_results.csv"
    spec_path = results_dir / "head_specialization_summary.csv"

    if not all(p.exists() for p in [enc_path, probing_path, spec_path]):
        return

    enc_df = pd.read_csv(str(enc_path))
    prob_df = pd.read_csv(str(probing_path))
    spec_df = pd.read_csv(str(spec_path))

    rows = []
    for model in MODEL_ORDER:
        if model not in enc_df['model'].unique():
            continue
        row = {'Model': model}

        # Probing
        psub = prob_df[prob_df['model'] == model]
        row['Probing Acc (%)'] = f"{psub['accuracy'].max()*100:.1f}"
        row['Best Probing Layer'] = int(psub.loc[psub['accuracy'].idxmax(), 'layer'])

        # Encoding r (social, rTPJ and mPFC)
        for roi in ['rTPJ', 'mPFC']:
            esub = enc_df[(enc_df['model'] == model) & (enc_df['task'] == 'shapessocial') & (enc_df['roi'] == roi)]
            if len(esub) > 0:
                layer_means = esub.groupby('layer')['encoding_r'].mean()
                row[f'Enc r ({roi})'] = f"{layer_means.max():.3f}"
                row[f'Best Enc Layer ({roi})'] = int(layer_means.idxmax())

        # Sparsity
        srow = spec_df[(spec_df['model'] == model) & (spec_df['task'] == 'shapessocial')]
        if len(srow) > 0:
            row['Mean Gini (Social)'] = f"{srow['mean_gini'].values[0]:.3f}"
            row['Sparse Heads (%)'] = f"{srow['pct_sparse_gini'].values[0]*100:.1f}"

        rows.append(row)

    table1 = pd.DataFrame(rows)
    table1.to_csv(str(figures_dir / 'table_summary.csv'), index=False)
    logger.info(f"  Table 1 (summary): {len(table1)} rows")
    print("\n=== Table 1: Core Results Summary ===")
    print(table1.to_string(index=False))

    # Table 2: Social vs Physical contrast for best layers
    contrast_path = results_dir / "social_vs_physical.csv"
    if contrast_path.exists():
        con_df = pd.read_csv(str(contrast_path))
        rows2 = []
        for model in MODEL_ORDER:
            if model not in con_df['model'].unique():
                continue
            for roi in ['rTPJ', 'lTPJ', 'mPFC']:
                sub = con_df[(con_df['model'] == model) & (con_df['roi'] == roi)]
                if len(sub) == 0:
                    continue
                # Best layer (highest social_mean_r)
                best = sub.loc[sub['social_mean_r'].idxmax()]
                rows2.append({
                    'Model': model,
                    'ROI': roi,
                    'Layer': int(best['layer']),
                    'Social r': f"{best['social_mean_r']:.4f}",
                    'Physical r': f"{best['physical_mean_r']:.4f}",
                    'Diff': f"{best['difference']:.4f}",
                    't': f"{best['t_statistic']:.2f}",
                    'p': f"{best['p_value']:.3f}",
                })

        table2 = pd.DataFrame(rows2)
        table2.to_csv(str(figures_dir / 'table_contrast.csv'), index=False)
        logger.info(f"  Table 2 (contrast): {len(table2)} rows")
        print("\n=== Table 2: Social vs Physical Contrast (best layers) ===")
        print(table2.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def run_stage7():
    logger.info("=" * 70)
    logger.info("STAGE 7: Publication-Quality Visualization")
    logger.info("=" * 70)

    config.ensure_run_directories()
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = config.CROSS_DOMAIN_DIR

    plot_fig1(figures_dir, results_dir)
    plot_fig2(figures_dir, results_dir)
    plot_fig3(figures_dir, results_dir)
    plot_fig4(figures_dir, results_dir)
    plot_fig5(figures_dir, results_dir)
    plot_fig6(figures_dir, results_dir)
    plot_probing_fig(figures_dir, results_dir)
    plot_encoding_layers_fig(figures_dir, results_dir)
    generate_tables(figures_dir, results_dir)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 7 Complete")
    for f in sorted(figures_dir.glob("*.pdf")):
        logger.info(f"    {f.name}")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_stage7()
