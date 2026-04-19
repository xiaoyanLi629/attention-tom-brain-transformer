#!/usr/bin/env python3
"""
Patch existing result CSVs without re-running expensive pipelines:
  1. social_vs_physical.csv  — add BH-FDR columns
  2. probing_results.csv     — compute permutation p-value for each model's best layer
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from configs import config
from scripts.s06_probing import probe_layer, PCA_COMPONENTS, N_PERMUTATIONS
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(
        (p[order] * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    return np.clip(adjusted_sorted[ranks - 1], 0, 1)


def patch_encoding_fdr(cross_domain_dir):
    csv_path = cross_domain_dir / "social_vs_physical.csv"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}")
        return
    df = pd.read_csv(str(csv_path))
    df['p_fdr'] = bh_fdr(df['p_value'].values)
    df['significant_fdr'] = df['p_fdr'] < 0.05
    df.to_csv(str(csv_path), index=False)

    n_uncorr = int((df['p_value'] < 0.05).sum())
    n_fdr = int(df['significant_fdr'].sum())
    logger.info(f"social_vs_physical: {n_uncorr}/{len(df)} uncorrected, "
                f"{n_fdr}/{len(df)} after FDR")

    # Update summary json
    summary_path = cross_domain_dir / "encoding_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary.setdefault('social_vs_physical', {})
        summary['social_vs_physical']['significant_contrasts'] = n_uncorr
        summary['social_vs_physical']['significant_contrasts_fdr05'] = n_fdr
        summary['social_vs_physical']['total_contrasts'] = len(df)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Updated encoding_summary.json")


def _perm_iter(X, y, seed):
    rng = np.random.default_rng(seed)
    return probe_layer(X, rng.permutation(y))


def patch_probing_pvalues(cross_domain_dir, features_dir, n_jobs=8):
    csv_path = cross_domain_dir / "probing_results.csv"
    summary_path = cross_domain_dir / "probing_summary.json"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}")
        return
    df = pd.read_csv(str(csv_path))

    for model_name in df['model'].unique():
        mask = df['model'] == model_name
        model_data = df[mask]
        best_idx = model_data['accuracy'].idxmax()
        if not pd.isna(df.loc[best_idx, 'p_value']):
            logger.info(f"  {model_name}: p-value already present, skipping")
            continue

        best_layer = int(df.loc[best_idx, 'layer'])
        social_path = features_dir / model_name / f"shapessocial_layer{best_layer}.npy"
        physical_path = features_dir / model_name / f"shapesphysical_layer{best_layer}.npy"
        if not (social_path.exists() and physical_path.exists()):
            logger.warning(f"  {model_name}: features missing, cannot compute")
            continue

        social = np.load(str(social_path))
        physical = np.load(str(physical_path))
        n = min(social.shape[0], physical.shape[0])
        X = np.vstack([social[:n], physical[:n]])
        y = np.array([1] * n + [0] * n)

        bad = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
        X, y = X[~bad], y[~bad]

        n_comp = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        X_pca = PCA(n_components=n_comp).fit_transform(X)

        real_acc = probe_layer(X_pca, y)
        seeds = [int(s.generate_state(1)[0])
                 for s in np.random.SeedSequence(42).spawn(N_PERMUTATIONS)]

        if n_jobs == 1:
            null_accs = [_perm_iter(X_pca, y, s) for s in seeds]
        else:
            null_accs = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_perm_iter)(X_pca, y, s) for s in seeds
            )
        null_accs = np.array(null_accs)
        p_val = float(np.mean(null_accs >= real_acc))

        df.loc[best_idx, 'p_value'] = p_val
        logger.info(f"  {model_name}: layer={best_layer}, "
                    f"acc={real_acc:.3f}, p={p_val:.4f}")

    df.to_csv(str(csv_path), index=False)

    # Rewrite summary
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for model_name, meta in summary.items():
            model_data = df[df['model'] == model_name]
            best_idx = model_data['accuracy'].idxmax()
            p = df.loc[best_idx, 'p_value']
            summary[model_name]['best_p_value'] = float(p) if not pd.isna(p) else None
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info("Updated probing_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-jobs', type=int, default=8)
    args = parser.parse_args()

    config.ensure_run_directories()
    cross_domain = config.CROSS_DOMAIN_DIR
    features = config.TRANSFORMER_ATTENTION_DIR / "llm_features"

    logger.info("=== Patch encoding FDR ===")
    patch_encoding_fdr(cross_domain)

    logger.info("\n=== Patch probing permutation p-values ===")
    patch_probing_pvalues(cross_domain, features, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
