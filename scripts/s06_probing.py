#!/usr/bin/env python3
"""
=============================================================================
Stage 6: Probing Analysis — Social vs Physical Classification
=============================================================================

Train linear probes on LLM hidden states at each layer to classify whether
the model is processing social or physical narrative content.

Key fixes over original s04:
  - ~540 samples instead of 7 (temporal block CV)
  - No silent `except → return 0.5`
  - PCA dimensionality reduction before classification
  - Permutation test for significance

Outputs:
  - probing_results.csv      — accuracy + p-value per model × layer
  - probing_summary.json     — best layers, peak accuracy
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from joblib import Parallel, delayed
import json
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PCA_COMPONENTS = 50
N_FOLDS = 5
N_PERMUTATIONS = 500


def temporal_block_cv(n_samples, n_folds=N_FOLDS):
    """Temporal block CV — same as Stage 3."""
    fold_size = n_samples // n_folds
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start),
                                     np.arange(test_end, n_samples)])
        yield train_idx, test_idx


def probe_layer(X, y, n_folds=N_FOLDS):
    """Train a linear probe with temporal block CV.

    Args:
        X: (n_samples, n_features) — concatenated social + physical features
        y: (n_samples,) — binary labels (1=social, 0=physical)

    Returns:
        accuracy: float
        accuracies_per_fold: list
    """
    predictions = np.zeros(len(y))

    for train_idx, test_idx in temporal_block_cv(len(y), n_folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
        ])
        clf.fit(X_train, y_train)
        predictions[test_idx] = clf.predict(X_test)

    accuracy = float((predictions == y).mean())
    return accuracy


def _perm_iteration(X, y, seed):
    rng = np.random.default_rng(seed)
    return probe_layer(X, rng.permutation(y))


def permutation_test(X, y, n_permutations=N_PERMUTATIONS, n_jobs=1):
    """Compute p-value by permuting labels; optionally parallelized."""
    real_acc = probe_layer(X, y)
    seeds = np.random.SeedSequence().spawn(n_permutations)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]
    if n_jobs == 1:
        null_accs = [_perm_iteration(X, y, s) for s in seeds]
    else:
        null_accs = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_perm_iteration)(X, y, s) for s in seeds
        )
    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= real_acc))
    return real_acc, p_value, null_accs


def run_stage6(n_jobs=1):
    """Run Stage 6: Probing analysis."""
    logger.info("=" * 70)
    logger.info("STAGE 6: Probing Analysis (Social vs Physical)")
    logger.info("=" * 70)

    config.ensure_run_directories()
    features_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_features"
    output_dir = config.CROSS_DOMAIN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_dir.exists():
        logger.error("No LLM features found! Run Stage 2 first.")
        return

    models = [d.name for d in features_dir.iterdir() if d.is_dir()]
    logger.info(f"Models: {models}")

    all_results = []

    for model_name in models:
        model_dir = features_dir / model_name
        logger.info(f"\n{'='*50}")
        logger.info(f"Probing: {model_name}")

        # Find layers
        social_layers = sorted(model_dir.glob("shapessocial_layer*.npy"),
                                key=lambda p: int(p.stem.split('layer')[1]))
        physical_layers = sorted(model_dir.glob("shapesphysical_layer*.npy"),
                                  key=lambda p: int(p.stem.split('layer')[1]))

        if not social_layers or not physical_layers:
            logger.warning(f"  Missing features, skipping")
            continue

        n_layers = min(len(social_layers), len(physical_layers))

        for li in tqdm(range(n_layers), desc=f"  Layers"):
            social_feat = np.load(str(social_layers[li]))   # (n_trs_s, hidden_dim)
            physical_feat = np.load(str(physical_layers[li]))  # (n_trs_p, hidden_dim)

            # Use same number of TRs
            min_trs = min(social_feat.shape[0], physical_feat.shape[0])
            social_feat = social_feat[:min_trs]
            physical_feat = physical_feat[:min_trs]

            # Concatenate: social first, then physical (preserves temporal order within each)
            X = np.vstack([social_feat, physical_feat])
            y = np.array([1] * min_trs + [0] * min_trs)

            # Handle NaN/Inf values
            nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
            if nan_mask.any():
                X = X[~nan_mask]
                y = y[~nan_mask]
                if len(y) < 20:
                    logger.warning(f"  Layer {li}: too few valid samples ({len(y)}), skipping")
                    continue

            # PCA
            n_comp = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X)

            # Probe with permutation test (only for every 4th layer to save time)
            if li % 4 == 0 or li == n_layers - 1:
                acc, p_val, _ = permutation_test(X_pca, y, n_permutations=N_PERMUTATIONS, n_jobs=n_jobs)
            else:
                acc = probe_layer(X_pca, y)
                p_val = np.nan  # skip permutation for speed

            all_results.append({
                'model': model_name,
                'layer': int(social_layers[li].stem.split('layer')[1]),
                'accuracy': acc,
                'p_value': p_val,
                'n_samples': len(y),
                'n_social': min_trs,
                'n_physical': min_trs,
                'pca_components': n_comp,
                'above_chance': acc > 0.5,
            })

    results_df = pd.DataFrame(all_results)

    # Ensure best layer has a permutation p-value for every model
    logger.info("\nComputing permutation p-values for best layer of each model...")
    for model_name in results_df['model'].unique():
        model_mask = results_df['model'] == model_name
        model_data = results_df[model_mask]
        best_idx = model_data['accuracy'].idxmax()
        if not np.isnan(results_df.loc[best_idx, 'p_value']):
            continue

        best_row = results_df.loc[best_idx]
        best_layer = int(best_row['layer'])
        model_dir = features_dir / model_name
        social_path = model_dir / f"shapessocial_layer{best_layer}.npy"
        physical_path = model_dir / f"shapesphysical_layer{best_layer}.npy"
        if not (social_path.exists() and physical_path.exists()):
            logger.warning(f"  {model_name}: best-layer features missing, cannot compute p-value")
            continue

        social_feat = np.load(str(social_path))
        physical_feat = np.load(str(physical_path))
        min_trs = min(social_feat.shape[0], physical_feat.shape[0])
        X = np.vstack([social_feat[:min_trs], physical_feat[:min_trs]])
        y = np.array([1] * min_trs + [0] * min_trs)

        nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
        if nan_mask.any():
            X = X[~nan_mask]
            y = y[~nan_mask]

        n_comp = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        _, p_val, _ = permutation_test(X_pca, y, n_permutations=N_PERMUTATIONS, n_jobs=n_jobs)
        results_df.loc[best_idx, 'p_value'] = p_val
        logger.info(f"  {model_name}: layer {best_layer} p = {p_val:.4f}")

    results_df.to_csv(str(output_dir / "probing_results.csv"), index=False)

    # Summary
    summary = {}
    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name]
        best_idx = model_data['accuracy'].idxmax()
        best = model_data.loc[best_idx]
        summary[model_name] = {
            'best_layer': int(best['layer']),
            'best_accuracy': float(best['accuracy']),
            'best_p_value': float(best['p_value']) if not np.isnan(best['p_value']) else None,
            'n_layers': int(model_data['layer'].nunique()),
            'mean_accuracy': float(model_data['accuracy'].mean()),
        }

    with open(str(output_dir / "probing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 6 Complete")
    for model_name, data in summary.items():
        logger.info(f"  {model_name}: best_layer={data['best_layer']}, "
                    f"acc={data['best_accuracy']:.3f}")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Parallel workers for permutation test (default: 8; CPU-light, ~8-16 safe on autodl)')
    args = parser.parse_args()
    run_stage6(n_jobs=args.n_jobs)
