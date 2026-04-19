#!/usr/bin/env python3
"""
=============================================================================
Stage 3: Encoding Models — Core Analysis
=============================================================================

Ridge regression predicting each ROI's fMRI time series from LLM layer
representations. This is the central analysis of the paper.

Key analyses:
  1. For each {model, layer, ROI, condition, subject}:
     predict brain time series from LLM features using ridge regression
  2. Social vs Physical contrast: paired t-test on encoding r (N=59)
  3. ISC-based noise ceiling
  4. Layer selectivity: which layers best encode each ROI?

Outputs:
  - encoding_results.csv         — full results table
  - noise_ceiling.csv            — ISC-based ceiling per ROI × condition
  - social_vs_physical.csv       — paired contrast per ROI × model
  - encoding_summary.json        — best layers, significance, etc.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import stats
import json
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROI_NAMES = ['rTPJ', 'lTPJ', 'mPFC', 'PC', 'rSTS', 'lSTS']

# HRF delay: shift LLM features forward to account for hemodynamic lag
HRF_DELAY_TRS = 3  # ~4.5 seconds at TR=1.5s
PCA_COMPONENTS = 50
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS = 5
N_PERMUTATIONS = 500


# =============================================================================
# TEMPORAL BLOCK CV
# =============================================================================

def temporal_block_cv(n_samples, n_folds=N_FOLDS):
    """Generate temporal block cross-validation splits.

    Unlike random CV, this preserves temporal structure by splitting
    into contiguous blocks, avoiding data leakage from autocorrelation.
    """
    fold_size = n_samples // n_folds
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start),
                                     np.arange(test_end, n_samples)])
        yield train_idx, test_idx


# =============================================================================
# ENCODING MODEL
# =============================================================================

def fit_encoding_model(X, y, n_folds=N_FOLDS):
    """Fit ridge regression with temporal block CV.

    Args:
        X: (n_trs, n_features) LLM features (already PCA-reduced)
        y: (n_trs,) brain ROI time series

    Returns:
        r: Pearson correlation between predicted and actual
        r_per_fold: list of per-fold correlations
    """
    predictions = np.zeros_like(y)

    for train_idx, test_idx in temporal_block_cv(len(y), n_folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(X_train, y_train)
        predictions[test_idx] = ridge.predict(X_test)

    # Overall correlation
    r, p = stats.pearsonr(predictions, y)
    return float(r), float(p)


def apply_hrf_delay(X, delay_trs=HRF_DELAY_TRS):
    """Shift LLM features forward by HRF delay.

    The BOLD signal peaks ~4-6s after neural activity.
    Shifting LLM features forward aligns them with the delayed fMRI response.

    Args:
        X: (n_trs, n_features)
        delay_trs: number of TRs to shift

    Returns:
        X_shifted: (n_trs - delay_trs, n_features)
        valid_slice: slice for corresponding brain data
    """
    X_shifted = X[:-delay_trs] if delay_trs > 0 else X
    valid_slice = slice(delay_trs, None)
    return X_shifted, valid_slice


# =============================================================================
# NOISE CEILING (ISC-BASED)
# =============================================================================

def compute_noise_ceiling(roi_data_all_subjects):
    """Compute ISC-based noise ceiling for encoding models.

    The noise ceiling represents the maximum possible encoding performance,
    bounded by the reliability of the neural signal across subjects.

    Args:
        roi_data_all_subjects: (n_subjects, n_trs) array

    Returns:
        upper: upper bound (correlation of each subject with mean of all others)
        lower: lower bound (correlation of each subject with mean of all)
    """
    n_subjects = roi_data_all_subjects.shape[0]

    # Filter out subjects with zero variance (e.g., FOV doesn't cover ROI)
    valid = np.std(roi_data_all_subjects, axis=1) > 1e-10
    roi_data_all_subjects = roi_data_all_subjects[valid]
    n_subjects = roi_data_all_subjects.shape[0]
    if n_subjects < 3:
        return float('nan'), float('nan')

    # Upper bound: leave-one-out ISC
    upper_rs = []
    for i in range(n_subjects):
        others = np.delete(roi_data_all_subjects, i, axis=0)
        others_mean = others.mean(axis=0)
        r, _ = stats.pearsonr(roi_data_all_subjects[i], others_mean)
        upper_rs.append(r)

    # Lower bound: correlation with grand mean
    grand_mean = roi_data_all_subjects.mean(axis=0)
    lower_rs = []
    for i in range(n_subjects):
        r, _ = stats.pearsonr(roi_data_all_subjects[i], grand_mean)
        lower_rs.append(r)

    return float(np.mean(upper_rs)), float(np.mean(lower_rs))


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_stage3():
    """Run Stage 3: Encoding models."""
    logger.info("=" * 70)
    logger.info("STAGE 3: Encoding Models (Core Analysis)")
    logger.info("=" * 70)

    config.ensure_run_directories()
    output_dir = config.CROSS_DOMAIN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load brain data
    # ------------------------------------------------------------------
    roi_dir = config.BRAIN_ATTENTION_DIR / "roi_timeseries"
    subjects = config.get_subject_list()
    tasks = config.SHAPES_TASK['task_names']

    logger.info(f"Loading brain data: {len(subjects)} subjects × {len(tasks)} tasks")

    # Organize brain data: {task: {sub: (n_trs, n_rois)}}
    brain_data = {}
    for task in tasks:
        brain_data[task] = {}
        for sub in subjects:
            f = roi_dir / f"{sub}_{task}.npy"
            if f.exists():
                brain_data[task][sub] = np.load(str(f))

    # Find common TRs (min across all subjects/tasks)
    all_trs = [d.shape[0] for t in brain_data.values() for d in t.values()]
    min_trs = min(all_trs)
    logger.info(f"TR range: {min(all_trs)}-{max(all_trs)}, using min={min_trs}")

    # ------------------------------------------------------------------
    # Compute noise ceiling
    # ------------------------------------------------------------------
    logger.info("Computing noise ceiling (ISC)...")
    noise_ceiling_records = []

    for task in tasks:
        subs_with_data = sorted(brain_data[task].keys())
        for roi_idx, roi_name in enumerate(ROI_NAMES):
            roi_all = np.array([brain_data[task][s][:min_trs, roi_idx]
                                for s in subs_with_data])
            upper, lower = compute_noise_ceiling(roi_all)
            noise_ceiling_records.append({
                'task': task, 'roi': roi_name,
                'ceiling_upper': upper, 'ceiling_lower': lower,
                'n_subjects': len(subs_with_data),
            })
            logger.info(f"  {task} {roi_name}: ceiling=[{lower:.3f}, {upper:.3f}]")

    noise_df = pd.DataFrame(noise_ceiling_records)
    noise_df.to_csv(str(output_dir / "noise_ceiling.csv"), index=False)

    # ------------------------------------------------------------------
    # Load LLM features and run encoding
    # ------------------------------------------------------------------
    features_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_features"
    models_available = [d.name for d in features_dir.iterdir() if d.is_dir()] if features_dir.exists() else []

    if not models_available:
        logger.error("No LLM features found! Run Stage 2 first.")
        return

    logger.info(f"Models available: {models_available}")

    all_results = []

    for model_name in models_available:
        model_dir = features_dir / model_name
        logger.info(f"\n{'='*50}")
        logger.info(f"Encoding model: {model_name}")
        logger.info(f"{'='*50}")

        for task in tasks:
            # Find available layers
            layer_files = sorted(model_dir.glob(f"{task}_layer*.npy"),
                                 key=lambda p: int(p.stem.split('layer')[1]))
            if not layer_files:
                logger.warning(f"  No features for {task}, skipping")
                continue

            n_layers = len(layer_files)
            logger.info(f"  {task}: {n_layers} layers")

            for layer_file in tqdm(layer_files, desc=f"  {task} layers", leave=False):
                layer_idx = int(layer_file.stem.split('layer')[1])

                # Load LLM features for this layer
                llm_features = np.load(str(layer_file))  # (n_trs, hidden_dim)

                # Truncate to min_trs
                llm_features = llm_features[:min_trs]

                # Skip layers with NaN/Inf
                if np.isnan(llm_features).any() or np.isinf(llm_features).any():
                    logger.warning(f"    Layer {layer_idx}: NaN/Inf detected, skipping")
                    continue

                # Apply HRF delay
                X_shifted, brain_slice = apply_hrf_delay(llm_features, HRF_DELAY_TRS)

                # PCA dimensionality reduction
                n_components = min(PCA_COMPONENTS, X_shifted.shape[0] - 1, X_shifted.shape[1])
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_shifted)

                # Encode each subject × ROI
                for sub in subjects:
                    if sub not in brain_data[task]:
                        continue

                    brain_ts = brain_data[task][sub][:min_trs]

                    for roi_idx, roi_name in enumerate(ROI_NAMES):
                        y = brain_ts[brain_slice, roi_idx]

                        if len(X_pca) != len(y):
                            continue

                        r, p = fit_encoding_model(X_pca, y)

                        all_results.append({
                            'model': model_name,
                            'layer': layer_idx,
                            'task': task,
                            'subject': sub,
                            'roi': roi_name,
                            'encoding_r': r,
                            'p_value': p,
                            'n_trs': len(y),
                            'pca_components': n_components,
                            'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
                        })

    if not all_results:
        logger.error("No encoding results produced!")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(str(output_dir / "encoding_results.csv"), index=False)
    logger.info(f"\nEncoding results saved: {len(results_df)} entries")

    # ------------------------------------------------------------------
    # Social vs Physical contrast
    # ------------------------------------------------------------------
    logger.info("\nComputing social vs physical contrast...")
    contrast_records = []

    for model_name in results_df['model'].unique():
        for roi_name in ROI_NAMES:
            # For each subject, get best-layer encoding r for each task
            for layer in results_df[results_df['model'] == model_name]['layer'].unique():
                social_rs = results_df[
                    (results_df['model'] == model_name) &
                    (results_df['layer'] == layer) &
                    (results_df['task'] == 'shapessocial') &
                    (results_df['roi'] == roi_name)
                ]['encoding_r'].values

                physical_rs = results_df[
                    (results_df['model'] == model_name) &
                    (results_df['layer'] == layer) &
                    (results_df['task'] == 'shapesphysical') &
                    (results_df['roi'] == roi_name)
                ]['encoding_r'].values

                if len(social_rs) >= 10 and len(physical_rs) >= 10:
                    n_paired = min(len(social_rs), len(physical_rs))
                    t_stat, p_val = stats.ttest_rel(social_rs[:n_paired],
                                                     physical_rs[:n_paired])
                    contrast_records.append({
                        'model': model_name,
                        'layer': layer,
                        'roi': roi_name,
                        'social_mean_r': float(np.mean(social_rs)),
                        'physical_mean_r': float(np.mean(physical_rs)),
                        'difference': float(np.mean(social_rs) - np.mean(physical_rs)),
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'n_subjects': n_paired,
                    })

    contrast_df = pd.DataFrame(contrast_records)

    # Benjamini-Hochberg FDR correction across all contrasts
    if len(contrast_df) > 0:
        p_vals = contrast_df['p_value'].values
        n_tests = len(p_vals)
        order = np.argsort(p_vals)
        ranks = np.empty(n_tests, dtype=int)
        ranks[order] = np.arange(1, n_tests + 1)
        p_fdr = np.minimum.accumulate(
            (p_vals[order] * n_tests / np.arange(1, n_tests + 1))[::-1]
        )[::-1]
        p_fdr_full = np.clip(p_fdr[ranks - 1], 0, 1)
        contrast_df['p_fdr'] = p_fdr_full
        contrast_df['significant_fdr'] = p_fdr_full < 0.05

    contrast_df.to_csv(str(output_dir / "social_vs_physical.csv"), index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {'models': {}}

    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        model_summary = {}

        for task in tasks:
            task_results = model_results[model_results['task'] == task]
            if len(task_results) == 0:
                continue

            # Best layer per ROI (averaged across subjects)
            best_layers = {}
            for roi in ROI_NAMES:
                roi_data = task_results[task_results['roi'] == roi]
                if len(roi_data) == 0:
                    continue
                layer_means = roi_data.groupby('layer')['encoding_r'].mean()
                best_layer = int(layer_means.idxmax())
                best_r = float(layer_means.max())
                best_layers[roi] = {'layer': best_layer, 'mean_r': best_r}

            model_summary[task] = {
                'n_layers': int(task_results['layer'].nunique()),
                'best_layers': best_layers,
                'overall_mean_r': float(task_results['encoding_r'].mean()),
            }

        summary['models'][model_name] = model_summary

    # Add contrast summary
    if len(contrast_df) > 0:
        sig_contrasts = contrast_df[contrast_df['p_value'] < 0.05]
        sig_fdr = contrast_df[contrast_df.get('p_fdr', 1.0) < 0.05]
        summary['social_vs_physical'] = {
            'significant_contrasts': len(sig_contrasts),
            'significant_contrasts_fdr05': int(len(sig_fdr)),
            'total_contrasts': len(contrast_df),
        }

    with open(str(output_dir / "encoding_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Log key results
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Stage 3 Complete — Key Results")
    logger.info("=" * 70)

    for model_name, model_data in summary['models'].items():
        logger.info(f"\n  {model_name}:")
        for task, task_data in model_data.items():
            logger.info(f"    {task}: mean_r={task_data['overall_mean_r']:.4f}")
            for roi, best in task_data.get('best_layers', {}).items():
                logger.info(f"      {roi}: best_layer={best['layer']}, r={best['mean_r']:.4f}")

    if len(contrast_df) > 0:
        sv = summary.get('social_vs_physical', {})
        logger.info(f"\n  Social > Physical significant contrasts: "
                    f"{sv.get('significant_contrasts', 0)}/{sv.get('total_contrasts', 0)} "
                    f"(uncorrected), {sv.get('significant_contrasts_fdr05', 0)} after FDR")

    logger.info("=" * 70)


if __name__ == '__main__':
    run_stage3()
