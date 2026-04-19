#!/usr/bin/env python3
"""
=============================================================================
Stage 4: Attention Mechanism Analysis
=============================================================================

Analyze how transformer attention differs between social and physical narratives:
  1. Sparsity comparison (social vs physical) with defined thresholds
  2. Head specialization for mental-state content
  3. Attention flow through mental-state tokens

Outputs:
  - attention_sparsity_comparison.csv
  - head_specialization_summary.csv
  - attention_analysis_summary.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
import json
import logging
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPARSITY_THRESHOLD_GINI = 0.5
SPARSITY_THRESHOLD_ENTROPY = 0.5


def run_stage4():
    """Run Stage 4: Attention analysis."""
    logger.info("=" * 70)
    logger.info("STAGE 4: Attention Mechanism Analysis")
    logger.info("=" * 70)

    config.ensure_run_directories()
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_attention"
    output_dir = config.CROSS_DOMAIN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not attention_dir.exists():
        logger.error("No attention data found! Run Stage 2 first.")
        return

    models = [d.name for d in attention_dir.iterdir() if d.is_dir()]
    logger.info(f"Models: {models}")

    # ------------------------------------------------------------------
    # 1. Sparsity comparison: social vs physical
    # ------------------------------------------------------------------
    logger.info("\n--- Sparsity Comparison ---")
    all_sparsity = []

    for model_name in models:
        for task in ['shapessocial', 'shapesphysical']:
            csv_path = attention_dir / model_name / f"{task}_sparsity.csv"
            if csv_path.exists():
                df = pd.read_csv(str(csv_path))
                all_sparsity.append(df)

    if not all_sparsity:
        logger.error("No sparsity CSVs found!")
        return

    sparsity_df = pd.concat(all_sparsity, ignore_index=True)

    # Compare social vs physical per model per layer
    comparison_records = []
    for model_name in sparsity_df['model'].unique():
        model_data = sparsity_df[sparsity_df['model'] == model_name]

        for layer in model_data['layer'].unique():
            social = model_data[(model_data['task'] == 'shapessocial') &
                                (model_data['layer'] == layer)]['sparsity']
            physical = model_data[(model_data['task'] == 'shapesphysical') &
                                  (model_data['layer'] == layer)]['sparsity']

            if len(social) > 0 and len(physical) > 0:
                t_stat, p_val = stats.ttest_ind(social.values, physical.values)
                comparison_records.append({
                    'model': model_name,
                    'layer': int(layer),
                    'social_mean_sparsity': float(social.mean()),
                    'physical_mean_sparsity': float(physical.mean()),
                    'difference': float(social.mean() - physical.mean()),
                    'social_mean_gini': float(
                        model_data[(model_data['task'] == 'shapessocial') &
                                   (model_data['layer'] == layer)]['gini'].mean()),
                    'physical_mean_gini': float(
                        model_data[(model_data['task'] == 'shapesphysical') &
                                   (model_data['layer'] == layer)]['gini'].mean()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                })

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(str(output_dir / "attention_sparsity_comparison.csv"), index=False)
    logger.info(f"  Sparsity comparison: {len(comparison_df)} layer×model entries")

    # ------------------------------------------------------------------
    # 2. Head specialization summary
    # ------------------------------------------------------------------
    logger.info("\n--- Head Specialization ---")

    # Count sparse and specialized heads per model
    spec_records = []
    for model_name in sparsity_df['model'].unique():
        model_data = sparsity_df[sparsity_df['model'] == model_name]

        for task in ['shapessocial', 'shapesphysical']:
            task_data = model_data[model_data['task'] == task]
            n_heads_total = len(task_data)
            n_sparse_gini = int((task_data['gini'] > SPARSITY_THRESHOLD_GINI).sum())
            n_sparse_entropy = int((task_data['sparsity'] > SPARSITY_THRESHOLD_ENTROPY).sum())
            mean_gini = float(task_data['gini'].mean())
            mean_entropy = float(task_data['entropy'].mean())
            mean_top5 = float(task_data['top5_concentration'].mean())

            spec_records.append({
                'model': model_name,
                'task': task,
                'n_heads_total': n_heads_total,
                'n_sparse_gini': n_sparse_gini,
                'pct_sparse_gini': n_sparse_gini / max(n_heads_total, 1),
                'n_sparse_entropy': n_sparse_entropy,
                'mean_gini': mean_gini,
                'mean_entropy': mean_entropy,
                'mean_top5_concentration': mean_top5,
            })

    spec_df = pd.DataFrame(spec_records)
    spec_df.to_csv(str(output_dir / "head_specialization_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    summary = {
        'thresholds': {
            'gini': SPARSITY_THRESHOLD_GINI,
            'entropy': SPARSITY_THRESHOLD_ENTROPY,
        },
        'models': {},
    }

    for model_name in sparsity_df['model'].unique():
        model_spec = spec_df[spec_df['model'] == model_name]
        social_row = model_spec[model_spec['task'] == 'shapessocial']
        physical_row = model_spec[model_spec['task'] == 'shapesphysical']

        summary['models'][model_name] = {
            'social_mean_gini': float(social_row['mean_gini'].values[0]) if len(social_row) > 0 else None,
            'physical_mean_gini': float(physical_row['mean_gini'].values[0]) if len(physical_row) > 0 else None,
            'social_pct_sparse': float(social_row['pct_sparse_gini'].values[0]) if len(social_row) > 0 else None,
            'physical_pct_sparse': float(physical_row['pct_sparse_gini'].values[0]) if len(physical_row) > 0 else None,
        }

    with open(str(output_dir / "attention_analysis_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 4 Complete")
    for model_name, data in summary['models'].items():
        logger.info(f"  {model_name}: social_gini={data['social_mean_gini']:.3f}, "
                    f"physical_gini={data['physical_mean_gini']:.3f}")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_stage4()
