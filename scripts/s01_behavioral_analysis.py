#!/usr/bin/env python3
"""
=============================================================================
Stage 1: Behavioral Analysis - Social Cognition Task
=============================================================================

Analyze behavioral data from HCP Social Cognition (Theory of Mind) task.

Extracts:
    - Response times for Mental vs Random conditions
    - Accuracy rates
    - Heuristic processing index
    - Links to P1 efficiency groups

Outputs:
    - behavioral_summary.csv
    - behavioral_stats.json
    - efficiency_tom_link.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure directories are initialized (for subprocess runs)
config.ensure_run_directories()

logger = config.setup_logging('behavioral_analysis')

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_social_behavioral_data(subject):
    """Extract behavioral data from Social task for one subject"""
    
    behavioral_data = {
        'subject': subject,
        'mental_accuracy': np.nan,
        'random_accuracy': np.nan,
        'mental_rt': np.nan,
        'random_rt': np.nan,
    }
    
    for run in ['LR', 'RL']:
        stats_path = (config.DATA_ROOT / subject / "MNINonLinear" / "Results" / 
                     f"tfMRI_SOCIAL_{run}" / "EVs" / "SOCIAL_Stats.csv")
        
        if not stats_path.exists():
            continue
        
        try:
            df = pd.read_csv(stats_path)
            
            # Extract accuracy (proportion judged as ToM)
            for _, row in df.iterrows():
                condition = row['ConditionName']
                measure = row['Measure']
                value = row['Value']
                
                if condition == 'TOM' and measure == 'PROP_TOM':
                    behavioral_data['mental_accuracy'] = value
                elif condition == 'RANDOM' and measure == 'PROP_RANDOM':
                    # For random, correct response is NOT judging as ToM
                    behavioral_data['random_accuracy'] = 1.0 - value if pd.notna(value) else np.nan
                elif condition == 'TOM' and measure == 'MEDIAN_RT_TOM':
                    behavioral_data['mental_rt'] = value
                elif condition == 'RANDOM' and measure == 'MEDIAN_RT_TOM':
                    behavioral_data['random_rt'] = value
                    
        except Exception as e:
            logger.warning(f"Error reading {stats_path}: {e}")
            continue
    
    return behavioral_data


def compute_heuristic_index(row):
    """
    Compute heuristic processing index.
    
    Higher values indicate more heuristic (fast, intuitive) processing.
    
    Heuristic users: Fast on Mental (intuitive ToM inference)
    Analytical users: Slow on Mental (deliberate reasoning)
    """
    rt_mental = row['mental_rt']
    rt_random = row['random_rt']
    acc_mental = row['mental_accuracy']
    
    if pd.isna(rt_mental) or pd.isna(rt_random) or rt_mental == 0:
        return np.nan
    
    # Speed advantage: higher if faster on mental relative to random
    speed_ratio = rt_random / rt_mental
    
    # Weight by accuracy
    heuristic_index = speed_ratio * acc_mental
    
    return heuristic_index


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_behavioral_analysis():
    """Run complete behavioral analysis"""
    
    logger.info("="*60)
    logger.info("Stage 1: Behavioral Analysis - Social Cognition")
    logger.info("="*60)
    
    # Extract data for all subjects
    logger.info(f"Extracting behavioral data for {len(config.SUBJECTS)} subjects...")
    
    all_data = []
    for subject in tqdm(config.SUBJECTS, desc="Extracting"):
        data = extract_social_behavioral_data(subject)
        all_data.append(data)
    
    df = pd.DataFrame(all_data)
    
    # Compute derived metrics
    logger.info("Computing derived metrics...")
    
    # Heuristic index
    df['heuristic_index'] = df.apply(compute_heuristic_index, axis=1)
    
    # Overall accuracy
    df['overall_accuracy'] = (df['mental_accuracy'] + df['random_accuracy']) / 2
    
    # RT cost (mental - random, positive = slower on mental)
    df['rt_cost'] = df['mental_rt'] - df['random_rt']
    
    # Classify as heuristic vs analytical
    median_hi = df['heuristic_index'].median()
    df['processing_style'] = df['heuristic_index'].apply(
        lambda x: 'heuristic' if x >= median_hi else 'analytical'
    )
    
    # Link to P1 efficiency groups
    logger.info("Linking to P1 efficiency groups...")
    p1_groups = config.load_p1_efficiency_groups()
    
    if p1_groups is not None:
        if isinstance(p1_groups, dict):
            # JSON format
            high_eff = p1_groups.get('high', [])
            low_eff = p1_groups.get('low', [])
            df['p1_efficiency_group'] = df['subject'].apply(
                lambda x: 'high' if x in high_eff else ('low' if x in low_eff else 'unknown')
            )
        else:
            # DataFrame format
            df = df.merge(
                p1_groups[['subject', 'efficiency_group']], 
                on='subject', 
                how='left'
            )
            df.rename(columns={'efficiency_group': 'p1_efficiency_group'}, inplace=True)
    else:
        logger.warning("Could not load P1 efficiency groups")
        df['p1_efficiency_group'] = 'unknown'
    
    # Save behavioral summary
    df.to_csv(config.BEHAVIORAL_DIR / "behavioral_summary.csv", index=False)
    logger.info(f"Saved: {config.BEHAVIORAL_DIR / 'behavioral_summary.csv'}")
    
    # Compute statistics
    stats_results = {
        'n_subjects': len(df),
        'mental_accuracy': {
            'mean': float(df['mental_accuracy'].mean()),
            'std': float(df['mental_accuracy'].std()),
        },
        'random_accuracy': {
            'mean': float(df['random_accuracy'].mean()),
            'std': float(df['random_accuracy'].std()),
        },
        'mental_rt': {
            'mean': float(df['mental_rt'].mean()),
            'std': float(df['mental_rt'].std()),
        },
        'random_rt': {
            'mean': float(df['random_rt'].mean()),
            'std': float(df['random_rt'].std()),
        },
        'heuristic_index': {
            'mean': float(df['heuristic_index'].mean()),
            'std': float(df['heuristic_index'].std()),
            'median': float(df['heuristic_index'].median()),
        },
        'processing_style_counts': df['processing_style'].value_counts().to_dict(),
    }
    
    # Test RT difference between conditions
    valid_df = df.dropna(subset=['mental_rt', 'random_rt'])
    if len(valid_df) > 1:
        t_stat, p_val = stats.ttest_rel(valid_df['mental_rt'], valid_df['random_rt'])
        stats_results['rt_condition_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
        }
    
    # Test if efficiency group predicts heuristic index
    if 'p1_efficiency_group' in df.columns:
        high_hi = df[df['p1_efficiency_group'] == 'high']['heuristic_index'].dropna()
        low_hi = df[df['p1_efficiency_group'] == 'low']['heuristic_index'].dropna()
        
        if len(high_hi) > 1 and len(low_hi) > 1:
            t_stat, p_val = stats.ttest_ind(high_hi, low_hi)
            stats_results['efficiency_heuristic_link'] = {
                'high_eff_heuristic_mean': float(high_hi.mean()),
                'low_eff_heuristic_mean': float(low_hi.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'hypothesis_supported': bool(high_hi.mean() > low_hi.mean() and p_val < 0.1),
            }
    
    with open(config.BEHAVIORAL_DIR / "behavioral_stats.json", 'w') as f:
        json.dump(stats_results, f, indent=2)
    logger.info(f"Saved: {config.BEHAVIORAL_DIR / 'behavioral_stats.json'}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Behavioral Analysis Summary")
    logger.info("="*60)
    logger.info(f"Subjects analyzed: {stats_results['n_subjects']}")
    logger.info(f"Mental accuracy: {stats_results['mental_accuracy']['mean']:.3f} ± {stats_results['mental_accuracy']['std']:.3f}")
    logger.info(f"Random accuracy: {stats_results['random_accuracy']['mean']:.3f} ± {stats_results['random_accuracy']['std']:.3f}")
    logger.info(f"Mental RT: {stats_results['mental_rt']['mean']:.1f} ± {stats_results['mental_rt']['std']:.1f} ms")
    logger.info(f"Random RT: {stats_results['random_rt']['mean']:.1f} ± {stats_results['random_rt']['std']:.1f} ms")
    logger.info(f"Heuristic index: {stats_results['heuristic_index']['mean']:.3f} ± {stats_results['heuristic_index']['std']:.3f}")
    
    if 'efficiency_heuristic_link' in stats_results:
        link = stats_results['efficiency_heuristic_link']
        logger.info(f"\nP1 Efficiency → Heuristic Processing:")
        logger.info(f"  High efficiency: {link['high_eff_heuristic_mean']:.3f}")
        logger.info(f"  Low efficiency: {link['low_eff_heuristic_mean']:.3f}")
        logger.info(f"  t={link['t_statistic']:.2f}, p={link['p_value']:.4f}")
        if link['hypothesis_supported']:
            logger.info("  ✓ H1 SUPPORTED: High efficiency → More heuristic processing")
        else:
            logger.info("  ✗ H1 not supported")
    
    return df, stats_results


if __name__ == "__main__":
    df, stats = run_behavioral_analysis()
    print("\n✅ Stage 1 completed: Behavioral analysis")

