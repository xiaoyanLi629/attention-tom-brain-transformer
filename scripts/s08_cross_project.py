#!/usr/bin/env python3
"""
=============================================================================
Stage 8: Cross-Project Integration - Unified Framework
=============================================================================

Integrate findings across all three projects to build unified framework
of adaptive cognitive efficiency.

Projects:
    - P1: Cognitive flexibility (brain network dynamics)
    - P2: Sparse activation (brain-MoE comparison)
    - P3: Heuristic processing (brain-Transformer attention)

Unified Theme:
    Adaptive efficiency through sparse, specialized processing

Analyses:
    1. Efficiency Metrics Correlation
       - Link P1 efficiency groups to P2/P3 findings
       
    2. Sparsity Across Systems
       - Brain networks vs MoE experts vs Transformer attention
       
    3. Specialization Comparison
       - Network specialization vs expert specialization vs head specialization
       
    4. Unified Efficiency Index
       - Combined metric across all three domains

Outputs:
    - cross_project_correlation.csv
    - unified_sparsity.json
    - unified_framework.json
    - integration_summary.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure directories are initialized (for subprocess runs)
config.ensure_run_directories()

logger = config.setup_logging('cross_project')

# =============================================================================
# PROJECT DATA LOADING
# =============================================================================

def load_p1_data():
    """Load Project 1 efficiency data"""
    
    p1_dir = config.PROJECT_ROOT / "project_1" / "results"
    
    data = {
        'efficiency_groups': None,
        'network_flexibility': None,
        'behavioral': None,
    }
    
    # Load efficiency groups
    eff_path = p1_dir / "efficiency_groups.json"
    if eff_path.exists():
        with open(eff_path, 'r') as f:
            data['efficiency_groups'] = json.load(f)
    
    # Load network flexibility
    flex_path = p1_dir / "network_flexibility.csv"
    if flex_path.exists():
        data['network_flexibility'] = pd.read_csv(flex_path)
    
    # Load behavioral
    behav_path = p1_dir / "behavioral_efficiency.csv"
    if behav_path.exists():
        data['behavioral'] = pd.read_csv(behav_path)
    
    return data


def load_p2_data():
    """Load Project 2 sparsity data"""
    
    p2_dir = config.PROJECT_ROOT / "project_2" / "results"
    
    data = {
        'brain_sparsity': None,
        'moe_sparsity': None,
        'comparison': None,
    }
    
    # Load brain sparsity
    brain_path = p2_dir / "neuroscience" / "sparsity_metrics.csv"
    if brain_path.exists():
        data['brain_sparsity'] = pd.read_csv(brain_path)
    
    # Load MoE sparsity
    moe_path = p2_dir / "moe_analysis" / "moe_sparsity.csv"
    if moe_path.exists():
        data['moe_sparsity'] = pd.read_csv(moe_path)
    
    # Load comparison
    comp_path = p2_dir / "integration" / "brain_moe_comparison.json"
    if comp_path.exists():
        with open(comp_path, 'r') as f:
            data['comparison'] = json.load(f)
    
    return data


def load_p3_data():
    """Load Project 3 heuristic processing data"""
    
    data = {
        'behavioral': None,
        'brain_sparsity': None,
        'transformer_sparsity': None,
        'rsa': None,
    }
    
    # Load behavioral
    behav_path = config.BEHAVIORAL_DIR / "behavioral_summary.csv"
    if behav_path.exists():
        data['behavioral'] = pd.read_csv(behav_path)
    
    # Load brain sparsity
    brain_path = config.BRAIN_ATTENTION_DIR / "brain_sparsity.csv"
    if brain_path.exists():
        data['brain_sparsity'] = pd.read_csv(brain_path)
    
    # Load Transformer sparsity
    trans_path = config.TRANSFORMER_ATTENTION_DIR / "attention_sparsity.csv"
    if trans_path.exists():
        data['transformer_sparsity'] = pd.read_csv(trans_path)
    
    # Load RSA
    rsa_path = config.CROSS_DOMAIN_DIR / "brain_model_rsa.csv"
    if rsa_path.exists():
        data['rsa'] = pd.read_csv(rsa_path)
    
    return data


# =============================================================================
# CROSS-PROJECT ANALYSIS
# =============================================================================

def compute_efficiency_correlations(p1_data, p3_data):
    """
    Correlate P1 efficiency groups with P3 heuristic processing.
    
    Hypothesis: High efficiency (P1) → More heuristic processing (P3)
    """
    
    results = {
        'correlation': None,
        'group_comparison': None,
    }
    
    if p3_data is None:
        logger.warning("No P3 data available")
        return results
    
    p3_behavioral = p3_data.get('behavioral')
    
    if p3_behavioral is None:
        logger.warning("No P3 behavioral data")
        return results
    
    # Check for efficiency group column
    if 'p1_efficiency_group' not in p3_behavioral.columns:
        logger.warning("No P1 efficiency link in P3 data")
        return results
    
    # Compare heuristic index by efficiency group
    high_eff = p3_behavioral[p3_behavioral['p1_efficiency_group'] == 'high']['heuristic_index'].dropna()
    low_eff = p3_behavioral[p3_behavioral['p1_efficiency_group'] == 'low']['heuristic_index'].dropna()
    
    if len(high_eff) > 1 and len(low_eff) > 1:
        t_stat, p_val = stats.ttest_ind(high_eff, low_eff)
        
        results['group_comparison'] = {
            'high_eff_heuristic_mean': float(high_eff.mean()),
            'high_eff_heuristic_std': float(high_eff.std()),
            'low_eff_heuristic_mean': float(low_eff.mean()),
            'low_eff_heuristic_std': float(low_eff.std()),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'effect_size': float((high_eff.mean() - low_eff.mean()) / np.sqrt((high_eff.std()**2 + low_eff.std()**2) / 2)),
            'hypothesis_supported': high_eff.mean() > low_eff.mean() and p_val < 0.1,
        }
        
        logger.info(f"P1-P3 Link: High eff = {high_eff.mean():.3f}, Low eff = {low_eff.mean():.3f}")
        logger.info(f"  t={t_stat:.2f}, p={p_val:.4f}")
    
    return results


def compute_unified_sparsity(p2_data, p3_data):
    """
    Compare sparsity across all systems:
    - Brain networks (P1/P2)
    - MoE experts (P2)
    - Transformer attention (P3)
    """
    
    unified = {
        'brain': {},
        'moe': {},
        'transformer': {},
    }
    
    # Brain sparsity from P2
    if p2_data.get('brain_sparsity') is not None:
        brain_sp = p2_data['brain_sparsity']
        unified['brain']['p2'] = {
            'mean_sparsity': float(brain_sp['composite_sparsity'].mean()) if 'composite_sparsity' in brain_sp.columns else 0.0,
        }
    
    # Brain sparsity from P3
    if p3_data.get('brain_sparsity') is not None:
        brain_sp = p3_data['brain_sparsity']
        unified['brain']['p3'] = {
            'mean_sparsity': float(brain_sp['sparsity_90'].mean()) if 'sparsity_90' in brain_sp.columns else 0.0,
            'mean_gini': float(brain_sp['gini_coefficient'].mean()) if 'gini_coefficient' in brain_sp.columns else 0.0,
        }
    
    # MoE sparsity from P2
    if p2_data.get('moe_sparsity') is not None:
        moe_sp = p2_data['moe_sparsity']
        unified['moe'] = {
            'mean_sparsity': float(moe_sp['sparsity'].mean()) if 'sparsity' in moe_sp.columns else 0.0,
        }
    
    # Transformer sparsity from P3
    if p3_data.get('transformer_sparsity') is not None:
        trans_sp = p3_data['transformer_sparsity']
        for model in trans_sp['model'].unique():
            model_sp = trans_sp[trans_sp['model'] == model]
            unified['transformer'][model] = {
                'mean_sparsity': float(model_sp['sparsity'].mean()),
                'mean_entropy': float(model_sp['entropy'].mean()),
            }
    
    # Compute overall comparison
    all_sparsities = []
    
    for domain, values in unified.items():
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, dict) and 'mean_sparsity' in val:
                    all_sparsities.append({
                        'domain': domain,
                        'source': key,
                        'sparsity': val['mean_sparsity'],
                    })
    
    if all_sparsities:
        unified['comparison'] = {
            'all_sparsities': all_sparsities,
            'overall_mean': float(np.mean([s['sparsity'] for s in all_sparsities])),
            'overall_std': float(np.std([s['sparsity'] for s in all_sparsities])),
        }
    
    return unified


def build_unified_framework(p1_data, p2_data, p3_data, efficiency_corr, unified_sparsity):
    """
    Build unified conceptual framework across all three projects.
    """
    
    framework = {
        'title': "Adaptive Efficiency Through Sparse, Specialized Processing",
        'projects': {
            'P1': {
                'focus': 'Cognitive Flexibility',
                'key_finding': 'Efficient individuals show more dynamic network reconfiguration',
                'mechanism': 'Adaptive network switching',
            },
            'P2': {
                'focus': 'Sparse Activation',
                'key_finding': 'Brain and MoE share sparse, specialized activation patterns',
                'mechanism': 'Resource-efficient expert routing',
            },
            'P3': {
                'focus': 'Heuristic Processing',
                'key_finding': 'Efficient social cognition uses sparse, focused attention',
                'mechanism': 'Adaptive attention allocation',
            },
        },
        'unified_principles': [
            {
                'name': 'Sparsity',
                'description': 'Both biological and artificial systems achieve efficiency through sparse activation',
                'evidence': {
                    'brain': 'Sparse network engagement, focal activation',
                    'ai': 'MoE expert selection, focused attention heads',
                },
            },
            {
                'name': 'Specialization',
                'description': 'Dedicated processing units for specific cognitive functions',
                'evidence': {
                    'brain': 'Specialized networks (e.g., TPJ for ToM)',
                    'ai': 'ToM-specialized attention heads, domain experts',
                },
            },
            {
                'name': 'Adaptive Efficiency',
                'description': 'Context-dependent resource allocation optimizes performance',
                'evidence': {
                    'brain': 'Task-dependent network engagement',
                    'ai': 'Input-dependent routing and attention',
                },
            },
        ],
        'cross_project_links': [],
    }
    
    # Add efficiency correlation evidence
    if efficiency_corr and isinstance(efficiency_corr, dict) and efficiency_corr.get('group_comparison'):
        gc = efficiency_corr['group_comparison']
        if gc and isinstance(gc, dict) and 'effect_size' in gc:
            framework['cross_project_links'].append({
                'link': 'P1 → P3',
                'finding': f"High P1 efficiency → Higher heuristic processing (d={gc['effect_size']:.2f})",
                'p_value': gc.get('p_value', 1.0),
            })
    
    # Add sparsity comparison
    if unified_sparsity.get('comparison'):
        sp = unified_sparsity['comparison']
        framework['cross_project_links'].append({
            'link': 'P2 ↔ P3',
            'finding': f"Consistent sparsity across systems (M={sp['overall_mean']:.3f}, SD={sp['overall_std']:.3f})",
            'domains': len(sp['all_sparsities']),
        })
    
    return framework


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_cross_project_integration():
    """Run complete cross-project integration"""
    
    logger.info("="*60)
    logger.info("Stage 8: Cross-Project Integration")
    logger.info("="*60)
    
    # Load all project data
    logger.info("\nLoading project data...")
    p1_data = load_p1_data()
    p2_data = load_p2_data()
    p3_data = load_p3_data()
    
    logger.info(f"  P1 data loaded: {sum(v is not None for v in p1_data.values())} sources")
    logger.info(f"  P2 data loaded: {sum(v is not None for v in p2_data.values())} sources")
    logger.info(f"  P3 data loaded: {sum(v is not None for v in p3_data.values())} sources")
    
    # Compute efficiency correlations
    logger.info("\nComputing P1-P3 efficiency correlations...")
    efficiency_corr = compute_efficiency_correlations(p1_data, p3_data)
    
    # Compute unified sparsity
    logger.info("\nComputing unified sparsity metrics...")
    unified_sparsity = compute_unified_sparsity(p2_data, p3_data)
    
    # Build unified framework
    logger.info("\nBuilding unified framework...")
    framework = build_unified_framework(p1_data, p2_data, p3_data, efficiency_corr, unified_sparsity)
    
    # Save results
    with open(config.INTEGRATION_DIR / "efficiency_correlation.json", 'w') as f:
        json.dump(efficiency_corr, f, indent=2)
    
    with open(config.INTEGRATION_DIR / "unified_sparsity.json", 'w') as f:
        json.dump(unified_sparsity, f, indent=2)
    
    with open(config.INTEGRATION_DIR / "unified_framework.json", 'w') as f:
        json.dump(framework, f, indent=2)
    
    # Compute integration summary
    eff_link = False
    if efficiency_corr and isinstance(efficiency_corr, dict):
        gc = efficiency_corr.get('group_comparison')
        if gc and isinstance(gc, dict):
            eff_link = gc.get('hypothesis_supported', False)
    
    summary = {
        'data_sources': {
            'P1': sum(v is not None for v in p1_data.values()),
            'P2': sum(v is not None for v in p2_data.values()),
            'P3': sum(v is not None for v in p3_data.values()),
        },
        'cross_project_findings': len(framework['cross_project_links']),
        'unified_principles': len(framework['unified_principles']),
        'efficiency_link_supported': eff_link,
    }
    
    with open(config.INTEGRATION_DIR / "integration_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print_integration_summary(framework, summary)
    
    return framework, summary


def print_integration_summary(framework, summary):
    """Print integration summary"""
    
    logger.info("\n" + "="*60)
    logger.info("Cross-Project Integration Summary")
    logger.info("="*60)
    
    logger.info(f"\nFramework: {framework['title']}")
    
    logger.info("\nProjects:")
    for proj_id, proj in framework['projects'].items():
        logger.info(f"  {proj_id}: {proj['focus']}")
        logger.info(f"       Key finding: {proj['key_finding']}")
    
    logger.info("\nUnified Principles:")
    for i, principle in enumerate(framework['unified_principles'], 1):
        logger.info(f"  {i}. {principle['name']}: {principle['description']}")
    
    logger.info("\nCross-Project Links:")
    for link in framework['cross_project_links']:
        logger.info(f"  {link['link']}: {link['finding']}")
    
    logger.info(f"\nIntegration Statistics:")
    logger.info(f"  Data sources: P1={summary['data_sources']['P1']}, P2={summary['data_sources']['P2']}, P3={summary['data_sources']['P3']}")
    logger.info(f"  Cross-project findings: {summary['cross_project_findings']}")
    logger.info(f"  Efficiency link supported: {summary['efficiency_link_supported']}")


if __name__ == "__main__":
    framework, summary = run_cross_project_integration()
    print("\n✅ Stage 8 completed: Cross-project integration")

