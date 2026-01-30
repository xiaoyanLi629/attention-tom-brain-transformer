#!/usr/bin/env python3
"""
=============================================================================
Stage 4: Probing Classifier Analysis
=============================================================================

Train linear probing classifiers on Transformer hidden states to understand
what information is encoded at each layer.

Analyses:
    1. ToM vs Factual Classification
       - Which layers distinguish ToM from factual prompts?
       
    2. Mental State Word Detection
       - Can we decode mental state concepts from representations?
       
    3. Layer-wise Information Content
       - How does ToM-relevant information evolve across layers?
       
    4. Comparison with Brain ROI Representations
       - Do probing accuracies correlate with brain activations?

Outputs:
    - probing_results.csv
    - layer_information.json
    - probing_vs_brain.csv
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure directories are initialized (for subprocess runs)
config.ensure_run_directories()

logger = config.setup_logging('probing_analysis')

# =============================================================================
# PROBING CLASSIFIERS
# =============================================================================

def load_hidden_states(model_name):
    """Load hidden states for a model"""
    
    hidden_dir = config.TRANSFORMER_ATTENTION_DIR / "hidden_states" / model_name.replace(' ', '_').replace('-', '_')
    
    if not hidden_dir.exists():
        logger.warning(f"No hidden states for {model_name}")
        return None, None, None
    
    all_hidden_states = []
    all_labels = []
    all_metadata = []
    
    for pkl_file in sorted(hidden_dir.glob("prompt_*.pkl")):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        hidden_states = data['hidden_states']  # (n_layers+1, seq_len, hidden_dim)
        prompt_idx = data['prompt_idx']
        
        # Load corresponding attention file for metadata
        attn_dir = config.TRANSFORMER_ATTENTION_DIR / "attention_weights" / model_name.replace(' ', '_').replace('-', '_')
        attn_file = attn_dir / f"prompt_{prompt_idx}.pkl"
        
        if attn_file.exists():
            with open(attn_file, 'rb') as f:
                attn_data = pickle.load(f)
            category = attn_data.get('category', 'unknown')
            prompt_type = attn_data.get('prompt_type', 'unknown')
        else:
            category = 'unknown'
            prompt_type = 'unknown'
        
        all_hidden_states.append(hidden_states)
        all_labels.append({
            'prompt_idx': prompt_idx,
            'category': category,
            'prompt_type': prompt_type,
            'is_tom': 1 if category == 'mental_intentional' else 0,
        })
        all_metadata.append({
            'prompt_idx': prompt_idx,
            'n_layers': hidden_states.shape[0] - 1,
            'seq_len': hidden_states.shape[1],
            'hidden_dim': hidden_states.shape[2],
        })
    
    return all_hidden_states, all_labels, all_metadata


def train_probing_classifier(hidden_states_list, labels, layer_idx, pooling='mean'):
    """
    Train a probing classifier for a specific layer.
    
    Args:
        hidden_states_list: List of hidden state arrays (n_layers+1, seq_len, hidden_dim)
        labels: List of dicts with 'is_tom' label
        layer_idx: Which layer to probe
        pooling: 'mean', 'last', or 'cls'
    
    Returns:
        Cross-validation accuracy and trained classifier
    """
    
    X = []
    y = []
    
    for hidden_states, label in zip(hidden_states_list, labels):
        if layer_idx >= hidden_states.shape[0]:
            continue
        
        layer_repr = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        
        if pooling == 'mean':
            vec = layer_repr.mean(axis=0)
        elif pooling == 'last':
            vec = layer_repr[-1]
        elif pooling == 'cls':
            vec = layer_repr[0]
        else:
            vec = layer_repr.mean(axis=0)
        
        X.append(vec)
        y.append(label['is_tom'])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(np.unique(y)) < 2:
        logger.warning(f"Only one class present for layer {layer_idx}")
        return 0.5, None
    
    # Create pipeline with scaling and logistic regression
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    ])
    
    # Cross-validation
    n_splits = min(5, len(X))
    if n_splits < 2:
        logger.warning(f"Not enough samples for CV at layer {layer_idx}")
        return 0.5, None
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        
        # Fit final model
        clf.fit(X, y)
        
        return mean_acc, clf
        
    except Exception as e:
        logger.warning(f"Error training classifier for layer {layer_idx}: {e}")
        return 0.5, None


def probe_all_layers(model_name, hidden_states_list, labels):
    """Probe all layers for ToM classification"""
    
    if not hidden_states_list or len(hidden_states_list) == 0:
        return []
    
    n_layers = hidden_states_list[0].shape[0] - 1  # Exclude embedding layer
    
    results = []
    
    for layer_idx in tqdm(range(n_layers + 1), desc=f"Probing {model_name}"):
        for pooling in ['mean', 'last']:
            acc, clf = train_probing_classifier(
                hidden_states_list, labels, layer_idx, pooling
            )
            
            results.append({
                'model': model_name,
                'layer': layer_idx,
                'pooling': pooling,
                'accuracy': acc,
                'above_chance': acc > 0.5,
            })
    
    return results


# =============================================================================
# MENTAL STATE DECODING
# =============================================================================

def compute_mental_state_selectivity(hidden_states_list, labels):
    """
    Compute how selective each layer is for mental state content.
    
    Uses the difference in representation between mental and non-mental tokens.
    """
    
    # This requires token-level labels which we'll approximate
    # by comparing representations for ToM vs control prompts
    
    if len(hidden_states_list) == 0:
        return []
    
    n_layers = hidden_states_list[0].shape[0]
    
    tom_reprs = []
    ctrl_reprs = []
    
    for hidden_states, label in zip(hidden_states_list, labels):
        mean_repr = hidden_states.mean(axis=1)  # (n_layers, hidden_dim)
        
        if label['is_tom']:
            tom_reprs.append(mean_repr)
        else:
            ctrl_reprs.append(mean_repr)
    
    if len(tom_reprs) == 0 or len(ctrl_reprs) == 0:
        return []
    
    tom_mean = np.mean(tom_reprs, axis=0)  # (n_layers, hidden_dim)
    ctrl_mean = np.mean(ctrl_reprs, axis=0)
    
    results = []
    for layer_idx in range(n_layers):
        # Compute distance between ToM and control representations
        diff = tom_mean[layer_idx] - ctrl_mean[layer_idx]
        distance = np.linalg.norm(diff)
        
        # Compute cosine similarity
        cos_sim = np.dot(tom_mean[layer_idx], ctrl_mean[layer_idx]) / (
            np.linalg.norm(tom_mean[layer_idx]) * np.linalg.norm(ctrl_mean[layer_idx]) + 1e-10
        )
        
        results.append({
            'layer': layer_idx,
            'tom_ctrl_distance': float(distance),
            'tom_ctrl_cosine': float(cos_sim),
            'selectivity': float(1 - cos_sim),  # Higher = more selective
        })
    
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_probing_analysis():
    """Run complete probing analysis"""
    
    logger.info("="*60)
    logger.info("Stage 4: Probing Classifier Analysis")
    logger.info("="*60)
    
    all_probing_results = []
    all_selectivity_results = []
    layer_info = {}
    
    for model_config in config.TRANSFORMER_CONFIG['models']:
        model_name = model_config['short_name']
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*40}")
        
        # Load hidden states
        hidden_states_list, labels, metadata = load_hidden_states(model_name)
        
        if hidden_states_list is None or len(hidden_states_list) == 0:
            logger.warning(f"No data for {model_name}")
            continue
        
        logger.info(f"Loaded {len(hidden_states_list)} prompts")
        
        # Probe all layers
        probing_results = probe_all_layers(model_name, hidden_states_list, labels)
        all_probing_results.extend(probing_results)
        
        # Compute mental state selectivity
        selectivity_results = compute_mental_state_selectivity(hidden_states_list, labels)
        for sr in selectivity_results:
            sr['model'] = model_name
        all_selectivity_results.extend(selectivity_results)
        
        # Find best probing layer
        if probing_results:
            best_layer = max(probing_results, key=lambda x: x['accuracy'])
            layer_info[model_name] = {
                'best_layer': best_layer['layer'],
                'best_accuracy': best_layer['accuracy'],
                'n_layers': metadata[0]['n_layers'],
                'hidden_dim': metadata[0]['hidden_dim'],
            }
    
    # Save results
    if all_probing_results:
        probing_df = pd.DataFrame(all_probing_results)
        probing_df.to_csv(config.TRANSFORMER_ATTENTION_DIR / "probing_results.csv", index=False)
        logger.info(f"Saved probing results: {len(probing_df)} records")
    
    if all_selectivity_results:
        selectivity_df = pd.DataFrame(all_selectivity_results)
        selectivity_df.to_csv(config.TRANSFORMER_ATTENTION_DIR / "layer_selectivity.csv", index=False)
    
    with open(config.TRANSFORMER_ATTENTION_DIR / "layer_information.json", 'w') as f:
        json.dump(layer_info, f, indent=2)
    
    # Compute comparison with brain
    compare_probing_to_brain(probing_df if all_probing_results else pd.DataFrame())
    
    # Print summary
    print_probing_summary(all_probing_results, layer_info)
    
    return all_probing_results, layer_info


def compare_probing_to_brain(probing_df):
    """Compare probing accuracies to brain activation patterns"""
    
    # Load brain activation data
    brain_path = config.BRAIN_ATTENTION_DIR / "brain_activation_rois.csv"
    
    if not brain_path.exists():
        logger.warning("No brain activation data for comparison")
        return
    
    brain_df = pd.read_csv(brain_path)
    
    # Compute brain ToM selectivity (TPJ activation)
    if 'rTPJ_mean' in brain_df.columns:
        brain_tom_activation = brain_df['rTPJ_mean'].mean()
    else:
        brain_tom_activation = 0.0
    
    # Compute correlation between probing accuracy and layer depth
    if len(probing_df) > 0:
        for model in probing_df['model'].unique():
            model_df = probing_df[(probing_df['model'] == model) & (probing_df['pooling'] == 'mean')]
            
            if len(model_df) > 2:
                # Relative layer position (0 to 1)
                model_df = model_df.copy()
                model_df['rel_layer'] = model_df['layer'] / model_df['layer'].max()
                
                # Correlation with layer depth
                r, p = stats.pearsonr(model_df['rel_layer'], model_df['accuracy'])
                
                logger.info(f"{model}: Layer-accuracy correlation r={r:.3f}, p={p:.4f}")
    
    # This is a conceptual comparison - actual integration would require
    # matching prompts to fMRI stimuli
    comparison_data = {
        'brain_tom_activation': float(brain_tom_activation),
        'note': 'Full brain-model comparison requires stimulus-matched data',
    }
    
    with open(config.CROSS_DOMAIN_DIR / "probing_vs_brain.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)


def print_probing_summary(results, layer_info):
    """Print summary of probing results"""
    
    logger.info("\n" + "="*60)
    logger.info("Probing Analysis Summary")
    logger.info("="*60)
    
    for model, info in layer_info.items():
        logger.info(f"\n{model}:")
        logger.info(f"  Best probing layer: {info['best_layer']} / {info['n_layers']}")
        logger.info(f"  Best accuracy: {info['best_accuracy']:.3f}")
        
        # Layer position interpretation
        rel_pos = info['best_layer'] / info['n_layers']
        if rel_pos < 0.33:
            position = "early (syntactic)"
        elif rel_pos < 0.66:
            position = "middle (semantic)"
        else:
            position = "late (task-specific)"
        logger.info(f"  Layer position: {position}")
    
    # Overall interpretation
    if layer_info:
        avg_rel_pos = np.mean([info['best_layer'] / info['n_layers'] for info in layer_info.values()])
        logger.info(f"\nAverage best layer position: {avg_rel_pos:.2f}")
        
        if avg_rel_pos > 0.5:
            logger.info("→ ToM information emerges in later layers (higher-level reasoning)")
        else:
            logger.info("→ ToM information present in earlier layers (lower-level features)")


if __name__ == "__main__":
    results, layer_info = run_probing_analysis()
    print("\n✅ Stage 4 completed: Probing analysis")

