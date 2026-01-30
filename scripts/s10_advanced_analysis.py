#!/usr/bin/env python3
"""
=============================================================================
Stage 10: Advanced Analysis - Mechanistic Interpretability & Geometry
=============================================================================

Advanced analyses for understanding ToM processing in transformers:

1. Representational Geometry Analysis
   - PCA, UMAP, t-SNE of hidden states
   - Manifold structure of ToM representations
   
2. Attention Pattern Analysis
   - Hierarchical clustering of attention heads
   - Attention head specialization profiles
   
3. Information Flow Analysis
   - Attention rollout/flow
   - Layer-wise information transfer
   
4. Cross-Model Comparison
   - Centered Kernel Alignment (CKA)
   - Representational similarity across architectures
   
5. Circuit Discovery
   - Activation patching
   - Component importance scoring

Outputs:
    - geometry_analysis.json
    - attention_clustering.pkl
    - information_flow.json
    - cross_model_cka.csv
    - circuit_components.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, will skip UMAP analysis")

from configs import config

# Ensure directories are initialized (for subprocess runs)
config.ensure_run_directories()

logger = config.setup_logging('advanced_analysis')

# =============================================================================
# 1. REPRESENTATIONAL GEOMETRY ANALYSIS
# =============================================================================

def analyze_representational_geometry(hidden_states_dict):
    """
    Analyze the geometry of hidden state representations.
    
    Uses PCA, t-SNE, and UMAP to understand manifold structure.
    """
    logger.info("\n" + "="*60)
    logger.info("Representational Geometry Analysis")
    logger.info("="*60)
    
    geometry_results = {}
    
    for model_name, data in hidden_states_dict.items():
        logger.info(f"\nProcessing {model_name}...")
        
        try:
            hidden_states = data['hidden_states']  # List of (n_layers, seq_len, hidden_dim)
            labels = data['labels']  # List of label dicts
            
            if len(hidden_states) == 0:
                continue
            
            # Aggregate representations (use [CLS] or last token)
            n_samples = len(hidden_states)
            n_layers = hidden_states[0].shape[0]
            hidden_dim = hidden_states[0].shape[-1]
            
            # Limit hidden_dim to prevent memory issues
            max_hidden_dim = 512  # Reduced further to prevent segfaults
            truncate_dim = hidden_dim > max_hidden_dim
            
            # Extract last token representation from each layer
            layer_representations = {}
            for layer_idx in range(n_layers):
                reps = np.array([hs[layer_idx, -1, :max_hidden_dim] if truncate_dim else hs[layer_idx, -1, :] 
                                for hs in hidden_states])
                layer_representations[layer_idx] = reps
            
            if truncate_dim:
                logger.info(f"  Truncated hidden dim from {hidden_dim} to {max_hidden_dim}")
            
            # Analyze key layers (early, middle, late)
            key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
            key_layers = [l for l in key_layers if l < n_layers]
            
            model_geometry = {
                'n_samples': n_samples,
                'n_layers': n_layers,
                'hidden_dim': hidden_dim,
                'layers_analyzed': key_layers,
                'layer_geometry': {},
            }
            
            for layer_idx in key_layers:
                reps = layer_representations[layer_idx]
                
                if len(reps) < 3:
                    continue
                
                # Handle NaN values
                if np.isnan(reps).any():
                    # Replace NaN with mean of non-NaN values
                    col_means = np.nanmean(reps, axis=0)
                    nan_mask = np.isnan(reps)
                    reps[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                
                # Standardize
                scaler = StandardScaler()
                reps_scaled = scaler.fit_transform(reps)
                
                # Double check for any remaining NaN/Inf
                reps_scaled = np.nan_to_num(reps_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                layer_geo = {}
                
                # PCA analysis
                n_components = min(reps.shape[0] - 1, reps.shape[1], 10)
                if n_components > 1:
                    pca = PCA(n_components=n_components)
                    pca_coords = pca.fit_transform(reps_scaled)
                    
                    layer_geo['pca'] = {
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                        'effective_dimensionality': float(1 / np.sum(pca.explained_variance_ratio_ ** 2)),
                        'coordinates': pca_coords[:, :2].tolist() if pca_coords.shape[1] >= 2 else pca_coords.tolist(),
                    }
                
                # t-SNE (skip for large models to prevent segfaults)
                # if len(reps) >= 5:
                #     perplexity = min(30, len(reps) - 1)
                #     try:
                #         tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                #         tsne_coords = tsne.fit_transform(reps_scaled)
                #         layer_geo['tsne'] = {
                #             'coordinates': tsne_coords.tolist(),
                #             'perplexity': perplexity,
                #         }
                #     except Exception as e:
                #         logger.warning(f"t-SNE failed for layer {layer_idx}: {e}")
                
                # UMAP (skip to prevent memory issues - can cause segfaults with large data)
                # if HAS_UMAP and len(reps) >= 5:
                #     try:
                #         n_neighbors = min(15, len(reps) - 1)
                #         umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
                #         umap_coords = umap.fit_transform(reps_scaled)
                #         layer_geo['umap'] = {
                #             'coordinates': umap_coords.tolist(),
                #             'n_neighbors': n_neighbors,
                #         }
                #     except Exception as e:
                #         logger.warning(f"UMAP failed for layer {layer_idx}: {e}")
                
                # Compute intrinsic dimensionality estimates (simplified)
                if len(reps) >= 10 and reps_scaled.shape[1] < 200:
                    # Two-NN estimator - only for small representations
                    try:
                        # Use simpler estimation based on PCA effective dimensionality
                        if 'pca' in layer_geo:
                            layer_geo['intrinsic_dimensionality'] = layer_geo['pca'].get('effective_dimensionality', 0)
                    except Exception as e:
                        pass
                
                # Class separability (if we have labels)
                tom_labels = [1 if l.get('is_tom', 0) == 1 else 0 for l in labels]
                if len(set(tom_labels)) > 1:
                    # Compute between/within class variance ratio
                    tom_mask = np.array(tom_labels) == 1
                    control_mask = ~tom_mask
                    
                    if tom_mask.sum() > 0 and control_mask.sum() > 0:
                        tom_center = reps_scaled[tom_mask].mean(axis=0)
                        control_center = reps_scaled[control_mask].mean(axis=0)
                        
                        between_var = np.linalg.norm(tom_center - control_center) ** 2
                        within_var_tom = np.mean([np.linalg.norm(r - tom_center) ** 2 for r in reps_scaled[tom_mask]])
                        within_var_control = np.mean([np.linalg.norm(r - control_center) ** 2 for r in reps_scaled[control_mask]])
                        within_var = (within_var_tom + within_var_control) / 2
                        
                        layer_geo['class_separability'] = {
                            'between_class_variance': float(between_var),
                            'within_class_variance': float(within_var),
                            'fisher_ratio': float(between_var / (within_var + 1e-10)),
                        }
                
                model_geometry['layer_geometry'][layer_idx] = layer_geo
            
            geometry_results[model_name] = model_geometry
            logger.info(f"  Analyzed {len(key_layers)} layers")
            
            # Free memory
            del layer_representations
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            # Continue with other models
            continue
    
    return geometry_results


# =============================================================================
# 2. ATTENTION PATTERN ANALYSIS
# =============================================================================

def analyze_attention_patterns(attention_weights_dict):
    """
    Analyze attention patterns using hierarchical clustering.
    
    Discovers attention head types and specialization profiles.
    """
    logger.info("\n" + "="*60)
    logger.info("Attention Pattern Analysis")
    logger.info("="*60)
    
    clustering_results = {}
    
    for model_name, data in attention_weights_dict.items():
        logger.info(f"\nProcessing {model_name}...")
        
        attention_weights = data['attention_weights']  # List of (n_layers, n_heads, seq_len, seq_len)
        
        if len(attention_weights) == 0:
            continue
        
        n_layers = attention_weights[0].shape[0]
        n_heads = attention_weights[0].shape[1]
        
        # Compute attention pattern signatures for each head
        head_signatures = []
        head_labels = []
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                try:
                    # Average attention pattern across all prompts
                    # Handle variable sequence lengths by extracting features first
                    all_features = []
                    for aw in attention_weights:
                        pattern = aw[layer_idx, head_idx]
                        features = extract_attention_features(pattern)
                        all_features.append(features)
                    
                    # Average features across prompts
                    avg_features = np.mean(all_features, axis=0)
                    head_signatures.append(avg_features)
                    head_labels.append({'layer': layer_idx, 'head': head_idx})
                except Exception as e:
                    logger.warning(f"Error processing layer {layer_idx} head {head_idx}: {e}")
                    continue
        
        head_signatures = np.array(head_signatures)
        
        # Handle any NaN/Inf in signatures
        head_signatures = np.nan_to_num(head_signatures, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Hierarchical clustering
        if len(head_signatures) > 1:
            # Compute pairwise distances
            distances = pdist(head_signatures, metric='cosine')
            
            # Handle any NaN/Inf in distances
            distances = np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Hierarchical clustering
            linkage_matrix = linkage(distances, method='ward')
            
            # Determine optimal number of clusters
            n_clusters = min(8, len(head_signatures) // 2)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Analyze cluster characteristics
            cluster_profiles = {}
            for c in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == c
                cluster_heads = [head_labels[i] for i in range(len(head_labels)) if cluster_mask[i]]
                cluster_features = head_signatures[cluster_mask]
                
                # Compute cluster statistics
                feature_names = ['entropy', 'sparsity', 'diagonal_attention', 'first_token_attention',
                               'local_attention', 'global_attention']
                
                cluster_profiles[c] = {
                    'n_heads': int(cluster_mask.sum()),
                    'heads': cluster_heads,
                    'mean_features': {name: float(cluster_features[:, i].mean()) 
                                     for i, name in enumerate(feature_names) if i < cluster_features.shape[1]},
                    'layer_distribution': get_layer_distribution(cluster_heads, n_layers),
                }
            
            clustering_results[model_name] = {
                'n_layers': n_layers,
                'n_heads': n_heads,
                'n_clusters': n_clusters,
                'linkage_matrix': linkage_matrix.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'head_labels': head_labels,
                'cluster_profiles': cluster_profiles,
            }
            
            logger.info(f"  Found {n_clusters} attention head types")
    
    return clustering_results


def extract_attention_features(attention_pattern):
    """Extract interpretable features from attention pattern."""
    seq_len = attention_pattern.shape[0]
    
    features = []
    
    # 1. Entropy (how spread out is attention)
    flat_attn = attention_pattern.flatten()
    flat_attn = flat_attn[flat_attn > 0]
    if len(flat_attn) > 0:
        entropy = -np.sum(flat_attn * np.log(flat_attn + 1e-10))
        entropy_normalized = entropy / np.log(len(flat_attn) + 1)
    else:
        entropy_normalized = 0
    features.append(entropy_normalized)
    
    # 2. Sparsity (what fraction of attention is concentrated)
    top_10_percent = np.percentile(attention_pattern.flatten(), 90)
    sparsity = np.mean(attention_pattern.flatten() >= top_10_percent)
    features.append(sparsity)
    
    # 3. Diagonal attention (self-attention)
    if seq_len > 1:
        diagonal = np.diag(attention_pattern)
        diagonal_attention = np.mean(diagonal)
    else:
        diagonal_attention = 1.0
    features.append(diagonal_attention)
    
    # 4. First token attention (BOS/CLS attention)
    first_token_attention = np.mean(attention_pattern[:, 0])
    features.append(first_token_attention)
    
    # 5. Local attention (nearby tokens)
    local_window = 3
    local_attention = 0
    for i in range(seq_len):
        start = max(0, i - local_window)
        end = min(seq_len, i + local_window + 1)
        local_attention += np.sum(attention_pattern[i, start:end])
    local_attention /= (seq_len * (2 * local_window + 1))
    features.append(local_attention)
    
    # 6. Global attention (distant tokens)
    global_attention = 1 - local_attention
    features.append(global_attention)
    
    return np.array(features)


def get_layer_distribution(heads, n_layers):
    """Get distribution of heads across layer thirds."""
    early = sum(1 for h in heads if h['layer'] < n_layers // 3)
    middle = sum(1 for h in heads if n_layers // 3 <= h['layer'] < 2 * n_layers // 3)
    late = sum(1 for h in heads if h['layer'] >= 2 * n_layers // 3)
    total = len(heads)
    
    return {
        'early': early / total if total > 0 else 0,
        'middle': middle / total if total > 0 else 0,
        'late': late / total if total > 0 else 0,
    }


# =============================================================================
# 3. INFORMATION FLOW ANALYSIS
# =============================================================================

def analyze_information_flow(attention_weights_dict, hidden_states_dict):
    """
    Analyze information flow through the network.
    
    Uses attention rollout to track how information propagates.
    """
    logger.info("\n" + "="*60)
    logger.info("Information Flow Analysis")
    logger.info("="*60)
    
    flow_results = {}
    
    for model_name in attention_weights_dict.keys():
        if model_name not in hidden_states_dict:
            continue
        
        logger.info(f"\nProcessing {model_name}...")
        
        attn_data = attention_weights_dict[model_name]
        hidden_data = hidden_states_dict[model_name]
        
        attention_weights = attn_data['attention_weights']
        hidden_states = hidden_data['hidden_states']
        labels = hidden_data['labels']
        
        if len(attention_weights) == 0:
            continue
        
        n_layers = attention_weights[0].shape[0]
        
        # Compute attention rollout for each prompt
        rollout_results = []
        
        for idx, aw in enumerate(attention_weights):
            # Attention rollout: multiply attention matrices
            rollout = compute_attention_rollout(aw)
            
            # Get label
            is_tom = labels[idx].get('is_tom', 0) if idx < len(labels) else 0
            
            rollout_results.append({
                'prompt_idx': idx,
                'is_tom': is_tom,
                'final_rollout': rollout.tolist(),
            })
        
        # Compute layer-wise information change
        layer_info = []
        for layer_idx in range(n_layers):
            # Average attention entropy at this layer
            layer_entropies = []
            for aw in attention_weights:
                layer_attn = aw[layer_idx]  # (n_heads, seq_len, seq_len)
                avg_attn = layer_attn.mean(axis=0)  # Average over heads
                
                # Compute entropy for each position
                for pos in range(avg_attn.shape[0]):
                    attn_dist = avg_attn[pos]
                    attn_dist = attn_dist[attn_dist > 0]
                    if len(attn_dist) > 0:
                        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                        layer_entropies.append(entropy)
            
            layer_info.append({
                'layer': layer_idx,
                'mean_attention_entropy': float(np.mean(layer_entropies)) if layer_entropies else 0,
                'std_attention_entropy': float(np.std(layer_entropies)) if layer_entropies else 0,
            })
        
        flow_results[model_name] = {
            'n_layers': n_layers,
            'n_prompts': len(attention_weights),
            'rollout_samples': rollout_results[:5],  # Save first 5 examples
            'layer_information': layer_info,
        }
        
        logger.info(f"  Computed attention rollout for {len(attention_weights)} prompts")
    
    return flow_results


def compute_attention_rollout(attention_weights):
    """
    Compute attention rollout following Abnar & Zuidema (2020).
    
    Args:
        attention_weights: (n_layers, n_heads, seq_len, seq_len)
    
    Returns:
        rollout: (seq_len, seq_len) final attention distribution
    """
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    # Average attention across heads
    attn_avg = attention_weights.mean(axis=1)  # (n_layers, seq_len, seq_len)
    
    # Add residual connection (identity)
    residual_attn = np.zeros_like(attn_avg)
    for i in range(n_layers):
        residual_attn[i] = 0.5 * attn_avg[i] + 0.5 * np.eye(seq_len)
    
    # Rollout: multiply attention matrices
    rollout = residual_attn[0]
    for i in range(1, n_layers):
        rollout = np.matmul(residual_attn[i], rollout)
    
    # Normalize rows
    rollout = rollout / (rollout.sum(axis=-1, keepdims=True) + 1e-10)
    
    return rollout


# =============================================================================
# 4. CROSS-MODEL COMPARISON
# =============================================================================

def compare_models_cka(hidden_states_dict):
    """
    Compare representations across models using CKA.
    
    Centered Kernel Alignment measures similarity of representations.
    """
    logger.info("\n" + "="*60)
    logger.info("Cross-Model Comparison (CKA)")
    logger.info("="*60)
    
    model_names = list(hidden_states_dict.keys())
    
    if len(model_names) < 2:
        logger.warning("Need at least 2 models for comparison")
        return {}
    
    cka_results = []
    
    # Compare each pair of models
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue
            
            logger.info(f"\nComparing {model1} vs {model2}...")
            
            data1 = hidden_states_dict[model1]
            data2 = hidden_states_dict[model2]
            
            # Get representations at final layer
            hs1 = data1['hidden_states']
            hs2 = data2['hidden_states']
            
            # Need same number of samples
            n_samples = min(len(hs1), len(hs2))
            
            if n_samples < 3:
                continue
            
            # Extract final layer representations
            reps1 = np.array([hs[hs.shape[0] // 2, -1, :] for hs in hs1[:n_samples]])  # Middle layer
            reps2 = np.array([hs[hs.shape[0] // 2, -1, :] for hs in hs2[:n_samples]])
            
            # Compute CKA
            cka_value = compute_cka(reps1, reps2)
            
            cka_results.append({
                'model1': model1,
                'model2': model2,
                'cka': float(cka_value),
                'n_samples': n_samples,
            })
            
            logger.info(f"  CKA = {cka_value:.4f}")
    
    # Also compare layers within models
    layer_cka = {}
    for model_name, data in hidden_states_dict.items():
        hs = data['hidden_states']
        if len(hs) < 3:
            continue
        
        n_layers = hs[0].shape[0]
        
        # Sample layers to compare
        layers_to_compare = list(range(0, n_layers, max(1, n_layers // 8)))
        
        layer_cka_matrix = np.zeros((len(layers_to_compare), len(layers_to_compare)))
        
        for i, l1 in enumerate(layers_to_compare):
            for j, l2 in enumerate(layers_to_compare):
                if i > j:
                    layer_cka_matrix[i, j] = layer_cka_matrix[j, i]
                    continue
                
                reps1 = np.array([h[l1, -1, :] for h in hs])
                reps2 = np.array([h[l2, -1, :] for h in hs])
                
                layer_cka_matrix[i, j] = compute_cka(reps1, reps2)
        
        layer_cka[model_name] = {
            'layers': layers_to_compare,
            'cka_matrix': layer_cka_matrix.tolist(),
        }
    
    return {
        'cross_model_cka': cka_results,
        'layer_cka': layer_cka,
    }


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment between two representation matrices.
    
    Args:
        X: (n_samples, dim1)
        Y: (n_samples, dim2)
    
    Returns:
        CKA similarity (0-1)
    """
    # Center the representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    
    # Center the Gram matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC
    hsic_kl = np.sum(K_centered * L_centered)
    hsic_kk = np.sum(K_centered * K_centered)
    hsic_ll = np.sum(L_centered * L_centered)
    
    # CKA
    cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)
    
    return cka


# =============================================================================
# 5. CIRCUIT DISCOVERY
# =============================================================================

def discover_circuits(attention_weights_dict, hidden_states_dict):
    """
    Discover important circuits for ToM processing.
    
    Identifies which components are most important for task performance.
    """
    logger.info("\n" + "="*60)
    logger.info("Circuit Discovery")
    logger.info("="*60)
    
    circuit_results = {}
    
    for model_name in attention_weights_dict.keys():
        if model_name not in hidden_states_dict:
            continue
        
        logger.info(f"\nProcessing {model_name}...")
        
        attn_data = attention_weights_dict[model_name]
        hidden_data = hidden_states_dict[model_name]
        
        attention_weights = attn_data['attention_weights']
        hidden_states = hidden_data['hidden_states']
        labels = hidden_data['labels']
        
        if len(attention_weights) == 0:
            continue
        
        n_layers = attention_weights[0].shape[0]
        n_heads = attention_weights[0].shape[1]
        
        # Compute head importance scores
        head_importance = np.zeros((n_layers, n_heads))
        
        # Use attention difference between ToM and control as importance
        tom_indices = [i for i, l in enumerate(labels) if l.get('is_tom', 0) == 1]
        control_indices = [i for i, l in enumerate(labels) if l.get('is_tom', 0) == 0]
        
        if len(tom_indices) > 0 and len(control_indices) > 0:
            for layer in range(n_layers):
                for head in range(n_heads):
                    try:
                        # Compute entropy for each pattern (handles variable lengths)
                        tom_entropies = []
                        for i in tom_indices:
                            pattern = attention_weights[i][layer, head]
                            entropy = -np.sum(pattern * np.log(pattern + 1e-10)) / (pattern.shape[0] + 1e-10)
                            tom_entropies.append(entropy)
                        
                        control_entropies = []
                        for i in control_indices:
                            pattern = attention_weights[i][layer, head]
                            entropy = -np.sum(pattern * np.log(pattern + 1e-10)) / (pattern.shape[0] + 1e-10)
                            control_entropies.append(entropy)
                        
                        # Importance = difference in average entropy
                        importance = abs(np.mean(tom_entropies) - np.mean(control_entropies))
                        head_importance[layer, head] = importance
                    except Exception as e:
                        head_importance[layer, head] = 0.0
        else:
            # Use attention entropy as proxy for importance
            for layer in range(n_layers):
                for head in range(n_heads):
                    try:
                        entropies = []
                        for aw in attention_weights:
                            pattern = aw[layer, head]
                            entropy = -np.sum(pattern * np.log(pattern + 1e-10)) / (pattern.shape[0] + 1e-10)
                            entropies.append(entropy)
                        
                        avg_entropy = np.mean(entropies)
                        # Lower entropy = more focused = potentially more important
                        head_importance[layer, head] = 1 / (avg_entropy + 1)
                    except Exception as e:
                        head_importance[layer, head] = 0.0
        
        # Identify top components
        flat_importance = head_importance.flatten()
        threshold = np.percentile(flat_importance, 90)
        
        important_heads = []
        for layer in range(n_layers):
            for head in range(n_heads):
                if head_importance[layer, head] >= threshold:
                    important_heads.append({
                        'layer': int(layer),
                        'head': int(head),
                        'importance': float(head_importance[layer, head]),
                    })
        
        # Sort by importance
        important_heads.sort(key=lambda x: x['importance'], reverse=True)
        
        # Identify circuit pathway
        layer_importance = head_importance.sum(axis=1)
        layer_importance_norm = layer_importance / layer_importance.max()
        
        circuit_results[model_name] = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'head_importance_matrix': head_importance.tolist(),
            'top_heads': important_heads[:20],
            'layer_importance': layer_importance_norm.tolist(),
            'circuit_size': len(important_heads),
        }
        
        logger.info(f"  Found {len(important_heads)} important heads (top 10%)")
    
    return circuit_results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load hidden states and attention weights for all models."""
    
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "attention_weights"
    hidden_dir = config.TRANSFORMER_ATTENTION_DIR / "hidden_states"
    
    attention_weights_dict = {}
    hidden_states_dict = {}
    
    # Load for each model
    for model_subdir in attention_dir.iterdir():
        if not model_subdir.is_dir():
            continue
        
        model_name = model_subdir.name.replace('_', '-')
        
        # Load attention weights
        attention_weights = []
        for pkl_file in sorted(model_subdir.glob("prompt_*.pkl")):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            attention_weights.append(data['attention_weights'])
        
        if attention_weights:
            attention_weights_dict[model_name] = {
                'attention_weights': attention_weights,
            }
    
    for model_subdir in hidden_dir.iterdir():
        if not model_subdir.is_dir():
            continue
        
        model_name = model_subdir.name.replace('_', '-')
        
        # Load hidden states
        hidden_states = []
        labels = []
        
        for pkl_file in sorted(model_subdir.glob("prompt_*.pkl")):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            hidden_states.append(data['hidden_states'])
            
            # Try to get label from attention file
            prompt_idx = data.get('prompt_idx', len(labels))
            attn_file = attention_dir / model_subdir.name / f"prompt_{prompt_idx}.pkl"
            if attn_file.exists():
                with open(attn_file, 'rb') as f:
                    attn_data = pickle.load(f)
                labels.append({
                    'is_tom': 1 if attn_data.get('category', '') == 'mental_intentional' else 0,
                    'category': attn_data.get('category', 'unknown'),
                })
            else:
                labels.append({'is_tom': 0, 'category': 'unknown'})
        
        if hidden_states:
            hidden_states_dict[model_name] = {
                'hidden_states': hidden_states,
                'labels': labels,
            }
    
    return attention_weights_dict, hidden_states_dict


# =============================================================================
# MAIN
# =============================================================================

def run_advanced_analysis():
    """Run all advanced analyses."""
    
    logger.info("="*60)
    logger.info("Stage 10: Advanced Analysis")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    attention_weights_dict, hidden_states_dict = load_all_data()
    
    logger.info(f"Loaded attention weights for {len(attention_weights_dict)} models")
    logger.info(f"Loaded hidden states for {len(hidden_states_dict)} models")
    
    results = {}
    
    # 1. Representational Geometry
    if hidden_states_dict:
        geometry_results = analyze_representational_geometry(hidden_states_dict)
        results['geometry'] = geometry_results
        
        with open(config.CROSS_DOMAIN_DIR / "geometry_analysis.json", 'w') as f:
            json.dump(geometry_results, f, indent=2)
    
    # 2. Attention Patterns
    if attention_weights_dict:
        clustering_results = analyze_attention_patterns(attention_weights_dict)
        results['attention_clustering'] = clustering_results
        
        with open(config.CROSS_DOMAIN_DIR / "attention_clustering.pkl", 'wb') as f:
            pickle.dump(clustering_results, f)
    
    # 3. Information Flow
    if attention_weights_dict and hidden_states_dict:
        flow_results = analyze_information_flow(attention_weights_dict, hidden_states_dict)
        results['information_flow'] = flow_results
        
        with open(config.CROSS_DOMAIN_DIR / "information_flow.json", 'w') as f:
            json.dump(flow_results, f, indent=2)
    
    # 4. Cross-Model Comparison
    if len(hidden_states_dict) >= 2:
        cka_results = compare_models_cka(hidden_states_dict)
        results['cka'] = cka_results
        
        with open(config.CROSS_DOMAIN_DIR / "cross_model_cka.json", 'w') as f:
            json.dump(cka_results, f, indent=2)
    
    # 5. Circuit Discovery
    if attention_weights_dict and hidden_states_dict:
        circuit_results = discover_circuits(attention_weights_dict, hidden_states_dict)
        results['circuits'] = circuit_results
        
        with open(config.CROSS_DOMAIN_DIR / "circuit_components.json", 'w') as f:
            json.dump(circuit_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Advanced Analysis Summary")
    logger.info("="*60)
    
    if 'geometry' in results:
        logger.info(f"\nGeometry Analysis: {len(results['geometry'])} models")
    if 'attention_clustering' in results:
        logger.info(f"Attention Clustering: {len(results['attention_clustering'])} models")
    if 'information_flow' in results:
        logger.info(f"Information Flow: {len(results['information_flow'])} models")
    if 'cka' in results:
        logger.info(f"Cross-Model CKA: {len(results['cka'].get('cross_model_cka', []))} comparisons")
    if 'circuits' in results:
        logger.info(f"Circuit Discovery: {len(results['circuits'])} models")
    
    return results


if __name__ == "__main__":
    results = run_advanced_analysis()
    print(f"\n✅ Stage 10 completed: Advanced analysis")

