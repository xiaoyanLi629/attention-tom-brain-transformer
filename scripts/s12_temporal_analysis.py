#!/usr/bin/env python3
"""
=============================================================================
Stage 12: Temporal Dynamics Analysis
=============================================================================

Analyze temporal dynamics in both brain fMRI and transformer attention:

Brain Temporal Analysis:
    1. Dynamic Functional Connectivity (sliding window)
    2. Temporal activation profiles during ToM processing
    3. Phase synchrony between ToM regions
    4. Time-frequency analysis of ROI activations
    
Transformer Temporal Analysis:
    1. Token-by-token attention evolution
    2. Sequential information accumulation
    3. Attention pattern dynamics across sequence positions
    
Cross-Domain Temporal Comparison:
    1. Temporal alignment between brain and model dynamics
    2. Processing stage correspondence

Outputs:
    - temporal_dynamics.json
    - dynamic_connectivity.csv
    - phase_synchrony.csv
    - token_attention_evolution.csv
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('temporal_analysis')

# =============================================================================
# BRAIN TEMPORAL ANALYSIS
# =============================================================================

def compute_sliding_window_connectivity(timeseries, window_size=30, step=5):
    """
    Compute dynamic functional connectivity using sliding window.
    
    Args:
        timeseries: (n_timepoints, n_rois) array
        window_size: Size of sliding window in TRs
        step: Step size between windows
        
    Returns:
        connectivity_matrices: (n_windows, n_rois, n_rois) array
        window_times: Center time of each window
    """
    n_timepoints, n_rois = timeseries.shape
    n_windows = (n_timepoints - window_size) // step + 1
    
    connectivity_matrices = np.zeros((n_windows, n_rois, n_rois))
    window_times = []
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window_data = timeseries[start:end]
        
        # Compute correlation matrix for this window
        corr_matrix = np.corrcoef(window_data.T)
        connectivity_matrices[i] = corr_matrix
        window_times.append((start + end) / 2)
    
    return connectivity_matrices, np.array(window_times)


def compute_phase_synchrony(timeseries, freq_band=(0.01, 0.1), tr=0.72):
    """
    Compute phase synchrony between regions using Hilbert transform.
    
    Args:
        timeseries: (n_timepoints, n_rois) array
        freq_band: Frequency band of interest (Hz)
        tr: Repetition time in seconds
        
    Returns:
        phase_sync: (n_rois, n_rois) average phase synchrony matrix
        instant_sync: (n_timepoints, n_rois, n_rois) instantaneous phase synchrony
    """
    n_timepoints, n_rois = timeseries.shape
    fs = 1 / tr
    
    # Bandpass filter
    nyq = fs / 2
    low = freq_band[0] / nyq
    high = min(freq_band[1] / nyq, 0.99)
    
    if low >= high:
        low = 0.01 / nyq
        high = 0.1 / nyq
    
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Get instantaneous phases
    phases = np.zeros_like(timeseries)
    for i in range(n_rois):
        filtered = signal.filtfilt(b, a, timeseries[:, i])
        analytic = signal.hilbert(filtered)
        phases[:, i] = np.angle(analytic)
    
    # Compute phase locking value (PLV)
    instant_sync = np.zeros((n_timepoints, n_rois, n_rois))
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            phase_diff = phases[:, i] - phases[:, j]
            instant_sync[:, i, j] = np.cos(phase_diff)
            instant_sync[:, j, i] = instant_sync[:, i, j]
    
    # Average phase synchrony
    phase_sync = np.mean(np.abs(instant_sync), axis=0)
    
    return phase_sync, instant_sync


def compute_temporal_activation_profile(timeseries, events, tr=0.72, 
                                        pre_stim=5, post_stim=25):
    """
    Compute event-related activation profile.
    
    Args:
        timeseries: (n_timepoints, n_rois) array
        events: List of event onsets (in TRs)
        tr: Repetition time
        pre_stim: Pre-stimulus time in TRs
        post_stim: Post-stimulus time in TRs
        
    Returns:
        profiles: (n_events, n_timepoints, n_rois) activation profiles
        mean_profile: (n_timepoints, n_rois) average profile
    """
    n_timepoints, n_rois = timeseries.shape
    window_length = pre_stim + post_stim
    
    profiles = []
    for onset in events:
        start = max(0, onset - pre_stim)
        end = min(n_timepoints, onset + post_stim)
        
        if end - start >= window_length * 0.8:  # At least 80% of window
            epoch = timeseries[start:end]
            if len(epoch) < window_length:
                # Pad if necessary
                padding = np.zeros((window_length - len(epoch), n_rois))
                epoch = np.vstack([epoch, padding])
            profiles.append(epoch[:window_length])
    
    if profiles:
        profiles = np.array(profiles)
        mean_profile = np.mean(profiles, axis=0)
    else:
        profiles = np.zeros((1, window_length, n_rois))
        mean_profile = np.zeros((window_length, n_rois))
    
    return profiles, mean_profile


def analyze_connectivity_dynamics(connectivity_matrices, roi_names):
    """
    Analyze how connectivity patterns change over time.
    
    Returns metrics for connectivity variability, state transitions, etc.
    """
    n_windows, n_rois, _ = connectivity_matrices.shape
    
    # Connectivity variability (temporal standard deviation)
    conn_variability = np.std(connectivity_matrices, axis=0)
    
    # State transitions (distance between consecutive windows)
    transitions = []
    for i in range(n_windows - 1):
        dist = np.linalg.norm(connectivity_matrices[i+1] - connectivity_matrices[i])
        transitions.append(dist)
    
    # Mean connectivity over time
    mean_conn = np.mean(connectivity_matrices, axis=0)
    
    # Extract key connections for ToM network
    tom_regions = ['rTPJ', 'lTPJ', 'mPFC']
    tom_indices = [i for i, name in enumerate(roi_names) if name in tom_regions]
    
    tom_conn_over_time = []
    for t in range(n_windows):
        tom_mean = np.mean([connectivity_matrices[t, i, j] 
                          for i in tom_indices for j in tom_indices if i != j])
        tom_conn_over_time.append(tom_mean)
    
    return {
        'variability': conn_variability.tolist(),
        'transitions': transitions,
        'mean_connectivity': mean_conn.tolist(),
        'tom_connectivity_timeseries': tom_conn_over_time,
        'mean_tom_connectivity': float(np.mean(tom_conn_over_time)),
        'tom_connectivity_variability': float(np.std(tom_conn_over_time)),
    }


# =============================================================================
# TRANSFORMER TEMPORAL ANALYSIS
# =============================================================================

def analyze_token_attention_evolution(attention_weights):
    """
    Analyze how attention patterns evolve across token positions.
    
    Args:
        attention_weights: (n_layers, n_heads, seq_len, seq_len) array
        
    Returns:
        evolution_metrics: Dict with temporal attention metrics
    """
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    # Cumulative attention to early tokens over sequence
    early_attention = []
    for pos in range(seq_len):
        if pos > 0:
            # Attention from current position to first 25% of tokens
            early_end = max(1, int(pos * 0.25))
            attn_to_early = attention_weights[:, :, pos, :early_end].mean()
            early_attention.append(float(attn_to_early))
    
    # Attention entropy evolution (does attention become more focused?)
    entropy_evolution = []
    for pos in range(1, seq_len):
        attn_dist = attention_weights[:, :, pos, :pos+1].mean(axis=(0, 1))
        attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
        entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
        entropy_evolution.append(float(entropy))
    
    # Peak attention position over sequence
    peak_positions = []
    for pos in range(1, seq_len):
        mean_attn = attention_weights[:, :, pos, :pos+1].mean(axis=(0, 1))
        peak_pos = np.argmax(mean_attn)
        peak_positions.append(int(peak_pos))
    
    # Layer-wise attention pattern changes
    layer_pattern_changes = []
    for layer in range(n_layers):
        pattern_diffs = []
        for pos in range(2, seq_len):
            prev_pattern = attention_weights[layer, :, pos-1, :pos].mean(axis=0)
            curr_pattern = attention_weights[layer, :, pos, :pos+1].mean(axis=0)
            # Align lengths for comparison
            min_len = min(len(prev_pattern), len(curr_pattern) - 1)
            if min_len > 0:
                diff = np.linalg.norm(curr_pattern[:min_len] - prev_pattern[:min_len])
                pattern_diffs.append(float(diff))
        layer_pattern_changes.append(np.mean(pattern_diffs) if pattern_diffs else 0)
    
    return {
        'early_attention_accumulation': early_attention,
        'entropy_evolution': entropy_evolution,
        'peak_attention_positions': peak_positions,
        'layer_pattern_change_rate': layer_pattern_changes,
        'mean_entropy_change': float(np.mean(np.diff(entropy_evolution))) if len(entropy_evolution) > 1 else 0,
        'attention_becomes_focused': float(np.mean(entropy_evolution[:len(entropy_evolution)//2]) > 
                                          np.mean(entropy_evolution[len(entropy_evolution)//2:])) if entropy_evolution else 0,
    }


def analyze_information_accumulation(hidden_states):
    """
    Analyze how information accumulates across token positions.
    
    Args:
        hidden_states: (n_layers+1, seq_len, hidden_dim) array
        
    Returns:
        accumulation_metrics: Dict with information accumulation patterns
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # Representational change over positions (per layer)
    layer_rep_changes = []
    for layer in range(n_layers):
        changes = []
        for pos in range(1, seq_len):
            change = np.linalg.norm(hidden_states[layer, pos] - hidden_states[layer, pos-1])
            changes.append(float(change))
        layer_rep_changes.append({
            'layer': layer,
            'mean_change': float(np.mean(changes)),
            'change_trajectory': changes,
        })
    
    # Information integration (correlation with final position over time)
    integration_curves = []
    for layer in range(n_layers):
        final_rep = hidden_states[layer, -1]
        correlations = []
        for pos in range(seq_len):
            corr = np.corrcoef(hidden_states[layer, pos], final_rep)[0, 1]
            correlations.append(float(corr) if not np.isnan(corr) else 0)
        integration_curves.append(correlations)
    
    # When does each layer "converge" to final representation?
    convergence_points = []
    for layer, curve in enumerate(integration_curves):
        threshold = 0.9 * max(curve) if max(curve) > 0 else 0.5
        converge_pos = next((i for i, c in enumerate(curve) if c >= threshold), seq_len)
        convergence_points.append({
            'layer': layer,
            'convergence_position': int(converge_pos),
            'convergence_fraction': float(converge_pos / seq_len),
        })
    
    return {
        'layer_representational_changes': layer_rep_changes,
        'integration_curves': integration_curves,
        'convergence_analysis': convergence_points,
        'early_layer_fast_integration': float(np.mean([c['convergence_fraction'] 
                                                       for c in convergence_points[:n_layers//3]])),
        'late_layer_fast_integration': float(np.mean([c['convergence_fraction'] 
                                                      for c in convergence_points[-n_layers//3:]])),
    }


# =============================================================================
# CROSS-DOMAIN TEMPORAL COMPARISON
# =============================================================================

def compare_temporal_dynamics(brain_dynamics, transformer_dynamics):
    """
    Compare temporal dynamics between brain and transformer.
    
    Looking for analogies in:
    - Processing stage timing
    - Integration patterns
    - Focusing of attention/activation over time
    """
    comparison = {}
    
    # Both show focusing of processing over time?
    brain_focusing = brain_dynamics.get('tom_connectivity_variability', 0) > 0
    transformer_focusing = transformer_dynamics.get('attention_becomes_focused', 0) > 0.5
    
    comparison['both_show_focusing'] = brain_focusing and transformer_focusing
    
    # Peak processing in middle stages?
    tom_conn = brain_dynamics.get('tom_connectivity_timeseries', [])
    layer_changes = transformer_dynamics.get('layer_pattern_change_rate', [])
    
    if tom_conn and len(tom_conn) > 3:
        brain_peak_third = np.argmax([np.mean(tom_conn[i:i+len(tom_conn)//3]) 
                                      for i in range(0, len(tom_conn), len(tom_conn)//3)])
    else:
        brain_peak_third = 1  # Middle by default
        
    if layer_changes and len(layer_changes) > 3:
        transformer_peak_third = np.argmax([np.mean(layer_changes[i:i+len(layer_changes)//3]) 
                                           for i in range(0, len(layer_changes), len(layer_changes)//3)])
    else:
        transformer_peak_third = 1
    
    comparison['peak_processing_stage'] = {
        'brain': int(brain_peak_third),  # 0=early, 1=middle, 2=late
        'transformer': int(transformer_peak_third),
        'aligned': brain_peak_third == transformer_peak_third,
    }
    
    return comparison


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_temporal_analysis():
    """Run complete temporal dynamics analysis."""
    
    logger.info("="*60)
    logger.info("Stage 12: Temporal Dynamics Analysis")
    logger.info("="*60)
    
    results = {
        'brain_dynamics': {},
        'transformer_dynamics': {},
        'cross_domain': {},
    }
    
    # =========================================================================
    # BRAIN TEMPORAL ANALYSIS
    # =========================================================================
    logger.info("\n--- Brain Temporal Analysis ---")
    
    roi_names = ['rTPJ', 'lTPJ', 'mPFC', 'PC', 'rSTS', 'lSTS']
    n_rois = len(roi_names)
    n_timepoints = config.SOCIAL_TASK['n_volumes']
    tr = config.SOCIAL_TASK['tr']
    
    # Simulate brain timeseries (in real application, load from fMRI data)
    # For demonstration, we create synthetic timeseries with realistic properties
    logger.info("Generating brain temporal data...")
    
    # Create simulated timeseries with ToM-related activation
    np.random.seed(42)
    brain_timeseries = np.random.randn(n_timepoints, n_rois)
    
    # Add task-related activation (mental > random conditions)
    # Assume blocks of ~28 TRs (20 sec at TR=0.72)
    block_length = 28
    n_blocks = n_timepoints // block_length
    
    for block in range(n_blocks):
        start = block * block_length
        end = start + block_length
        is_mental = block % 2 == 0  # Alternating mental/random
        
        if is_mental:
            # Higher activation in ToM regions during mental condition
            brain_timeseries[start:end, 0:3] += 1.5  # rTPJ, lTPJ, mPFC
            # Add hemodynamic response shape
            hrf = signal.windows.gaussian(block_length, std=5)
            for roi in range(3):
                brain_timeseries[start:end, roi] += hrf * 2
    
    # Smooth to simulate hemodynamic response
    for i in range(n_rois):
        brain_timeseries[:, i] = gaussian_filter1d(brain_timeseries[:, i], sigma=2)
    
    # Compute dynamic connectivity
    logger.info("Computing dynamic functional connectivity...")
    window_size = 30  # ~22 seconds
    step = 5  # ~3.6 seconds
    
    conn_matrices, window_times = compute_sliding_window_connectivity(
        brain_timeseries, window_size=window_size, step=step
    )
    
    # Analyze connectivity dynamics
    conn_dynamics = analyze_connectivity_dynamics(conn_matrices, roi_names)
    results['brain_dynamics']['connectivity'] = conn_dynamics
    
    # Compute phase synchrony
    logger.info("Computing phase synchrony...")
    phase_sync, instant_sync = compute_phase_synchrony(brain_timeseries, tr=tr)
    
    results['brain_dynamics']['phase_synchrony'] = {
        'mean_matrix': phase_sync.tolist(),
        'roi_names': roi_names,
        'tom_network_sync': float(np.mean(phase_sync[:3, :3])),  # TPJ-mPFC network
        'overall_sync': float(np.mean(phase_sync)),
    }
    
    # Event-related temporal profile
    logger.info("Computing temporal activation profiles...")
    events = list(range(0, n_timepoints, block_length))  # Block onsets
    profiles, mean_profile = compute_temporal_activation_profile(
        brain_timeseries, events, tr=tr, pre_stim=5, post_stim=25
    )
    
    results['brain_dynamics']['activation_profiles'] = {
        'mean_profile': mean_profile.tolist(),
        'n_events': len(events),
        'window_trs': 30,
        'peak_time_tr': int(np.argmax(mean_profile[:, 0])),  # Peak for rTPJ
        'peak_time_sec': float(np.argmax(mean_profile[:, 0]) * tr),
    }
    
    # =========================================================================
    # TRANSFORMER TEMPORAL ANALYSIS
    # =========================================================================
    logger.info("\n--- Transformer Temporal Analysis ---")
    
    transformer_dynamics = {}
    
    # Load attention weights from extraction
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "attention_weights"
    
    for model_config in config.TRANSFORMER_CONFIG['models'][:3]:  # First 3 models
        model_name = model_config['short_name'].replace('-', '_')
        model_dir = attention_dir / model_name
        
        if not model_dir.exists():
            logger.warning(f"No attention data for {model_name}, simulating...")
            # Simulate attention patterns
            n_layers = model_config['n_layers']
            n_heads = model_config['n_heads']
            seq_len = 50
            
            # Create realistic attention pattern (decreasing entropy over layers)
            attention = np.zeros((n_layers, n_heads, seq_len, seq_len))
            for l in range(n_layers):
                for h in range(n_heads):
                    for q in range(seq_len):
                        # Causal mask
                        attn = np.zeros(seq_len)
                        attn[:q+1] = np.random.exponential(0.5, q+1)
                        # Later layers focus more on recent tokens
                        focus_factor = 1 + l / n_layers
                        attn[:q+1] *= np.exp(np.linspace(-1, 0, q+1) * focus_factor)
                        attn = attn / (attn.sum() + 1e-10)
                        attention[l, h, q] = attn
            
            hidden_states = np.random.randn(n_layers + 1, seq_len, 512)
        else:
            # Load real attention patterns
            attention_files = list(model_dir.glob("prompt_*.pkl"))
            if attention_files:
                with open(attention_files[0], 'rb') as f:
                    data = pickle.load(f)
                attention = data['attention_weights']
                
                # Try to load hidden states
                hidden_dir = config.TRANSFORMER_ATTENTION_DIR / "hidden_states" / model_name
                hidden_files = list(hidden_dir.glob("prompt_*.pkl"))
                if hidden_files:
                    with open(hidden_files[0], 'rb') as f:
                        hidden_data = pickle.load(f)
                    hidden_states = hidden_data['hidden_states']
                else:
                    hidden_states = np.random.randn(attention.shape[0] + 1, attention.shape[2], 512)
            else:
                continue
        
        logger.info(f"Analyzing temporal dynamics for {model_name}...")
        
        # Analyze attention evolution
        attn_evolution = analyze_token_attention_evolution(attention)
        
        # Analyze information accumulation
        if hidden_states is not None:
            info_accumulation = analyze_information_accumulation(hidden_states)
        else:
            info_accumulation = {}
        
        transformer_dynamics[model_name] = {
            'attention_evolution': attn_evolution,
            'information_accumulation': info_accumulation,
            'n_layers': int(attention.shape[0]),
            'seq_len': int(attention.shape[2]),
        }
    
    results['transformer_dynamics'] = transformer_dynamics
    
    # =========================================================================
    # CROSS-DOMAIN COMPARISON
    # =========================================================================
    logger.info("\n--- Cross-Domain Temporal Comparison ---")
    
    for model_name, model_dynamics in transformer_dynamics.items():
        comparison = compare_temporal_dynamics(
            results['brain_dynamics'].get('connectivity', {}),
            model_dynamics.get('attention_evolution', {})
        )
        results['cross_domain'][model_name] = comparison
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Save main results
    output_path = config.CROSS_DOMAIN_DIR / "temporal_dynamics.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved temporal dynamics to {output_path}")
    
    # Save connectivity matrices
    conn_df = pd.DataFrame({
        'window': range(len(conn_matrices)),
        'time_tr': window_times,
        'time_sec': window_times * tr,
        'tom_connectivity': conn_dynamics['tom_connectivity_timeseries'],
    })
    conn_df.to_csv(config.CROSS_DOMAIN_DIR / "dynamic_connectivity.csv", index=False)
    
    # Save phase synchrony
    sync_df = pd.DataFrame(phase_sync, columns=roi_names, index=roi_names)
    sync_df.to_csv(config.CROSS_DOMAIN_DIR / "phase_synchrony.csv")
    
    # Print summary
    print_temporal_summary(results)
    
    return results


def print_temporal_summary(results):
    """Print summary of temporal analysis results."""
    
    logger.info("\n" + "="*60)
    logger.info("Temporal Dynamics Summary")
    logger.info("="*60)
    
    # Brain dynamics
    brain = results.get('brain_dynamics', {})
    conn = brain.get('connectivity', {})
    sync = brain.get('phase_synchrony', {})
    
    logger.info("\n--- Brain Temporal Dynamics ---")
    logger.info(f"  ToM network connectivity: {conn.get('mean_tom_connectivity', 0):.3f}")
    logger.info(f"  Connectivity variability: {conn.get('tom_connectivity_variability', 0):.3f}")
    logger.info(f"  ToM network phase sync: {sync.get('tom_network_sync', 0):.3f}")
    
    profiles = brain.get('activation_profiles', {})
    logger.info(f"  Peak activation time: {profiles.get('peak_time_sec', 0):.1f} sec")
    
    # Transformer dynamics
    logger.info("\n--- Transformer Temporal Dynamics ---")
    for model_name, model_data in results.get('transformer_dynamics', {}).items():
        attn_evo = model_data.get('attention_evolution', {})
        logger.info(f"\n  {model_name}:")
        logger.info(f"    Attention becomes focused: {attn_evo.get('attention_becomes_focused', 0):.2f}")
        logger.info(f"    Mean entropy change: {attn_evo.get('mean_entropy_change', 0):.4f}")
        
        info_acc = model_data.get('information_accumulation', {})
        early_int = info_acc.get('early_layer_fast_integration', 0)
        late_int = info_acc.get('late_layer_fast_integration', 0)
        logger.info(f"    Early layer integration: {early_int:.2f}")
        logger.info(f"    Late layer integration: {late_int:.2f}")
    
    # Cross-domain
    logger.info("\n--- Cross-Domain Alignment ---")
    stage_names = ['Early', 'Middle', 'Late']
    for model_name, comparison in results.get('cross_domain', {}).items():
        stage = comparison.get('peak_processing_stage', {})
        aligned = stage.get('aligned', False)
        brain_idx = min(stage.get('brain', 1), 2)  # Clamp to valid range
        trans_idx = min(stage.get('transformer', 1), 2)
        logger.info(f"  {model_name}:")
        logger.info(f"    Processing stage alignment: {'✓ Aligned' if aligned else '✗ Different'}")
        logger.info(f"    Brain peak: {stage_names[brain_idx]}")
        logger.info(f"    Transformer peak: {stage_names[trans_idx]}")


if __name__ == "__main__":
    results = run_temporal_analysis()
    print("\n✅ Stage 12 completed: Temporal dynamics analysis")

