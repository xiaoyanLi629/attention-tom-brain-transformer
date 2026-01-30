"""
=============================================================================
Project Configuration: Adaptive Heuristics in Social Cognition
CogSci 2026 Submission
=============================================================================

"When Fast is Better: Neural Signatures of Adaptive Heuristics in Social Cognition"

This configuration file centralizes all paths, parameters, and constants
used throughout the analysis pipeline.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths
PROJECT_ROOT = Path("/root/autodl-fs/CogSci")
DATA_ROOT = PROJECT_ROOT / "data"
PROJECT_DIR = PROJECT_ROOT / "project_3"

# Related project paths (for cross-project integration)
PROJECT_1_DIR = PROJECT_ROOT / "project_1"
PROJECT_2_DIR = PROJECT_ROOT / "project_2"

# Output directories (base)
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = PROJECT_DIR / "models" / "huggingface_cache"
MODEL_CACHE_DIR = MODELS_DIR  # Alias for HuggingFace cache

# Timestamped run directories (will be set by initialize_run_directories)
CURRENT_RUN_DIR = None
BEHAVIORAL_DIR = None
BRAIN_ATTENTION_DIR = None
TRANSFORMER_ATTENTION_DIR = None
CROSS_DOMAIN_DIR = None
INTEGRATION_DIR = None
FIGURES_DIR = None
RSA_DIR = None

# Create base directories if not exist
for d in [RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def initialize_run_directories(timestamp=None):
    """Initialize timestamped directories for current run"""
    global CURRENT_RUN_DIR, BEHAVIORAL_DIR, BRAIN_ATTENTION_DIR
    global TRANSFORMER_ATTENTION_DIR, CROSS_DOMAIN_DIR, INTEGRATION_DIR, FIGURES_DIR, RSA_DIR
    
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    CURRENT_RUN_DIR = RESULTS_DIR / f"run_{timestamp}"
    BEHAVIORAL_DIR = CURRENT_RUN_DIR / "behavioral"
    BRAIN_ATTENTION_DIR = CURRENT_RUN_DIR / "brain_attention"
    TRANSFORMER_ATTENTION_DIR = CURRENT_RUN_DIR / "transformer_attention"
    CROSS_DOMAIN_DIR = CURRENT_RUN_DIR / "cross_domain"
    INTEGRATION_DIR = CURRENT_RUN_DIR / "integration"
    FIGURES_DIR = CURRENT_RUN_DIR / "figures"
    RSA_DIR = CURRENT_RUN_DIR / "rsa"
    
    # Create directories
    for d in [CURRENT_RUN_DIR, BEHAVIORAL_DIR, BRAIN_ATTENTION_DIR,
              TRANSFORMER_ATTENTION_DIR, CROSS_DOMAIN_DIR, INTEGRATION_DIR, FIGURES_DIR, RSA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    return CURRENT_RUN_DIR

def ensure_run_directories():
    """Ensure run directories are initialized (for subprocess scripts)"""
    global CURRENT_RUN_DIR, BEHAVIORAL_DIR, BRAIN_ATTENTION_DIR
    global TRANSFORMER_ATTENTION_DIR, CROSS_DOMAIN_DIR, INTEGRATION_DIR, FIGURES_DIR, RSA_DIR
    
    import os
    timestamp = os.environ.get('CURRENT_RUN_TIMESTAMP')
    
    if timestamp:
        CURRENT_RUN_DIR = RESULTS_DIR / f"run_{timestamp}"
        BEHAVIORAL_DIR = CURRENT_RUN_DIR / "behavioral"
        BRAIN_ATTENTION_DIR = CURRENT_RUN_DIR / "brain_attention"
        TRANSFORMER_ATTENTION_DIR = CURRENT_RUN_DIR / "transformer_attention"
        CROSS_DOMAIN_DIR = CURRENT_RUN_DIR / "cross_domain"
        INTEGRATION_DIR = CURRENT_RUN_DIR / "integration"
        FIGURES_DIR = CURRENT_RUN_DIR / "figures"
        RSA_DIR = CURRENT_RUN_DIR / "rsa"
        
        for d in [CURRENT_RUN_DIR, BEHAVIORAL_DIR, BRAIN_ATTENTION_DIR,
                  TRANSFORMER_ATTENTION_DIR, CROSS_DOMAIN_DIR, INTEGRATION_DIR, FIGURES_DIR, RSA_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    else:
        # Fallback to latest run or create new
        runs = sorted(RESULTS_DIR.glob("run_*"))
        if runs:
            CURRENT_RUN_DIR = runs[-1]
            BEHAVIORAL_DIR = CURRENT_RUN_DIR / "behavioral"
            BRAIN_ATTENTION_DIR = CURRENT_RUN_DIR / "brain_attention"
            TRANSFORMER_ATTENTION_DIR = CURRENT_RUN_DIR / "transformer_attention"
            CROSS_DOMAIN_DIR = CURRENT_RUN_DIR / "cross_domain"
            INTEGRATION_DIR = CURRENT_RUN_DIR / "integration"
            FIGURES_DIR = CURRENT_RUN_DIR / "figures"
            RSA_DIR = CURRENT_RUN_DIR / "rsa"
        else:
            initialize_run_directories()

# =============================================================================
# SUBJECT LIST (Same as P1 and P2)
# =============================================================================

def get_subject_list():
    """Get list of all available subjects"""
    subjects = []
    for item in DATA_ROOT.iterdir():
        if item.is_dir() and item.name.isdigit():
            subjects.append(item.name)
    return sorted(subjects)

SUBJECTS = get_subject_list()

# =============================================================================
# SOCIAL COGNITION TASK PARAMETERS
# =============================================================================

SOCIAL_TASK = {
    'name': 'SOCIAL',
    'full_name': 'Social Cognition / Theory of Mind',
    'runs': ['LR', 'RL'],
    'conditions': {
        'mental': 'Theory of Mind (intentional)',
        'random': 'Random movement (mechanical)',
    },
    'contrasts': ['mental_vs_rnd'],
    'tr': 0.72,
    'n_volumes': 274,  # Per run
    'expected_networks': ['DMN', 'TPJ'],
    'description': 'Participants judge whether animated shapes move intentionally or randomly',
}

# =============================================================================
# COGNITIVE PROCESSING MODES
# =============================================================================

PROCESSING_MODES = {
    'heuristic': {
        'name': 'Heuristic / Intuitive',
        'description': 'Fast, automatic, pattern-based processing',
        'system': 'System 1',
        'networks': ['DMN', 'TPJ', 'mPFC'],
        'color': '#2ecc71',  # Green
    },
    'analytical': {
        'name': 'Analytical / Deliberate',
        'description': 'Slow, effortful, rule-based processing',
        'system': 'System 2',
        'networks': ['FPN', 'DAN', 'DLPFC'],
        'color': '#e74c3c',  # Red
    },
}

# =============================================================================
# BRAIN NETWORK DEFINITIONS
# =============================================================================

NETWORKS = {
    # Theory of Mind / Social Networks (Heuristic)
    'DMN': {
        'name': 'Default Mode Network',
        'role': 'Self-referential, social cognition',
        'processing_mode': 'heuristic',
        'color': '#CD3E4E',
    },
    'TPJ': {
        'name': 'Temporoparietal Junction',
        'role': 'Mentalizing, perspective-taking',
        'processing_mode': 'heuristic',
        'color': '#9b59b6',
    },
    'mPFC': {
        'name': 'Medial Prefrontal Cortex',
        'role': 'Social inference, self-other distinction',
        'processing_mode': 'heuristic',
        'color': '#3498db',
    },
    
    # Analytical / Control Networks
    'FPN': {
        'name': 'Frontoparietal Network',
        'role': 'Executive control, analytical reasoning',
        'processing_mode': 'analytical',
        'color': '#E69422',
    },
    'DAN': {
        'name': 'Dorsal Attention Network',
        'role': 'Top-down attention, focused analysis',
        'processing_mode': 'analytical',
        'color': '#00760E',
    },
    'DLPFC': {
        'name': 'Dorsolateral Prefrontal Cortex',
        'role': 'Working memory, rule application',
        'processing_mode': 'analytical',
        'color': '#e67e22',
    },
    
    # Other networks
    'VIS': {
        'name': 'Visual Network',
        'role': 'Visual processing',
        'processing_mode': 'sensory',
        'color': '#781286',
    },
    'SMN': {
        'name': 'Somatomotor Network',
        'role': 'Motor planning and execution',
        'processing_mode': 'motor',
        'color': '#4682B4',
    },
}

NETWORK_ORDER = ['DMN', 'TPJ', 'mPFC', 'FPN', 'DAN', 'VIS', 'SMN']

# =============================================================================
# TRANSFORMER MODEL CONFIGURATION
# =============================================================================

TRANSFORMER_CONFIG = {
    'models': [
        {
            'name': 'deepseek-ai/deepseek-moe-16b-base',
            'short_name': 'DeepSeek-MoE-16B',
            'type': 'Causal LM (MoE)',
            'parameters': '16B (2.8B active)',
            'n_layers': 28,
            'n_heads': 16,
            'hidden_size': 2048,
            'intermediate_size': 10944,
            'architecture': 'DeepSeek MoE',
            'num_experts': 64,
            'num_experts_per_tok': 6,
            'description': 'Mixture of Experts model with sparse activation, highly efficient',
        },
        {
            'name': 'Qwen/Qwen2-7B',
            'short_name': 'Qwen2-7B',
            'type': 'Causal LM',
            'parameters': '7B',
            'n_layers': 28,
            'n_heads': 28,
            'hidden_size': 3584,
            'intermediate_size': 18944,
            'architecture': 'Qwen2',
            'description': 'State-of-the-art multilingual model with strong reasoning',
        },
        {
            'name': 'microsoft/Phi-3-mini-4k-instruct',
            'short_name': 'Phi-3-Mini',
            'type': 'Causal LM',
            'parameters': '3.8B',
            'n_layers': 32,
            'n_heads': 32,
            'hidden_size': 3072,
            'intermediate_size': 8192,
            'architecture': 'Phi3',
            'description': 'Efficient small model with strong performance',
        },
        {
            'name': 'mistralai/Mistral-7B-v0.1',
            'short_name': 'Mistral-7B',
            'type': 'Causal LM',
            'parameters': '7B',
            'n_layers': 32,
            'n_heads': 32,
            'hidden_size': 4096,
            'intermediate_size': 14336,
            'architecture': 'Mistral',
            'description': 'Sliding window attention, efficient architecture',
        },
        {
            'name': 'gpt2-xl',
            'short_name': 'GPT-2-XL',
            'type': 'Causal LM',
            'parameters': '1.5B',
            'n_layers': 48,
            'n_heads': 25,
            'hidden_size': 1600,
            'intermediate_size': 6400,
            'architecture': 'GPT2',
            'description': 'Classic baseline, well-studied interpretability',
        },
        {
            'name': 'gpt2-medium',
            'short_name': 'GPT-2-Medium',
            'type': 'Causal LM',
            'parameters': '355M',
            'n_layers': 24,
            'n_heads': 16,
            'hidden_size': 1024,
            'intermediate_size': 4096,
            'architecture': 'GPT2',
            'description': 'Smaller baseline for comparison',
        },
    ],
    'inference': {
        'max_length': 256,
        'temperature': 0.0,  # Deterministic
        'do_sample': False,
    },
    'attention_analysis': {
        'extract_layers': 'all',  # or list of layer indices
        'aggregate_heads': False,  # Keep per-head analysis
        'normalize': True,
    },
}

# =============================================================================
# ADVANCED ANALYSIS PARAMETERS
# =============================================================================

ADVANCED_ANALYSIS = {
    # Representational geometry analysis
    'geometry': {
        'dimensionality_reduction': ['PCA', 'UMAP', 't-SNE'],
        'umap_neighbors': 15,
        'umap_min_dist': 0.1,
        'tsne_perplexity': 30,
        'pca_components': 50,
    },
    
    # Attention pattern analysis
    'attention_patterns': {
        'clustering_method': 'hierarchical',
        'n_clusters': 8,
        'linkage': 'ward',
        'distance_metric': 'cosine',
    },
    
    # Information flow analysis
    'information_flow': {
        'method': 'attention_rollout',  # or 'attention_flow', 'grad_cam'
        'layers_to_analyze': 'all',
        'aggregate_heads': True,
    },
    
    # Mechanistic interpretability
    'mechanistic': {
        'circuit_threshold': 0.1,
        'patching_method': 'activation_patching',
        'ablation_type': 'zero',  # or 'mean', 'resample'
    },
    
    # Cross-model comparison
    'cross_model': {
        'alignment_method': 'CKA',  # Centered Kernel Alignment
        'procrustes': True,
        'compare_layers': True,
    },
}

# ToM Task Categories for Transformers
TOM_CATEGORIES = {
    'mental': {
        'description': 'Intentional, goal-directed behavior',
        'keywords': ['wants', 'tries', 'follows', 'approaches', 'avoids'],
    },
    'random': {
        'description': 'Random, mechanical movement',
        'keywords': ['moves', 'floats', 'bounces', 'rotates'],
    },
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

ANALYSIS_PARAMS = {
    # Behavioral analysis
    'behavioral': {
        'rt_outlier_threshold': 3.0,  # Z-score
        'min_accuracy': 0.6,  # Minimum accuracy to include
        'heuristic_threshold': 0.8,  # RT ratio threshold for heuristic classification
    },
    
    # Brain attention analysis
    'brain_attention': {
        'activation_threshold': 2.0,  # Z-score for significant activation
        'sparsity_method': 'gini',  # From P2
        'tom_regions': ['TPJ', 'mPFC', 'DMN'],
        'analytical_regions': ['FPN', 'DAN', 'DLPFC'],
    },
    
    # Transformer attention analysis
    'transformer_attention': {
        'sparsity_method': 'entropy',  # Entropy of attention distribution
        'top_k_tokens': 10,  # Top attended tokens to analyze
        'min_attention_threshold': 0.01,  # Minimum attention to consider
    },
    
    # Cross-domain comparison
    'cross_domain': {
        'similarity_method': 'correlation',
        'normalize_sparsity': True,
    },
    
    # Cross-project integration
    'integration': {
        'correlation_method': 'spearman',
        'p_threshold': 0.05,
    },
}

# =============================================================================
# HEURISTIC INDEX CALCULATION
# =============================================================================

def compute_heuristic_index(rt_mental, rt_random, acc_mental, acc_random):
    """
    Compute heuristic processing index.
    
    Higher values indicate more heuristic (fast, intuitive) processing.
    Lower values indicate more analytical (slow, deliberate) processing.
    
    Formula: 
        HI = (Speed_advantage * Accuracy_mental) / (1 + RT_variability)
    
    Where:
        Speed_advantage = RT_random / RT_mental (>1 means faster for mental)
        Accuracy_mental = accuracy on mental condition
        RT_variability = std(RT) / mean(RT)
    """
    speed_advantage = rt_random / (rt_mental + 1e-10)
    accuracy_weight = acc_mental
    
    heuristic_index = speed_advantage * accuracy_weight
    
    return heuristic_index

# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

STATS_PARAMS = {
    'alpha': 0.05,
    'correction': 'fdr_bh',
    'n_permutations': 5000,
    'bootstrap_samples': 10000,
    'effect_size': 'cohens_d',
    'confidence_level': 0.95,
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Unified Color Scheme (consistent with P1 and P2)
UNIFIED_COLORS = {
    # Efficiency groups (from P1)
    'HIGH_EFF': '#1a5276',        # Deep Blue
    'LOW_EFF': '#c0392b',         # Deep Red
    
    # Processing modes
    'HEURISTIC': '#2ecc71',       # Green - Fast/Intuitive
    'ANALYTICAL': '#e74c3c',      # Red - Slow/Deliberate
    
    # System types
    'BRAIN': '#2ecc71',           # Green - Brain
    'AI': '#9b59b6',              # Purple - AI
    
    # Neutral
    'NEUTRAL': '#566573',
    'ACCENT': '#f39c12',
}

COLORMAPS = {
    'sequential': 'plasma',
    'diverging': 'RdBu_r',
    'categorical': 'Set2',
    'heatmap': 'YlOrRd',
    'correlation': 'coolwarm',
}

FIGURE_PARAMS = {
    'dpi': 300,
    'formats': ['png', 'svg'],
    'font_family': 'Arial',
    'font_size': {
        'title': 16,
        'subtitle': 14,
        'label': 12,
        'tick': 10,
        'legend': 10,
    },
    'figsize': {
        'single': (8, 6),
        'double': (14, 6),
        'triple': (18, 6),
        'square': (10, 10),
        'large': (16, 12),
        'summary': (16, 20),
    },
    'style': {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'legend.frameon': False,
        'figure.facecolor': 'white',
    },
}

# =============================================================================
# FILE PATH FUNCTIONS
# =============================================================================

def get_social_fmri_path(subject, run='LR'):
    """Get path to Social task fMRI data"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" / 
            f"tfMRI_SOCIAL_{run}" / 
            f"tfMRI_SOCIAL_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii")

def get_social_evs_path(subject, run='LR'):
    """Get path to Social task EVs directory"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" / 
            f"tfMRI_SOCIAL_{run}" / "EVs")

def get_social_stats_path(subject, run='LR'):
    """Get path to Social task behavioral stats"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" / 
            f"tfMRI_SOCIAL_{run}" / "EVs" / "Social_Stats.csv")

def get_p1_results_path():
    """Get path to Project 1 results"""
    return PROJECT_1_DIR / "results" / "latest"

def get_p2_results_path():
    """Get path to Project 2 results"""
    return PROJECT_2_DIR / "results"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

def setup_logging(name, level=logging.INFO):
    """Setup logging for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# =============================================================================
# CROSS-PROJECT INTEGRATION HELPERS
# =============================================================================

def load_p1_efficiency_groups():
    """Load efficiency group assignments from Project 1"""
    import json
    p1_path = get_p1_results_path()
    
    # Try to find efficiency groups file
    possible_paths = [
        p1_path / "behavioral" / "efficiency_groups.json",
        p1_path / "behavioral" / "flexibility_groups.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            if path.suffix == '.json':
                with open(path) as f:
                    return json.load(f)
            else:
                import pandas as pd
                return pd.read_csv(path)
    
    return None

def load_p2_sparsity_metrics():
    """Load sparsity metrics from Project 2"""
    import pandas as pd
    p2_path = get_p2_results_path()
    
    sparsity_path = p2_path / "neuroscience" / "sparsity_metrics.csv"
    if sparsity_path.exists():
        return pd.read_csv(sparsity_path)
    
    return None

