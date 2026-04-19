"""
=============================================================================
Project Configuration: Attention Mechanisms for Theory of Mind
IEEE BIBM 2026 Submission (revised from CogSci 2026 #3240)
=============================================================================

This configuration file centralizes all paths, parameters, and constants
used throughout the analysis pipeline.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths — everything lives under the project directory on autodl-fs
PROJECT_DIR = Path("/autodl-fs/data/CCN_Competition/"
                   "Attention Mechanisms for Theory of Mind "
                   "Weak Brain-Transformer Alignment Despite Behavioral Success")

# Data paths
DATA_DIR = PROJECT_DIR / "data"
DS002345_DIR = DATA_DIR / "ds002345"          # Narratives dataset (primary)
DS002345_FMRI = DS002345_DIR / "fmri"         # fMRI NIfTI files
DS002345_STIMULI = DS002345_DIR / "stimuli"    # Audio WAV files
DS002345_TRANSCRIPTS = DS002345_DIR / "transcripts"  # Whisper transcripts
DS002345_REPO = DS002345_DIR / "repo"          # OpenNeuro git-annex metadata
DS000109_DIR = DATA_DIR / "ds000109"           # Mitchell false-belief (backup)

# Output directories (base)
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = PROJECT_DIR / "models" / "huggingface_cache"
MODEL_CACHE_DIR = MODELS_DIR  # Alias for HuggingFace cache
CACHE_DIR = PROJECT_DIR / "cache"             # whisper, torch, etc.

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
# SUBJECT LIST (ds002345 Shapes task — 59 subjects)
# =============================================================================

EXCLUDED_SUBJECTS = ['sub-115']  # FOV doesn't cover posterior ROIs (rTPJ, lTPJ, PC ≈ 0)

def get_subject_list():
    """Get list of subjects with shapes fMRI data"""
    subjects = []
    if DS002345_FMRI.exists():
        for item in sorted(DS002345_FMRI.iterdir()):
            if item.is_dir() and item.name.startswith('sub-'):
                if item.name in EXCLUDED_SUBJECTS:
                    continue
                # Verify both tasks exist
                social = item / "func" / f"{item.name}_task-shapessocial_bold.nii.gz"
                physical = item / "func" / f"{item.name}_task-shapesphysical_bold.nii.gz"
                if social.exists() and physical.exists():
                    subjects.append(item.name)
    return subjects

SUBJECTS = get_subject_list()

# =============================================================================
# ds002345 SHAPES TASK PARAMETERS
# =============================================================================

SHAPES_TASK = {
    'name': 'shapes',
    'full_name': 'Animated Shapes: Social vs Physical Narration',
    'conditions': {
        'social': 'Social/intentional description (ToM)',
        'physical': 'Physical/geometric description (control)',
    },
    'task_names': ['shapessocial', 'shapesphysical'],
    'tr': 1.5,
    'intro_duration': 49.5,   # seconds of intro music to trim
    'intro_trs': 33,          # intro_duration / TR
    'story_onset': 49.5,      # seconds when story narration begins
    'story_duration': 408.0,  # seconds of story (social condition)
    'expected_networks': ['DMN', 'TPJ'],
    'description': 'Subjects listened to social vs physical descriptions of same animated shapes',
}

# Mental-state keywords for attention specialization analysis
MENTAL_STATE_WORDS = {
    'agents': ['boy', 'father', 'friend', 'monster', 'son', 'dad'],
    'mental_verbs': ['wants', 'tries', 'believes', 'thinks', 'dreams',
                     'feels', 'notices', 'knows', 'hopes', 'plans'],
    'mental_states': ['frightened', 'excited', 'disappointed', 'desperate',
                      'uncertain', 'peaceful', 'forlornly', 'unenthusiastically'],
    'social_actions': ['chasing', 'escaping', 'sneaks', 'greets', 'calls',
                       'wakes', 'tucks', 'kisses', 'speaks', 'talks'],
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
# FILE PATH FUNCTIONS (ds002345)
# =============================================================================

def get_fmri_path(subject, task='shapessocial'):
    """Get path to ds002345 BOLD data"""
    return DS002345_FMRI / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"

def get_events_path(subject, task='shapessocial'):
    """Get path to ds002345 events file"""
    return DS002345_FMRI / subject / "func" / f"{subject}_task-{task}_events.tsv"

def get_transcript_path(task='shapessocial'):
    """Get path to whisper transcript"""
    return DS002345_TRANSCRIPTS / f"{task}_transcript.txt"

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

