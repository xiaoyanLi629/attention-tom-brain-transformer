# Attention Mechanisms for Theory of Mind: Brain-Transformer Alignment and the Limits of Sparse Processing

**CogSci 2026 Submission**

*Part 3 of the "Adaptive Efficiency" Framework*

---

## Overview

This project investigates **attention mechanisms in social cognition** by comparing how human brains and transformer models process Theory of Mind (ToM) tasks. Building on Projects 1 and 2, we demonstrate that sparse, specialized attention characterizes efficient social cognition in both biological and artificial systems.

### The Unified Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE EFFICIENCY FRAMEWORK                    │
├─────────────────────────────────────────────────────────────────┤
│  PROJECT 1: Neural Efficiency                                   │
│  └── Network stability under cognitive load                     │
│                                                                  │
│  PROJECT 2: Sparse Routing                                      │
│  └── Brain (high specialization) vs MoE (load balancing)        │
│                                                                  │
│  PROJECT 3: Attention Mechanisms (THIS PROJECT)                 │
│  └── Brain-Transformer alignment in ToM representations         │
│                                                                  │
│  UNIFYING PRINCIPLE:                                             │
│  Efficient cognition = Sparse, selective processing             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Findings

### Hypothesis Testing Results

| Hypothesis | Description | Result | Evidence |
|------------|-------------|--------|----------|
| **H1** | High-efficiency individuals show sparser ToM activation | ⚠️ PARTIAL | Sparsity=0.85, but r=0.03 with behavior |
| **H2** | Transformers have specialized ToM mechanisms | ✅ SUPPORTED | Causal circuits (205-1200 components) |
| **H3** | Brain-Transformer representations align | ✅ SUPPORTED | RSA: r=0.45-0.57, all p<.05 |
| **H4** | Model layers map to brain regions | ⚠️ PARTIAL | Encoding R²=0.17 max |
| **H5** | Sparsity predicts ToM performance | ⚠️ PARTIAL | Circuits needed, behavioral link weak |
| **H6** | Adaptive efficiency is unified | ✅ SUPPORTED | Cross-project consistency |

### Major Discoveries

| Finding | Brain | Transformer |
|---------|-------|-------------|
| **Sparsity** | High (0.85) | Moderate (0.51-0.64) |
| **Specialization** | ToM ROIs (TPJ, mPFC) | ToM attention heads |
| **Alignment** | - | RSA r=0.45-0.57 with brain |
| **Causal Circuits** | Lesion literature | 205-1200 components |

---

## Data Sources

### Human Neuroimaging
- **Source**: Human Connectome Project (HCP)
- **Task**: Social Cognition / Theory of Mind
- **Subjects**: 20 (same as P1 and P2)
- **Conditions**: Mental (intentional) vs. Random (mechanical) animations
- **Brain ROIs**: TPJ, mPFC, STS, PC

### Transformer Models
| Model | Parameters | Layers | Heads | Type |
|-------|-----------|--------|-------|------|
| LLaMA-2-7B | 7B | 32 | 32 | Dense |
| DeepSeek-MoE-16B | 16B (2.8B active) | 28 | 16 | MoE |
| Qwen2-7B | 7B | 28 | 28 | Dense |
| Phi-3-Mini | 3.8B | 32 | 32 | Dense |
| Mistral-7B | 7B | 32 | 32 | Dense |
| GPT-2-XL | 1.5B | 48 | 25 | Dense |
| GPT-2-Medium | 355M | 24 | 16 | Dense |

---

## Analysis Pipeline

```
Stage 1: Behavioral Analysis
    ↓
Stage 2: Brain Activation Analysis
    ↓
Stage 3: Transformer Deep Extraction (6 models)
    ↓
Stage 4: Probing Analysis
    ↓
Stage 5: Causal Mechanism Analysis
    ↓
Stage 6: RSA Alignment
    ↓
Stage 7: Encoding Models
    ↓
Stage 8: Cross-Project Integration
    ↓
Stage 9: Basic Visualization (Fig 1-8)
    ↓
Stage 10: Advanced Analysis (Geometry, Clustering, CKA)
    ↓
Stage 11: Advanced Visualization (Fig 9-14)
    ↓
Stage 12: Temporal Dynamics Analysis
    ↓
Stage 13: Temporal Visualization (Fig 15-17)
```

---

## Project Structure

```
project_3/
├── configs/
│   ├── __init__.py                      # Package init
│   ├── config.py                        # Configuration & model registry
│   └── hcp_video_descriptions.py        # HCP video stimulus descriptions
├── scripts/
│   ├── __init__.py                      # Package init
│   ├── s01_behavioral_analysis.py       # Behavioral data extraction
│   ├── s02_brain_activation.py          # Brain ROI activation
│   ├── s03_transformer_extraction.py    # Model attention extraction
│   ├── s04_probing_analysis.py          # Hidden state probing
│   ├── s05_causal_analysis.py           # Causal ablation analysis
│   ├── s06_rsa_alignment.py             # Representational similarity
│   ├── s07_encoding_models.py           # Brain encoding models
│   ├── s08_cross_project.py             # Cross-project integration
│   ├── s09_visualization.py             # Basic figures (1-8)
│   ├── s10_advanced_analysis.py         # Advanced analysis methods
│   ├── s11_advanced_visualization.py    # Advanced figures (9-14)
│   ├── s12_temporal_analysis.py         # Temporal dynamics
│   ├── s13_temporal_visualization.py    # Temporal figures (15-17)
│   └── s14_glass_brain_visualization.py # Glass brain figures (18)
├── logs/                                # Log files (gitignored)
├── models/                              # Model cache directory (gitignored)
├── results/                             # Generated results (gitignored)
│   └── run_YYYYMMDD_HHMMSS/             # Timestamped results
│       ├── behavioral/
│       ├── brain_attention/
│       ├── transformer_attention/
│       ├── cross_domain/
│       ├── integration/
│       ├── rsa/
│       └── figures/                     # Publication figures (PNG + SVG)
├── .gitignore                           # Git ignore file
├── requirements.txt                     # Python dependencies
├── run_pipeline.py                      # Main pipeline script
└── README.md                            # This file
```

---

## Generated Figures

This project generates 22 publication-quality figures (in both PNG and SVG formats) that visualize the analysis results. Below is a detailed description of each figure, including how it is generated and what it represents.

---

### Basic Analysis Figures (Fig 1-8)

These figures are generated by `s09_visualization.py` and visualize the core analysis results.

#### **Figure 1: Behavioral Results** (`fig01_behavioral.png`)
- **Generation**: Extracts behavioral data from HCP Social Cognition task, computes accuracy, reaction time (RT), and heuristic index for each subject.
- **Contents**: 
  - *Panel A*: ToM accuracy distribution showing Mental vs Random condition performance
  - *Panel B*: Reaction time comparison between conditions with statistical significance markers
  - *Panel C*: Heuristic index distribution linking to P1 efficiency groups (high vs low)
- **Meaning**: Demonstrates that subjects show reliable ToM processing with faster responses to mental state attribution. High-efficiency individuals (from P1) show distinct behavioral patterns.

#### **Figure 2: Brain Activation** (`fig02_brain_activation.png`)
- **Generation**: Extracts BOLD signal from predefined social brain ROIs (rTPJ, lTPJ, mPFC, PC, rSTS, lSTS) using spherical masks in MNI space.
- **Contents**:
  - *Panel A*: Bar chart of mean activation (Mental > Random contrast) for each ROI
  - *Panel B*: Activation by efficiency group comparison
  - *Panel C*: ROI correlation matrix showing functional connectivity
- **Meaning**: Right TPJ shows highest activation during ToM processing, consistent with its role as the core mentalizing hub. High-efficiency individuals show sparser but more focused activation.

#### **Figure 3: Attention Specialization** (`fig03_attention_specialization.png`)
- **Generation**: For each of 6 transformer models, extracts attention weights from all layers and heads, then classifies heads by specialization type (ToM-relevant, syntax, position, etc.).
- **Contents**:
  - *Panel A*: Model comparison bar chart showing ToM-specialized head counts
  - *Panel B*: Layer-wise specialization trend across models
  - *Panels C-H*: Individual model heatmaps (layers × heads) with specialization scores
- **Meaning**: Transformer models develop specialized attention heads for ToM-like processing, concentrated in middle-to-late layers. Larger models show more distinct specialization.

#### **Figure 4: Attention Sparsity** (`fig04_attention_sparsity.png`)
- **Generation**: Computes Gini coefficient and top-k concentration for attention distributions across all layers and models.
- **Contents**:
  - *Panel A*: Layer-wise sparsity curves for each model
  - *Panel B*: Model comparison boxplot of overall sparsity
  - *Panel C*: Sparsity evolution across processing depth
- **Meaning**: Attention becomes increasingly sparse in later layers, supporting the "sparse attention = efficient processing" hypothesis. MoE models (DeepSeek) show highest sparsity.

#### **Figure 5: Causal Circuit** (`fig05_causal_circuit.png`)
- **Generation**: Performs ablation analysis by systematically zeroing attention heads and measuring performance drop on ToM prompts vs control prompts.
- **Contents**:
  - *Panel A*: Top 20 causal components ranked by ablation effect
  - *Panel B*: Average ablation effect by model
  - *Panels C-H*: Individual model heatmaps showing layer-head importance
- **Meaning**: Identifies the minimal circuit necessary for ToM reasoning. Critical heads cluster in specific layers, forming a discoverable "ToM circuit" in each model.

#### **Figure 6: RSA Alignment** (`fig06_rsa_alignment.png`)
- **Generation**: Computes Representational Similarity Analysis (RSA) between brain ROI activation patterns and transformer layer representations using Pearson correlation of dissimilarity matrices.
- **Contents**:
  - *Panel A*: RSA correlation values for each model at best-aligning layer
  - *Panel B*: Layer-by-layer RSA trajectory showing alignment evolution
  - *Panel C*: Significance testing results (p-values)
- **Meaning**: Mid-to-late transformer layers show significant alignment with human brain representations (r = 0.42-0.62), suggesting convergent computational strategies.

#### **Figure 7: Cross-Project Sparsity** (`fig07_cross_project_sparsity.png`)
- **Generation**: Integrates sparsity metrics from P1 (neural efficiency), P2 (routing sparsity), and P3 (attention sparsity) for unified comparison.
- **Contents**:
  - Unified bar chart comparing sparsity across all three projects
  - Overlay of brain and model sparsity distributions
- **Meaning**: Demonstrates the unifying principle: efficient systems (brains and models) consistently exhibit sparse, selective processing regardless of the specific cognitive domain.

#### **Figure 8: Unified Framework** (`fig08_unified_framework.png`)
- **Generation**: Programmatically creates a conceptual diagram using matplotlib shapes, arrows, and text annotations.
- **Contents**:
  - Central diagram showing P1→P2→P3 progression
  - Connections between human brain and transformer model components
  - Unified principle statement at bottom
- **Meaning**: Visualizes the theoretical integration of all three projects under the "Adaptive Efficiency" framework, showing how sparse processing is a universal principle.

---

### Advanced Analysis Figures (Fig 9-14)

These figures are generated by `s11_advanced_visualization.py` and visualize deeper analytical insights.

#### **Figure 9: Representational Geometry** (`fig09_representational_geometry.png`)
- **Generation**: Applies PCA to hidden state representations from each model, reducing dimensionality to visualize the representational manifold.
- **Contents**:
  - *Panel A*: 2D PCA projections for each model colored by prompt type (ToM vs Control)
  - *Panel B*: Explained variance by principal components
  - *Panel C*: Dimensionality estimates across layers
- **Meaning**: ToM and control prompts occupy separable regions in representation space, with separation increasing in later layers. Models develop structured, low-dimensional representations.

#### **Figure 10: Attention Clustering** (`fig10_attention_clustering.png`)
- **Generation**: Computes feature signatures for each attention head (entropy, sparsity, position bias), then applies hierarchical clustering with Ward linkage.
- **Contents**:
  - *Panel A*: Truncated dendrograms (top 12 clusters) for each model
  - *Panel B*: Cluster count comparison across models
  - *Panel C*: Cluster feature heatmaps
  - *Panel D*: Cluster type distribution (stacked bar)
- **Meaning**: Attention heads naturally cluster into functional types. Models with more layers show richer clustering structure, suggesting emergent functional specialization.

#### **Figure 11: Information Flow** (`fig11_information_flow.png`)
- **Generation**: Computes attention entropy at each layer as a proxy for information flow, then visualizes as ridge plot (overlapping density curves).
- **Contents**:
  - *Panel A*: Ridge plot showing entropy distribution across layers for all 6 models
  - *Panel B*: Peak and minimum entropy layer positions
- **Meaning**: Information flow follows a characteristic pattern: entropy increases in early layers (information gathering) then decreases in late layers (information integration). The entropy minimum indicates the "bottleneck" layer where information is most compressed.

#### **Figure 12: Cross-Model Comparison** (`fig12_cross_model_comparison.png`)
- **Generation**: Computes Centered Kernel Alignment (CKA) between all pairs of models using their hidden state representations on the same inputs.
- **Contents**:
  - *Panel A*: 6×6 CKA similarity matrix heatmap
  - *Panel B*: Hierarchical clustering of models by representation similarity
  - *Panel C*: Architecture comparison radar chart (layers, heads, parameters)
- **Meaning**: Despite architectural differences, all models show high representational similarity (CKA > 0.91), suggesting convergent ToM processing strategies. Model family (GPT vs LLaMA-family) shows in clustering.

#### **Figure 13: Circuit Discovery** (`fig13_circuit_discovery.png`)
- **Generation**: Identifies ToM-critical attention heads via ablation, then visualizes their distribution and connections across layers.
- **Contents**:
  - *Panel A*: Head importance heatmap for each model
  - *Panel B*: ToM circuit pathway diagram showing information flow through critical heads
  - *Panel C*: Layer-wise importance summary
  - *Panel D*: Circuit size comparison
- **Meaning**: ToM processing relies on a sparse circuit of attention heads (198-425 heads depending on model), concentrated in middle-to-late layers. This supports the hypothesis of specialized mechanisms for social cognition.

#### **Figure 14: Enhanced Alignment** (`fig14_enhanced_alignment.png`)
- **Generation**: Extends RSA analysis with encoding model results, layer-region mapping, and alignment trajectory visualization.
- **Contents**:
  - *Panel A*: Detailed layer-by-layer RSA curves for each model
  - *Panel B*: Encoding model R² values (predicting brain from model)
  - *Panel C*: Brain region to transformer layer correspondence matrix
  - *Panel D*: Summary statistics table
- **Meaning**: Provides comprehensive evidence for brain-transformer alignment, showing that specific transformer layers correspond to specific brain regions (e.g., TPJ aligns with late layers).

---

### Temporal Dynamics Figures (Fig 15-17)

These figures are generated by `s13_temporal_visualization.py` and visualize time-varying aspects of processing.

#### **Figure 15: Dynamic Connectivity** (`fig15_dynamic_connectivity.png`)
- **Generation**: Analyzes time-resolved fMRI data using sliding window correlation to capture dynamic functional connectivity between ToM ROIs.
- **Contents**:
  - *Panel A*: Connectivity matrices at different time points (Early, Middle, Late, Return)
  - *Panel B*: Temporal evolution of connectivity strength
  - *Panel C*: Phase synchrony analysis
- **Meaning**: Brain networks show dynamic reconfiguration during ToM tasks, with connectivity peaking during active mentalizing and decreasing during baseline periods.

#### **Figure 16: Transformer Temporal** (`fig16_transformer_temporal.png`)
- **Generation**: Tracks attention patterns across token positions (simulating temporal processing) and computes attention entropy evolution.
- **Contents**:
  - *Panel A*: Attention entropy evolution across tokens for each model
  - *Panel B*: Layer-wise information accumulation curves
  - *Panel C*: Attention pattern change rate (convergence analysis)
  - *Panel D*: Processing stage alignment indicators
- **Meaning**: Transformer attention evolves systematically across token processing, with early tokens showing high entropy (exploration) and later tokens showing focused attention (integration).

#### **Figure 17: Temporal Comparison** (`fig17_temporal_comparison.png`)
- **Generation**: Aligns brain temporal dynamics with transformer processing stages, computing correlation between time-resolved brain patterns and layer-wise model patterns.
- **Contents**:
  - *Panel A*: Brain-model temporal alignment curves
  - *Panel B*: Peak alignment timepoints
  - *Panel C*: Processing stage correspondence diagram
  - *Panel D*: Summary statistics
- **Meaning**: Brain processing stages (early perception → mentalizing → integration) show correspondence with transformer layer groups, supporting the idea of analogous computational stages.

---

## Usage

### Run Full Pipeline

```bash
# Complete analysis (all 13 stages)
python run_pipeline.py

# Specific stages only
python run_pipeline.py --stages 1 2 3

# Run visualization stages only
python run_pipeline.py --viz-only

# List available stages
python run_pipeline.py --list
```

### Output

Each run creates a **timestamped directory** under `results/`:
```
results/run_20251221_123456/
├── behavioral/            # Human behavioral data
├── brain_attention/       # Brain activation analysis
├── transformer_attention/ # Model attention data (6 models)
├── cross_domain/          # RSA and encoding results
├── integration/           # Cross-project analysis
└── figures/               # 17 figures (PNG + SVG formats)
    ├── fig01_behavioral.png/svg      # Behavioral analysis
    ├── fig02_brain_activation.png/svg # Brain ROI activation
    ├── fig03-08_*.png/svg            # Core analysis figures
    ├── fig09-14_*.png/svg            # Advanced analysis figures
    └── fig15-17_*.png/svg            # Temporal dynamics figures
```

---

## Statistical Summary

### Brain Analysis

| Metric | Value |
|--------|-------|
| Mean rTPJ Activation | 1.58 ± 0.70 |
| Mean lTPJ Activation | 1.25 ± 0.44 |
| Mean mPFC Activation | 0.83 ± 0.24 |
| Brain Sparsity | 0.85 |
| Gini Coefficient | 0.70 |

### RSA Alignment (6 Models)

| Model | Best RSA | Best Layer | p-value |
|-------|----------|------------|---------|
| DeepSeek-MoE-16B | r = 0.62 | L14 | 0.003** |
| Qwen2-7B | r = 0.58 | L16 | 0.008** |
| Phi-3-Mini | r = 0.55 | L18 | 0.015* |
| Mistral-7B | r = 0.52 | L20 | 0.028* |
| GPT-2-XL | r = 0.48 | L24 | 0.036* |
| GPT-2-Medium | r = 0.42 | L12 | 0.058 |

### Causal Circuits

| Model | Circuit Size | Sparsity | Top Layers |
|-------|--------------|----------|------------|
| DeepSeek-MoE-16B | 312 | 0.68 | 14, 16, 18 |
| Qwen2-7B | 287 | 0.65 | 15, 17, 19 |
| Phi-3-Mini | 245 | 0.72 | 16, 18, 20 |
| Mistral-7B | 356 | 0.58 | 18, 20, 22 |
| GPT-2-XL | 425 | 0.55 | 24, 26, 28 |
| GPT-2-Medium | 198 | 0.62 | 12, 14, 16 |

### Cross-Model CKA Similarity

Models show high representational similarity (CKA > 0.91) suggesting convergent ToM processing strategies across architectures.

---

## Connection to Other Projects

| Project | Focus | Link to P3 |
|---------|-------|------------|
| **P1** | Neural Efficiency | Same subjects, efficiency groups |
| **P2** | Sparse Routing | Sparsity metrics, cross-domain |
| **P3** | Attention Mechanisms | This project |

**Unified Finding:** Sparse, selective processing enables efficient cognition across:
- Network dynamics (P1)
- Information routing (P2)
- Attention allocation (P3)

---

## Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- torch, transformers (for LLMs)
- nibabel, nilearn (for fMRI)
- scipy, scikit-learn (for RSA/encoding)
- matplotlib, seaborn, plotly (for visualization)

### Hardware
- GPU: 16GB+ VRAM (for LLaMA/Mistral)
- RAM: 32GB+
- Storage: 50GB+

---

## Citation

```bibtex
@inproceedings{author2026sparse,
  title={Attention Mechanisms for Theory of Mind: Brain-Transformer 
         Alignment and the Limits of Sparse Processing},
  author={[Author Names]},
  booktitle={Proceedings of the Cognitive Science Society},
  year={2026}
}
```

---

## License

Academic research purposes only.

---

*Last Updated: 2026-01-30*
*Version: 3.1 (with 18 figures, 6 models, 14 stages)*
