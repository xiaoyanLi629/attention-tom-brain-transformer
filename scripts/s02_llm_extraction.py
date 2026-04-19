#!/usr/bin/env python3
"""
=============================================================================
Stage 2: LLM Time-Resolved Feature Extraction
=============================================================================

For each transformer model, process story transcripts and extract:
  - Hidden states per layer, binned by fMRI TR (using word-to-TR mapping)
  - Attention weights for sparsity/specialization analysis

This creates the predictor matrices for Stage 3 encoding models.

Outputs:
  - llm_features/{model}/{task}_layer{L}.npy  — (n_TRs, hidden_dim)
  - llm_attention/{model}/{task}_sparsity.csv  — attention metrics per layer/head
  - llm_extraction_summary.json               — model metadata and extraction stats
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import json
import logging
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Point HF cache to project directory
import os
os.environ['HF_HOME'] = str(config.CACHE_DIR / 'huggingface')
os.environ['TORCH_HOME'] = str(config.CACHE_DIR / 'torch')


# =============================================================================
# MODEL LOADING (reused from old s03)
# =============================================================================

def load_transformer_model(model_config):
    """Load a Transformer model with attention and hidden state output."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_name = model_config['name']
    short_name = model_config['short_name']
    logger.info(f"Loading {short_name} ({model_name})...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=config.MODEL_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        trust_remote_code=True,
        cache_dir=config.MODEL_CACHE_DIR,
        output_attentions=True,
        output_hidden_states=True,
    )

    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Parse parameter count from string like '7B', '355M', '16B (2.8B active)'
        params_str = str(model_config.get('parameters', '0'))
        params_num = params_str.split('(')[0].strip()
        if 'B' in params_num:
            params_b = float(params_num.replace('B', '')) * 1e9
        elif 'M' in params_num:
            params_b = float(params_num.replace('M', '')) * 1e6
        else:
            params_b = 0

        if params_b <= 2e9:  # GPT-2 class
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs).cuda()
        elif params_b <= 8e9 and gpu_gb >= 20:  # Up to 7B on 24GB GPU
            # bf16 when available: fp16 overflows on some models (e.g. Qwen2)
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto",
                **load_kwargs)
        else:  # Large models: 8-bit quantization
            logger.info("  Using 8-bit quantization...")
            load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", **load_kwargs)
    else:
        logger.warning("No GPU available, loading on CPU")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.eval()
    logger.info(f"  {short_name} loaded successfully")
    return model, tokenizer


# =============================================================================
# ATTENTION METRICS (reused from old s03)
# =============================================================================

def compute_attention_sparsity(attention_weights):
    """Compute sparsity metrics for attention weights.

    Args:
        attention_weights: (n_layers, n_heads, seq_len, seq_len)

    Returns:
        list of dicts with keys: layer, head, entropy, gini, top5, sparsity
    """
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    results = []

    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attention_weights[layer, head]  # (seq_len, seq_len)
            avg_attn = attn.mean(axis=0)  # average over queries
            avg_attn = avg_attn / (avg_attn.sum() + 1e-10)

            # Entropy
            log_attn = np.log2(avg_attn + 1e-10)
            entropy = -np.sum(avg_attn * log_attn) / np.log2(max(seq_len, 2))

            # Gini
            sorted_attn = np.sort(avg_attn)
            n = len(sorted_attn)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_attn) / (n * np.sum(sorted_attn) + 1e-10)) - (n + 1) / n

            # Top-5 concentration
            top5 = np.sort(avg_attn)[-5:].sum()

            results.append({
                'layer': layer, 'head': head,
                'entropy': float(entropy),
                'gini': float(gini),
                'top5_concentration': float(top5),
                'sparsity': float(1 - entropy),
            })

    return results


# =============================================================================
# TR-ALIGNED EXTRACTION
# =============================================================================

def extract_tr_aligned_features(model, tokenizer, text, word_to_tr, n_story_trs,
                                hrf_delays=(3, 4, 5)):
    """Extract hidden states aligned to fMRI TRs.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: full story transcript (string)
        word_to_tr: list of dicts with 'word', 'tr_idx' from Stage 1
        n_story_trs: number of TRs in the fMRI story period
        hrf_delays: TR delays to account for hemodynamic response

    Returns:
        layer_features: dict {layer_idx: (n_story_trs, hidden_dim)}
        attention_weights: (n_layers, n_heads, seq_len, seq_len) numpy array
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')[0]  # (n_tokens, 2)
    input_ids = inputs['input_ids']

    # Handle context window limit (e.g., GPT-2 max 1024 tokens)
    max_length = getattr(model.config, 'max_position_embeddings',
                         getattr(model.config, 'n_positions', 8192))
    if input_ids.shape[1] > max_length:
        logger.warning(f"  Truncating {input_ids.shape[1]} tokens to {max_length} (model limit)")
        input_ids = input_ids[:, :max_length]
        offset_mapping = offset_mapping[:max_length]
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if 'attention_mask' in inputs:
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

    seq_len = input_ids.shape[1]
    logger.info(f"  Tokenized: {seq_len} tokens from {len(text.split())} words (max={max_length})")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=inputs.get('attention_mask'))

    # Extract hidden states: tuple of (1, seq_len, hidden_dim), one per layer + embedding
    hidden_states = outputs.hidden_states  # (n_layers+1,)
    n_layers = len(hidden_states) - 1  # exclude embedding layer
    hidden_dim = hidden_states[0].shape[-1]

    # Extract attention: tuple of (1, n_heads, seq_len, seq_len), one per layer
    attentions = outputs.attentions
    n_heads = attentions[0].shape[1]

    # Stack attention weights for sparsity analysis
    attn_array = np.stack([a[0].cpu().float().numpy() for a in attentions])
    # shape: (n_layers, n_heads, seq_len, seq_len)

    # Build token-to-word mapping using character offsets
    # Each word in word_to_tr has a position in the text
    words_in_text = text.split()
    token_offsets = offset_mapping.numpy()  # (n_tokens, 2) start/end char positions

    # Map each token to a word index by character position
    word_char_starts = []
    pos = 0
    for w in words_in_text:
        idx = text.find(w, pos)
        word_char_starts.append(idx)
        pos = idx + len(w)

    # For each token, find which word it belongs to
    token_to_word_idx = []
    for tok_start, tok_end in token_offsets:
        if tok_start == tok_end:  # special token
            token_to_word_idx.append(-1)
            continue
        best_word = -1
        for wi, wstart in enumerate(word_char_starts):
            if wi < len(words_in_text):
                wend = wstart + len(words_in_text[wi])
                if tok_start >= wstart and tok_start < wend:
                    best_word = wi
                    break
        token_to_word_idx.append(best_word)

    # Build word_idx → tr_idx mapping from Stage 1 output
    # word_to_tr entries are only for story words (after intro trimming)
    # We need to map story words to their sequential index in the full text
    story_words = [w['word'] for w in word_to_tr]

    # Find where story words start in the full word list
    # Story starts after intro music words
    story_start_word_idx = None
    for i, w in enumerate(words_in_text):
        if i < len(words_in_text) - len(story_words) + 1:
            if words_in_text[i] == story_words[0] and words_in_text[i+1] == story_words[1]:
                story_start_word_idx = i
                break

    if story_start_word_idx is None:
        logger.warning("  Could not align story words to transcript, using all words")
        story_start_word_idx = 0

    # Create word_idx → TR mapping
    word_idx_to_tr = {}
    for si, entry in enumerate(word_to_tr):
        word_idx_to_tr[story_start_word_idx + si] = entry['tr_idx']

    # Create token → TR mapping
    token_to_tr = []
    for ti, wi in enumerate(token_to_word_idx):
        if wi in word_idx_to_tr:
            token_to_tr.append(word_idx_to_tr[wi])
        else:
            token_to_tr.append(-1)  # intro or unmapped token

    # Bin hidden states by TR
    layer_features = {}
    for layer_idx in range(n_layers):
        hs = hidden_states[layer_idx + 1][0].cpu().float().numpy()  # (seq_len, hidden_dim)
        tr_features = np.zeros((n_story_trs, hidden_dim))
        tr_counts = np.zeros(n_story_trs)

        for tok_idx, tr_idx in enumerate(token_to_tr):
            if 0 <= tr_idx < n_story_trs:
                tr_features[tr_idx] += hs[tok_idx]
                tr_counts[tr_idx] += 1

        # Mean pool tokens within each TR
        nonzero = tr_counts > 0
        tr_features[nonzero] /= tr_counts[nonzero, None]

        # Fill empty TRs with nearest neighbor interpolation
        if not nonzero.all():
            from scipy.interpolate import interp1d
            valid_trs = np.where(nonzero)[0]
            if len(valid_trs) > 1:
                for dim in range(hidden_dim):
                    interp_fn = interp1d(valid_trs, tr_features[valid_trs, dim],
                                         kind='nearest', fill_value='extrapolate')
                    empty_trs = np.where(~nonzero)[0]
                    tr_features[empty_trs, dim] = interp_fn(empty_trs)

        layer_features[layer_idx] = tr_features

    # Free GPU memory
    del outputs, hidden_states, attentions
    torch.cuda.empty_cache()

    return layer_features, attn_array


# =============================================================================
# MAIN
# =============================================================================

def run_stage2():
    """Run Stage 2: LLM time-resolved feature extraction."""
    logger.info("=" * 70)
    logger.info("STAGE 2: LLM Time-Resolved Feature Extraction")
    logger.info("=" * 70)

    config.ensure_run_directories()
    features_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_features"
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_attention"
    features_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)

    # Load word-to-TR mapping from Stage 1
    mapping_path = config.BRAIN_ATTENTION_DIR / "word_to_tr_mapping.json"
    if not mapping_path.exists():
        logger.error(f"Word-to-TR mapping not found: {mapping_path}. Run Stage 1 first.")
        return
    with open(str(mapping_path)) as f:
        word_mapping = json.load(f)

    # Determine n_story_trs from a sample ROI file
    roi_dir = config.BRAIN_ATTENTION_DIR / "roi_timeseries"
    sample_files = list(roi_dir.glob("*_shapessocial.npy"))
    if not sample_files:
        logger.error("No ROI time series found. Run Stage 1 first.")
        return
    sample_ts = np.load(str(sample_files[0]))
    n_story_trs = sample_ts.shape[0]
    logger.info(f"Story TRs from fMRI: {n_story_trs}")

    # Load transcripts (story portion only, skip intro)
    transcripts = {}
    for task in ['shapessocial', 'shapesphysical']:
        txt_path = config.DS002345_TRANSCRIPTS / f"{task}_transcript.txt"
        if txt_path.exists():
            transcripts[task] = txt_path.read_text().strip()
            logger.info(f"  {task} transcript: {len(transcripts[task].split())} words")

    # Get model configs
    models = config.TRANSFORMER_CONFIG['models']
    summary = {}

    for model_config in models:
        model_name = model_config['short_name']
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*50}")

        model_features_dir = features_dir / model_name
        model_attention_dir = attention_dir / model_name
        model_features_dir.mkdir(parents=True, exist_ok=True)
        model_attention_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        existing = list(model_features_dir.glob("*_layer0.npy"))
        if len(existing) >= 2:
            logger.info(f"  Already processed, skipping. Delete {model_features_dir} to re-run.")
            continue

        # Load model
        try:
            model, tokenizer = load_transformer_model(model_config)
        except Exception as e:
            logger.error(f"  Failed to load {model_name}: {e}")
            model, tokenizer = None, None
        if model is None:
            logger.error(f"  Skipping {model_name}")
            continue

        model_summary = {'model': model_name, 'tasks': {}}

        for task in ['shapessocial', 'shapesphysical']:
            if task not in transcripts or task not in word_mapping:
                logger.warning(f"  Missing data for {task}, skipping")
                continue

            logger.info(f"\n  --- {task} ---")

            try:
                layer_features, attn_array = extract_tr_aligned_features(
                    model, tokenizer, transcripts[task],
                    word_mapping[task], n_story_trs)
            except Exception as e:
                logger.error(f"  Extraction failed for {task}: {e}")
                continue

            n_layers = len(layer_features)
            hidden_dim = layer_features[0].shape[1]
            logger.info(f"  Extracted: {n_layers} layers × {hidden_dim} dims × {n_story_trs} TRs")

            # Save layer features
            for layer_idx, features in layer_features.items():
                out_path = model_features_dir / f"{task}_layer{layer_idx}.npy"
                np.save(str(out_path), features)

            # Compute and save attention sparsity
            sparsity_records = compute_attention_sparsity(attn_array)
            for r in sparsity_records:
                r['task'] = task
                r['model'] = model_name
            sparsity_df = pd.DataFrame(sparsity_records)
            sparsity_path = model_attention_dir / f"{task}_sparsity.csv"
            sparsity_df.to_csv(str(sparsity_path), index=False)

            model_summary['tasks'][task] = {
                'n_layers': n_layers,
                'hidden_dim': hidden_dim,
                'n_trs': n_story_trs,
                'attn_shape': list(attn_array.shape),
                'mean_sparsity': float(sparsity_df['sparsity'].mean()),
            }

            del layer_features, attn_array
            gc.collect()

        summary[model_name] = model_summary

        # Unload model to free GPU
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Save summary
    summary_path = config.TRANSFORMER_ATTENTION_DIR / "llm_extraction_summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved: {summary_path}")

    logger.info("=" * 70)
    logger.info("Stage 2 Complete")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_stage2()
