#!/usr/bin/env python3
"""
=============================================================================
Stage 5: Causal Analysis — Attention Head Ablation
=============================================================================

For each transformer model, zero-ablate individual attention heads and
measure their causal effect on mental-state word prediction.

Fix over original:
  - Hook the output-projection module via forward_pre_hook so we modify
    the *pre-projection* concatenated head tensor (shape: batch × seq ×
    n_heads*head_dim), slicing out the target head's contribution.
  - This is architecture-agnostic: works for GPT-2 (c_proj) and
    LLaMA/Mistral/Qwen (o_proj).

Outputs:
  - ablation_results.csv    — per-head ablation effect
  - causal_circuits.json    — top critical heads per model
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import gc
import json
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['HF_HOME'] = str(config.CACHE_DIR / 'huggingface')
os.environ['TORCH_HOME'] = str(config.CACHE_DIR / 'torch')


# Causal prompts — each ends just before a mental-state word the model predicts
CAUSAL_PROMPTS = [
    {
        'text': "The boy catches up to the ball and bounces it off the playground's fences several times. As he circles the ball, lining up his next shot, his father shows up and calls for him. The boy takes one final kick at the ball before leaving the playground to meet his waiting",
        'target_word': ' father',
        'control_word': ' ball',
    },
    {
        'text': "The four birds move in the same direction as the boy. The birds transform into a giant monster with two long arms and two wobbly legs. The monster swats at the boy. He misses and tries again. The boy flies over the monster, evading its grasp. The boy feels",
        'target_word': ' frightened',
        'control_word': ' nothing',
    },
    {
        'text': "The boys make a desperate plan and the new boy speeds away. He quickly returns with the ball. The boys put the ball in a slingshot and shoot it into the monster which breaks into a heap. The boy swoops over the heap until he notices that the other boy is gone. He goes looking for his new friend but can't find him. The boy feels",
        'target_word': ' disappointed',
        'control_word': ' tired',
    },
    {
        'text': "He forlornly walks to the single ball. He slowly and unenthusiastically kicks the ball halfway down the yard. As he kicks the ball again, his mysterious friend from the dream appears and sneaks towards the playground. The boy is",
        'target_word': ' surprised',
        'control_word': ' walking',
    },
]


def find_attention_output_projs(model):
    """Return list of (layer_idx, module) for each layer's output projection.

    Works for:
      - GPT-2 style: layer.attn.c_proj
      - LLaMA / Mistral / Qwen: layer.self_attn.o_proj
    """
    projs = []
    for name, module in model.named_modules():
        if name.endswith('.attn.c_proj') or name.endswith('.self_attn.o_proj'):
            # Extract layer index from name like 'transformer.h.3.attn.c_proj'
            # or 'model.layers.12.self_attn.o_proj'
            parts = name.split('.')
            for p in parts:
                if p.isdigit():
                    projs.append((int(p), module))
                    break
    projs.sort(key=lambda x: x[0])
    return projs


class HeadAblator:
    """Zero out a specific attention head by slicing the pre-projection tensor."""

    def __init__(self, proj_module, head_idx, head_dim):
        self.proj_module = proj_module
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.hook = None

    def _pre_hook(self, module, inputs):
        # inputs is a tuple; first element is the concatenated head output
        # shape: (batch, seq_len, n_heads * head_dim)
        x = inputs[0]
        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        x = x.clone()
        x[..., start:end] = 0.0
        return (x,) + inputs[1:]

    def __enter__(self):
        self.hook = self.proj_module.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *args):
        if self.hook is not None:
            self.hook.remove()


@torch.no_grad()
def compute_target_logit(model, tokenizer, text, target_word, control_word, device):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    max_len = getattr(model.config, 'max_position_embeddings',
                      getattr(model.config, 'n_positions', 8192))
    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, -max_len:]

    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    target_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    control_id = tokenizer.encode(control_word, add_special_tokens=False)[0]
    return float((logits[target_id] - logits[control_id]).cpu())


def run_model_ablation(model, tokenizer, prompts, n_layers, n_heads, device):
    projs = find_attention_output_projs(model)
    if len(projs) < n_layers:
        logger.warning(f"  Found {len(projs)} attention projections vs expected {n_layers} layers")
        n_layers = min(len(projs), n_layers)
    proj_by_layer = {li: m for li, m in projs}

    # Determine head dimension from weight shape
    first_proj = projs[0][1]
    # weight shape: (hidden_dim_out, hidden_dim_in); hidden_dim_in = n_heads * head_dim
    # GPT-2 Conv1D stores transposed; check both
    if hasattr(first_proj, 'weight'):
        w = first_proj.weight
        if w.dim() == 2:
            # For nn.Linear: shape (out, in)
            # For GPT-2 Conv1D: shape (in, out)
            hidden_in = max(w.shape)
        else:
            hidden_in = w.numel() // max(1, max(w.shape))
    head_dim = hidden_in // n_heads
    logger.info(f"  n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}")

    # Baseline (no ablation)
    baselines = [compute_target_logit(model, tokenizer, p['text'],
                                       p['target_word'], p['control_word'], device)
                 for p in prompts]
    mean_baseline = float(np.mean(baselines))
    logger.info(f"  Baseline logit diff: {mean_baseline:.3f}")

    effects = np.zeros((n_layers, n_heads))
    for li in tqdm(range(n_layers), desc="  Layers"):
        proj = proj_by_layer[li]
        for hi in range(n_heads):
            with HeadAblator(proj, hi, head_dim):
                ablated = [compute_target_logit(model, tokenizer, p['text'],
                                                 p['target_word'], p['control_word'], device)
                           for p in prompts]
            effects[li, hi] = mean_baseline - float(np.mean(ablated))

    return effects, mean_baseline


def run_stage5(models_arg=None):
    logger.info("=" * 70)
    logger.info("STAGE 5: Causal Head Ablation")
    logger.info("=" * 70)

    config.ensure_run_directories()
    output_dir = config.TRANSFORMER_ATTENTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from scripts.s02_llm_extraction import load_transformer_model

    features_dir = output_dir / "llm_features"
    available = [d.name for d in features_dir.iterdir() if d.is_dir()
                 and len(list(d.glob("*.npy"))) > 0] if features_dir.exists() else []

    target_models = set(models_arg.split(',')) if models_arg else None

    all_results = []
    circuits = {}

    for mc in config.TRANSFORMER_CONFIG['models']:
        name = mc['short_name']
        if name not in available:
            continue
        if target_models and name not in target_models:
            continue

        logger.info(f"\n{'=' * 50}\nCausal: {name}")
        try:
            model, tokenizer = load_transformer_model(mc)
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            continue
        if model is None:
            continue
        model.eval()

        effects, baseline = run_model_ablation(
            model, tokenizer, CAUSAL_PROMPTS,
            mc['n_layers'], mc['n_heads'], device)

        for li in range(effects.shape[0]):
            for hi in range(effects.shape[1]):
                all_results.append({
                    'model': name,
                    'layer': li,
                    'head': hi,
                    'ablation_effect': float(effects[li, hi]),
                    'baseline_logit_diff': baseline,
                })

        threshold = float(np.percentile(effects.ravel(), 95))
        critical = sorted(
            [{'layer': int(li), 'head': int(hi), 'effect': float(effects[li, hi])}
             for li in range(effects.shape[0])
             for hi in range(effects.shape[1])
             if effects[li, hi] >= threshold],
            key=lambda d: d['effect'], reverse=True)

        circuits[name] = {
            'n_critical_heads': len(critical),
            'threshold_top5pct': threshold,
            'top_heads': critical[:20],
            'mean_effect': float(np.mean(effects)),
            'max_effect': float(np.max(effects)),
            'std_effect': float(np.std(effects)),
            'baseline': baseline,
        }
        logger.info(f"  max_effect={circuits[name]['max_effect']:.3f}, "
                    f"critical={len(critical)}/{effects.size}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if all_results:
        new_df = pd.DataFrame(all_results)
        csv_path = output_dir / "ablation_results.csv"
        if csv_path.exists():
            # Merge: keep existing rows for models NOT re-run, overwrite those that are
            existing = pd.read_csv(str(csv_path))
            existing = existing[~existing['model'].isin(new_df['model'].unique())]
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_csv(str(csv_path), index=False)

    circuits_path = output_dir / "causal_circuits.json"
    if circuits_path.exists():
        with open(circuits_path) as f:
            existing_circuits = json.load(f)
        existing_circuits.update(circuits)
        circuits = existing_circuits
    with open(str(circuits_path), 'w') as f:
        json.dump(circuits, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 5 Complete")
    for name, c in circuits.items():
        logger.info(f"  {name}: max_effect={c['max_effect']:.3f}, "
                    f"top={len(c['top_heads'])}")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model short names (default: all)')
    args = parser.parse_args()
    run_stage5(models_arg=args.models)
