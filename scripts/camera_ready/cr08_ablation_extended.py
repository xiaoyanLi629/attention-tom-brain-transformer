#!/usr/bin/env python3
"""
Reviewer 3, point 1: stability of the "critical heads" set.

Zero-ablates every attention head (same hook as Stage 5) on the original
four prompts plus fifteen new ones, all built from the social narration and
ending just before a mental-state word, and stores the effect per prompt:
    effect = logit_diff(target, control)_baseline - logit_diff_ablated
The original four are kept verbatim so the Stage 5 numbers can be
reproduced from the per-prompt table (mean over prompts 0-3).

Output: <run>/camera_ready/ablation_per_prompt.csv
        <run>/camera_ready/ablation_prompts.csv
"""

import argparse
import gc
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')

from cr_common import init_run, MODEL_ORDER, config
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.s02_llm_extraction import load_transformer_model
from scripts.s05_causal_analysis import (CAUSAL_PROMPTS, HeadAblator,
                                         find_attention_output_projs, compute_target_logit)

# Context is taken from the social narration; the final short frame sets up a
# mental-state slot. Controls are grammatical non-mental continuations.
NEW_PROMPTS = [
    ("The boy plays with the birds, but then the flock gangs up on him and starts chasing him. "
     "He tries to fly away but cannot escape. The boy feels", ' scared', ' the'),
    ("Thankfully, the dream ends and the birds disappear. The boy sleeps peacefully for the rest "
     "of the night. The boy feels", ' safe', ' the'),
    ("The birds transform into a giant monster with two long arms and two wobbly legs. "
     "The monster swats at the boy. The boy is", ' terrified', ' standing'),
    ("The boy stays out of his reach and hovers uncertainly around the monster's feet. "
     "Suddenly he is joined by another boy. The boy is", ' relieved', ' standing'),
    ("The boys put the ball in a slingshot, shooting the ball into the monster which breaks "
     "into a heap. The boy feels", ' excited', ' the'),
    ("He goes looking for his new friend but can't find him. His friend is nowhere to be seen. "
     "The boy is", ' sad', ' standing'),
    ("He forlornly walks to the single ball. He slowly and unenthusiastically kicks the ball "
     "halfway down the yard. The boy feels", ' lonely', ' the'),
    ("He sneaks up on the boy who still doesn't see him, he taps the boy on the shoulder. "
     "The boy is", ' startled', ' standing'),
    ("The boy kicks the ball to his friend who quickly passes it back, playing together in "
     "real life just as they had in the dream. The boy feels", ' happy', ' the'),
    ("The day passes. The father waits for his son to return home. The father is", ' worried',
     ' standing'),
    ("The towering monster from the previous dream appears again, slowly following the boy and "
     "reaching for him with one arm. The boy feels", ' afraid', ' the'),
    ("Upstairs, the boy begins to dream. The ceiling opens up and he floats into the sky along "
     "with a huge bird. High above mountains, the boy can fly. The boy feels", ' free', ' the'),
    ("The boy speaks frantically as the monster reaches down. The boys fly over the monster "
     "which is still clawing at them. The boys are", ' desperate', ' standing'),
    ("Until he notices that the other boy is gone. He checks under the rubble for his friend. "
     "The boy is", ' confused', ' standing'),
    ("As he kicks the ball again, his mysterious friend from the dream appears and sneaks "
     "towards the playground. The friend wants to", ' surprise', ' kick'),
]

PROMPTS = ([{'id': i, 'set': 'original', **p} for i, p in enumerate(CAUSAL_PROMPTS)] +
           [{'id': len(CAUSAL_PROMPTS) + i, 'set': 'new', 'text': t, 'target_word': tw,
             'control_word': cw} for i, (t, tw, cw) in enumerate(NEW_PROMPTS)])


def head_dim_of(proj, n_heads):
    w = proj.weight
    return max(w.shape) // n_heads if w.dim() == 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODEL_ORDER))
    args = ap.parse_args()
    out = init_run()
    device = 'cuda'

    prompt_rows = []
    csv = out / 'ablation_per_prompt.csv'
    done = set(pd.read_csv(csv)['model']) if csv.exists() else set()

    for mc in config.TRANSFORMER_CONFIG['models']:
        name = mc['short_name']
        if name not in args.models.split(',') or name in done:
            continue
        model, tok = load_transformer_model(mc)
        model.eval()
        projs = dict(find_attention_output_projs(model))
        n_layers, n_heads = mc['n_layers'], mc['n_heads']
        hd = head_dim_of(projs[0], n_heads)

        for p in PROMPTS:
            prompt_rows.append({'model': name, 'id': p['id'], 'set': p['set'],
                                'target': p['target_word'], 'control': p['control_word'],
                                'target_n_tokens': len(tok.encode(p['target_word'], add_special_tokens=False)),
                                'control_n_tokens': len(tok.encode(p['control_word'], add_special_tokens=False))})

        base = np.array([compute_target_logit(model, tok, p['text'], p['target_word'],
                                              p['control_word'], device) for p in PROMPTS])
        rows = []
        for li in tqdm(range(n_layers), desc=name):
            for hi in range(n_heads):
                with HeadAblator(projs[li], hi, hd):
                    abl = np.array([compute_target_logit(model, tok, p['text'], p['target_word'],
                                                         p['control_word'], device) for p in PROMPTS])
                for p, b, a in zip(PROMPTS, base, abl):
                    rows.append({'model': name, 'layer': li, 'head': hi, 'prompt': p['id'],
                                 'set': p['set'], 'baseline': b, 'effect': b - a})
        df = pd.DataFrame(rows)
        df.to_csv(csv, mode='a', header=not csv.exists(), index=False)
        orig = df[df.set == 'original'].groupby(['layer', 'head'])['effect'].mean()
        print(f"{name}: baseline(orig 4)={base[:4].mean():.3f}  max effect(orig 4)={orig.max():.3f}  "
              f"baseline(all)={base.mean():.3f}", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    if prompt_rows:
        pr = pd.DataFrame(prompt_rows)
        pcsv = out / 'ablation_prompts.csv'
        pr.to_csv(pcsv, mode='a', header=not pcsv.exists(), index=False)


if __name__ == '__main__':
    main()
