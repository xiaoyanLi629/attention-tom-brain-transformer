#!/usr/bin/env python3
"""
=============================================================================
Stage 8: Behavioral Theory-of-Mind Benchmark
=============================================================================

Evaluate each transformer model on classic ToM tasks to quantify behavioral
performance. Complements the internal-representation analyses (probing,
encoding, attention) by reporting explicit ToM accuracy.

Task format: two-alternative forced choice. For each item, the model receives
the story + question, and we compare next-token logits for " A" vs " B"
(or for the correct vs. distractor option directly). Accuracy = fraction
of items where the correct option's logit exceeds the distractor's.

Categories:
  - False belief (Sally-Anne style, 10 items)
  - Faux pas detection (6 items)
  - Intention attribution (8 items)

Outputs:
  - behavioral_tom_results.csv   — per-item accuracy
  - behavioral_tom_summary.json  — per-model accuracy by category
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

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


# ---------------------------------------------------------------------
# ToM benchmark items
# ---------------------------------------------------------------------

FALSE_BELIEF = [
    ("Sally puts her marble in the basket and leaves the room. "
     "While she is away, Anne moves the marble to the box. "
     "When Sally returns, where will she look for her marble? Answer:",
     " basket", " box"),
    ("Tom puts his chocolate in the drawer and goes outside to play. "
     "While Tom is outside, his mother moves the chocolate to the cupboard. "
     "When Tom comes back inside hungry for chocolate, where will he look first? Answer:",
     " drawer", " cupboard"),
    ("Mary hides a cookie under her pillow and goes to the kitchen. "
     "Her brother finds it and moves it to the closet. "
     "When Mary returns wanting her cookie, where will she look? Answer:",
     " pillow", " closet"),
    ("A boy leaves his toy car in the garage and goes to school. "
     "While he is gone, his sister brings the car into the living room. "
     "When the boy comes home, where will he look for his toy car? Answer:",
     " garage", " living"),
    ("Lucy puts her book on the table and leaves for lunch. "
     "Her roommate moves the book to the shelf. "
     "When Lucy returns, where does she expect the book to be? Answer:",
     " table", " shelf"),
    ("Ben puts his keys in the bowl and goes to bed. "
     "During the night, his wife moves the keys to the hook by the door. "
     "In the morning, Ben looks for his keys in the",
     " bowl", " hook"),
    ("Anna places a letter in the blue envelope and leaves. "
     "Her friend swaps the letter into the red envelope. "
     "When Anna returns, she looks for the letter in the",
     " blue", " red"),
    ("John hides his watch in the top drawer and leaves for work. "
     "His daughter moves it to the bottom drawer. "
     "When John returns to find his watch, he opens the",
     " top", " bottom"),
    ("Kate puts her phone on the desk and goes to the bathroom. "
     "Her sister moves the phone to the sofa. "
     "When Kate comes back, she thinks her phone is on the",
     " desk", " sofa"),
    ("A girl leaves her doll in the bedroom closet and goes to the park. "
     "Her mother moves the doll to the attic. "
     "When the girl returns, she first searches in the",
     " closet", " attic"),
]

# Faux pas: did the speaker know they said something wrong? Answer yes/no.
FAUX_PAS = [
    ("At a birthday party, Mike said loudly: 'What an ugly sweater!' "
     "not realizing that the host was wearing the exact same sweater. "
     "Did Mike know he was saying something rude? Answer:",
     " no", " yes"),
    ("Emma told her co-worker that she hates the new office painting. "
     "She did not know that her co-worker had painted it herself. "
     "Did Emma intend to insult her co-worker? Answer:",
     " no", " yes"),
    ("Tom laughed at a joke about bald people while his uncle, who recently lost his hair, "
     "was sitting nearby. Tom did not see his uncle come in. "
     "Did Tom mean to make his uncle feel bad? Answer:",
     " no", " yes"),
    ("At dinner, Lisa complained about how unhealthy fried food is, "
     "forgetting that her host had just served fried chicken. "
     "Did Lisa say something socially awkward? Answer:",
     " yes", " no"),
    ("David praised his colleague's presentation to another coworker, "
     "but his colleague was standing around the corner, already upset. "
     "Was David's praise something that could comfort his colleague? Answer:",
     " yes", " no"),
    ("Sarah told her friend she does not like surprise parties, "
     "not knowing that her friend had planned one for her next week. "
     "Was Sarah aware of the planned surprise? Answer:",
     " no", " yes"),
]

# Intention attribution: was the action intentional?
INTENTION = [
    ("Jake was walking down the hallway carrying a glass of water when someone bumped into him "
     "and the glass fell. Did Jake drop the glass on purpose? Answer:",
     " no", " yes"),
    ("Amy threw the ball hard at the wall because she was angry. The ball broke the window. "
     "Did Amy intend to break the window? Answer:",
     " no", " yes"),
    ("A soccer player kicked the ball toward the goal, hoping to score. The ball went in. "
     "Did the player mean to score? Answer:",
     " yes", " no"),
    ("Bob was stirring the soup when the spoon slipped and splashed onto the counter. "
     "Did Bob intentionally splash the soup? Answer:",
     " no", " yes"),
    ("Clara drew a detailed picture of a cat for hours. She showed it to her teacher. "
     "Did Clara plan to create the drawing? Answer:",
     " yes", " no"),
    ("Paul kicked his brother under the table, hoping nobody would notice. "
     "Was Paul's action deliberate? Answer:",
     " yes", " no"),
    ("While riding a bike, Sam hit a rock and fell, scraping his knee. "
     "Did Sam fall on purpose? Answer:",
     " no", " yes"),
    ("Kevin rehearsed his speech for three weeks before the ceremony. "
     "Was his good performance intentional? Answer:",
     " yes", " no"),
]


def build_items():
    items = []
    for idx, (text, correct, distractor) in enumerate(FALSE_BELIEF):
        items.append({'id': f'fb_{idx:02d}', 'category': 'false_belief',
                      'text': text, 'correct': correct, 'distractor': distractor})
    for idx, (text, correct, distractor) in enumerate(FAUX_PAS):
        items.append({'id': f'fp_{idx:02d}', 'category': 'faux_pas',
                      'text': text, 'correct': correct, 'distractor': distractor})
    for idx, (text, correct, distractor) in enumerate(INTENTION):
        items.append({'id': f'in_{idx:02d}', 'category': 'intention',
                      'text': text, 'correct': correct, 'distractor': distractor})
    return items


@torch.no_grad()
def score_item(model, tokenizer, text, correct, distractor, device):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    ids = inputs['input_ids']
    max_len = getattr(model.config, 'max_position_embeddings',
                      getattr(model.config, 'n_positions', 8192))
    if ids.shape[1] > max_len:
        ids = ids[:, -max_len:]
    # Cast logits to fp32 to avoid fp16 NaN/overflow in large-vocab models
    logits = model(ids).logits[0, -1, :].float()
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        # Replace non-finite with the finite min so relative comparisons still work
        finite = logits[torch.isfinite(logits)]
        fill = float(finite.min().item()) if finite.numel() > 0 else 0.0
        logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, fill))
    c_id = tokenizer.encode(correct, add_special_tokens=False)[0]
    d_id = tokenizer.encode(distractor, add_special_tokens=False)[0]
    diff = float((logits[c_id] - logits[d_id]).cpu())
    return diff, int(diff > 0)


def run_stage8(models_arg=None):
    logger.info("=" * 70)
    logger.info("STAGE 8: Behavioral ToM Benchmark")
    logger.info("=" * 70)

    config.ensure_run_directories()
    out_dir = config.INTEGRATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from scripts.s02_llm_extraction import load_transformer_model

    items = build_items()
    logger.info(f"Loaded {len(items)} items "
                f"(false_belief={len(FALSE_BELIEF)}, "
                f"faux_pas={len(FAUX_PAS)}, "
                f"intention={len(INTENTION)})")

    features_dir = config.TRANSFORMER_ATTENTION_DIR / "llm_features"
    available = [d.name for d in features_dir.iterdir() if d.is_dir()
                 and len(list(d.glob("*.npy"))) > 0] if features_dir.exists() else []
    target_models = set(models_arg.split(',')) if models_arg else None

    rows = []
    summary = {}

    for mc in config.TRANSFORMER_CONFIG['models']:
        name = mc['short_name']
        if name not in available:
            continue
        if target_models and name not in target_models:
            continue

        logger.info(f"\n{'=' * 50}\n{name}")
        try:
            model, tokenizer = load_transformer_model(mc)
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            continue
        if model is None:
            continue
        model.eval()

        for item in tqdm(items, desc=f"  {name}"):
            diff, correct = score_item(model, tokenizer, item['text'],
                                        item['correct'], item['distractor'], device)
            rows.append({
                'model': name,
                'id': item['id'],
                'category': item['category'],
                'logit_diff': diff,
                'correct': correct,
            })

        df_model = pd.DataFrame([r for r in rows if r['model'] == name])
        summary[name] = {
            'overall_acc': float(df_model['correct'].mean()),
            'false_belief_acc': float(
                df_model[df_model['category'] == 'false_belief']['correct'].mean()),
            'faux_pas_acc': float(
                df_model[df_model['category'] == 'faux_pas']['correct'].mean()),
            'intention_acc': float(
                df_model[df_model['category'] == 'intention']['correct'].mean()),
            'n_items': int(len(df_model)),
        }
        logger.info(f"  Overall: {summary[name]['overall_acc']:.2%} "
                    f"(FB={summary[name]['false_belief_acc']:.2%}, "
                    f"FP={summary[name]['faux_pas_acc']:.2%}, "
                    f"INT={summary[name]['intention_acc']:.2%})")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(str(out_dir / "behavioral_tom_results.csv"), index=False)
    with open(str(out_dir / "behavioral_tom_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 8 Complete")
    for name, s in summary.items():
        logger.info(f"  {name}: {s['overall_acc']:.2%} overall")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model short names')
    args = parser.parse_args()
    run_stage8(models_arg=args.models)
