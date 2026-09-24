#!/usr/bin/env python3
"""
Reviewer 4, point 3: how the GPT-2 1024-token truncation enters encoding.

In Stage 2, story TRs whose words fall beyond the context window receive no
tokens and are filled with the nearest covered TR's features (i.e. the last
covered TR is repeated). This script
  1. recovers exactly which story TRs are covered for each model/condition
     by re-tokenising the model input, and
  2. re-runs encoding for the GPT-2 models with the uncovered (repeated)
     TRs dropped from both features and brain data,
so the GPT-2 numbers can be compared with the full-coverage 7B models.

Output: <run>/camera_ready/truncation_coverage.csv
        <run>/camera_ready/gpt2_truncation_encoding.csv
"""

import json
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')

from cr_common import (init_run, load_brain, load_features, lagged_pca, encode_multi,
                       brain_matrix, unpack, model_layers, TASKS, MODEL_ORDER, ROI_NAMES,
                       HRF_DELAY_TRS, config)
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

HF_NAMES = {m['short_name']: m['name'] for m in config.TRANSFORMER_CONFIG['models']}
MAX_CTX = {'GPT-2-Medium': 1024, 'GPT-2-XL': 1024}


def covered_trs(model, task, mapping, n_trs):
    with open(config.DS002345_TRANSCRIPTS / f'{task}_words.txt') as f:
        words = [l.rstrip('\n').split('\t')[2] for l in f if l.strip()]
    text = ' '.join(words)
    tok = AutoTokenizer.from_pretrained(HF_NAMES[model], cache_dir=config.MODEL_CACHE_DIR)
    offs = tok(text, return_offsets_mapping=True)['offset_mapping']
    offs = offs[:MAX_CTX.get(model, len(offs))]
    last_char = max(b for a, b in offs)
    starts, pos = [], 0
    for w in words:
        i = text.find(w, pos)
        starts.append(i)
        pos = i + len(w)
    cov = np.zeros(n_trs, bool)
    for e in mapping[task]:
        if starts[e['word_idx']] < last_char and e['tr_idx'] < n_trs:
            cov[e['tr_idx']] = True
    last = int(np.max(np.where(cov)[0]))
    story_len = max(e['tr_idx'] for e in mapping[task]) + 1
    n_words_in = sum(starts[e['word_idx']] < last_char for e in mapping[task])
    return cov, last, story_len, n_words_in, len(mapping[task])


def main():
    out = init_run()
    subjects, brain, n_trs = load_brain()
    mapping = json.load(open(config.BRAIN_ATTENTION_DIR / 'word_to_tr_mapping.json'))

    cov_rows, keep = [], {}
    for m in MODEL_ORDER:
        for t in TASKS:
            cov, last, story_len, n_in, n_all = covered_trs(m, t, mapping, n_trs)
            cov_rows.append({'model': m, 'task': t, 'story_words': n_all, 'words_in_context': n_in,
                             'pct_words': n_in / n_all, 'last_covered_tr': last,
                             'story_trs': story_len, 'pct_story_trs': (last + 1) / story_len,
                             'trs_with_words': int(cov.sum())})
            # mask over brain rows: keep a pair when its feature row (brain row - lag) is covered
            keep[(m, t)] = np.arange(n_trs) - HRF_DELAY_TRS <= last
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(out / 'truncation_coverage.csv', index=False)
    print(cov_df.round(3).to_string())

    rows = []
    for m in ['GPT-2-Medium', 'GPT-2-XL']:
        for t in TASKS:
            for layer in model_layers(m):
                feats = load_features(m, t, layer, n_trs)
                for variant, k in [('all_trs', None), ('covered_only', keep[(m, t)])]:
                    X, br = lagged_pca(feats, keep=k)
                    r = unpack(encode_multi(X, brain_matrix(brain[t], br)), len(subjects))
                    for ri, roi in enumerate(ROI_NAMES):
                        rows.append({'model': m, 'task': t, 'layer': layer, 'variant': variant,
                                     'roi': roi, 'n_trs': len(br), 'mean_r': float(np.nanmean(r[:, ri]))})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'gpt2_truncation_encoding.csv', index=False)
    best = df.loc[df.groupby(['model', 'task', 'roi', 'variant'])['mean_r'].idxmax()]
    print(best.pivot_table(index=['model', 'task', 'roi'], columns='variant',
                           values='mean_r').round(4).to_string())


if __name__ == '__main__':
    main()
