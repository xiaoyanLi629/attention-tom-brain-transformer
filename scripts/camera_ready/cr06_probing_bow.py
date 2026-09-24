#!/usr/bin/env python3
"""
Reviewer 2, point 1: does the social/physical probe need more than vocabulary?

Builds bag-of-words features per TR from the same word-to-TR mapping the
LLM features use (lower-cased, punctuation stripped; TRs without words are
filled from the nearest TR, as in Stage 2) and runs the identical probe
(PCA-50, standardise, L2 logistic regression C=0.1, 5-fold temporal-block
CV, 500-shuffle permutation test).

  bow_all      full vocabulary of both narratives
  bow_shared   only words that occur in BOTH narratives, i.e. the
               condition-specific words (mental-state verbs, 'triangle',
               'circle', ...) are removed

Output: <run>/camera_ready/probing_bow.csv
"""

import json
import re

from cr_common import init_run, load_brain, TASKS, config
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.s06_probing import probe_layer, permutation_test

CONTEXT_TRS = 10


def norm(w):
    return re.sub(r"[^a-z']", '', w.lower()).strip("'")


def tr_word_lists(mapping, task, n_trs):
    bins = [[] for _ in range(n_trs)]
    for e in mapping[task]:
        if e['tr_idx'] < n_trs and norm(e['word']):
            bins[e['tr_idx']].append(norm(e['word']))
    return bins


def bow_matrix(bins, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(bins), len(vocab)))
    for t, ws in enumerate(bins):
        for w in ws:
            if w in idx:
                X[t, idx[w]] += 1
    filled = X.sum(1) > 0
    src = np.where(filled)[0]
    for t in np.where(~filled)[0]:  # nearest-neighbour fill, as for LLM features
        X[t] = X[src[np.argmin(np.abs(src - t))]]
    return X


def main():
    out = init_run()
    _, _, n_trs = load_brain()
    mapping = json.load(open(config.BRAIN_ATTENTION_DIR / 'word_to_tr_mapping.json'))
    bins = {t: tr_word_lists(mapping, t, n_trs) for t in TASKS}
    vocab = {t: {w for ws in bins[t] for w in ws} for t in TASKS}
    vocabs = {'bow_all': sorted(vocab[TASKS[0]] | vocab[TASKS[1]]),
              'bow_shared': sorted(vocab[TASKS[0]] & vocab[TASKS[1]])}

    # LLM hidden states integrate all preceding text, so also give the
    # bag-of-words the preceding CONTEXT_TRS TRs (~15 s) of words.
    def with_context(X, k=CONTEXT_TRS):
        c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), 0)
        return np.stack([c[t + 1] - c[max(0, t + 1 - k)] for t in range(len(X))])

    rows = []
    for name, v, ctx in [(n, v, c) for n, v in vocabs.items() for c in (False, True)]:
        mats = [bow_matrix(bins[t], v) for t in TASKS]
        if ctx:
            mats = [with_context(m) for m in mats]
            name = f'{name}_ctx{CONTEXT_TRS}'
        X = np.vstack(mats)
        y = np.array([1] * n_trs + [0] * n_trs)
        Xp = PCA(n_components=min(50, X.shape[0] - 1, X.shape[1]), svd_solver="full").fit_transform(X)
        acc, p, null = permutation_test(Xp, y, n_permutations=500, n_jobs=8)
        rows.append({'features': name, 'vocab_size': len(v), 'accuracy': acc, 'p_value': p,
                     'null_mean': float(np.mean(null)), 'null_95pct': float(np.percentile(null, 95))})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / 'probing_bow.csv', index=False)


if __name__ == '__main__':
    main()
