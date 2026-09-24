#!/usr/bin/env python3
"""
Reviewer 3, points 1 and 4.

Stability (point 1), per model, from the per-prompt ablation table:
  - Jaccard overlap of the per-prompt top-5% head sets among the original
    four prompts (pairwise mean)
  - split-half Jaccard of the top-5% set (mean effect over a random half of
    the 19 prompts vs the other half, 500 splits), against the Jaccard
    expected for two random sets of the same size
  - how many of the original 4-prompt critical heads remain critical when
    all 19 prompts are used
  - layer-depth distribution (early/middle/late thirds) of the 19-prompt set

Cross-reference (point 4): per layer, the number of 19-prompt critical heads
vs the subject-mean encoding r (social narrative) in mPFC and rTPJ, with a
Spearman correlation, and the relative depth of the critical-head centre of
mass vs the encoding peak.

Output: <run>/camera_ready/ablation_stability.csv
        <run>/camera_ready/ablation_layer_crossref.csv
"""

from cr_common import init_run, MODEL_ORDER, config
import numpy as np
import pandas as pd
from scipy import stats

TOP = 95  # percentile
N_SPLIT = 500


def top_set(effects):
    thr = np.percentile(effects.values, TOP)
    return set(effects.index[effects.values >= thr])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


def main():
    out = init_run()
    abl = pd.read_csv(out / 'ablation_per_prompt.csv')
    enc = pd.read_csv(config.CROSS_DOMAIN_DIR / 'encoding_results.csv')
    rng = np.random.default_rng(0)
    stab, cross = [], []

    for m in MODEL_ORDER:
        a = abl[abl.model == m]
        if a.empty:
            continue
        E = a.pivot_table(index=['layer', 'head'], columns='prompt', values='effect')
        n_layers = a.layer.max() + 1
        orig = [p for p in E.columns if p < 4]
        per_prompt = [top_set(E[p]) for p in orig]
        pair_j = [jaccard(per_prompt[i], per_prompt[j])
                  for i in range(len(orig)) for j in range(i + 1, len(orig))]
        set4, set19 = top_set(E[orig].mean(1)), top_set(E.mean(1))
        prompts = np.array(E.columns)
        split_j = []
        for _ in range(N_SPLIT):
            perm = rng.permutation(prompts)
            h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
            split_j.append(jaccard(top_set(E[h1].mean(1)), top_set(E[h2].mean(1))))
        k, n = len(set19), len(E)
        chance_j = (k * k / n) / (2 * k - k * k / n)
        rho = E.corr(method='spearman').values
        depth = np.array([l for l, h in set19]) / n_layers
        mean19 = E.mean(1)
        top_head = mean19.idxmax()
        stab.append({'model': m, 'n_heads': n, 'n_critical': k,
                     'orig4_pairwise_jaccard': float(np.mean(pair_j)),
                     'split_half_jaccard': float(np.mean(split_j)),
                     'split_half_jaccard_sd': float(np.std(split_j)),
                     'chance_jaccard': chance_j,
                     'orig4_critical_retained_in_19': len(set4 & set19) / len(set4),
                     'mean_pairwise_spearman_19': float(rho[np.triu_indices_from(rho, 1)].mean()),
                     'pct_early': float(np.mean(depth < 1 / 3)),
                     'pct_middle': float(np.mean((depth >= 1 / 3) & (depth < 2 / 3))),
                     'pct_late': float(np.mean(depth >= 2 / 3)),
                     'top_head_19': f'L{top_head[0]}H{top_head[1]}',
                     'top_effect_19': float(mean19.max()),
                     'baseline_19': float(a.groupby('prompt').baseline.first().mean()),
                     'critical_depth_com': float(depth.mean())})

        counts = pd.Series([l for l, h in set19]).value_counts().reindex(range(n_layers), fill_value=0)
        for roi in ['mPFC', 'rTPJ']:
            g = enc[(enc.model == m) & (enc.roi == roi) & (enc.task == 'shapessocial')]
            prof = g.groupby('layer').encoding_r.mean().reindex(range(n_layers))
            ok = prof.notna()
            r, p = stats.spearmanr(counts[ok], prof[ok])
            cross.append({'model': m, 'roi': roi, 'spearman_rho': r, 'p': p,
                          'encoding_peak_layer': int(prof.idxmax()),
                          'encoding_peak_depth': float(prof.idxmax() / n_layers),
                          'critical_depth_com': float(depth.mean()),
                          'encoding_r_at_critical_layers': float(
                              np.average(prof.fillna(0), weights=counts)),
                          'encoding_r_mean_all_layers': float(prof.mean())})

    s = pd.DataFrame(stab)
    s.to_csv(out / 'ablation_stability.csv', index=False)
    c = pd.DataFrame(cross)
    c.to_csv(out / 'ablation_layer_crossref.csv', index=False)
    print(s.round(3).to_string())
    print(c.round(3).to_string())


if __name__ == '__main__':
    main()
