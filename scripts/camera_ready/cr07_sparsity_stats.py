#!/usr/bin/env python3
"""
Reviewer 3, point 3: put a number on "slightly higher sparsity for social".

Per model and metric (Gini, normalised entropy, top-5 concentration), a
paired comparison of social vs physical over matched (layer, head) units:
Wilcoxon signed-rank test, paired t-test, Cohen's d_z, and the fraction of
heads moving in the "sparser for social" direction. Because heads within a
layer are not independent, the same test is repeated on layer means
(n = number of layers), which is the conservative version.

Input:  <run>/transformer_attention/llm_attention/{model}/{task}_sparsity.csv
Output: <run>/camera_ready/sparsity_stats.csv
"""

from cr_common import init_run, bh_fdr, MODEL_ORDER, config
import numpy as np
import pandas as pd
from scipy import stats

# sign: +1 if a larger value means sparser
METRICS = {'gini': +1, 'entropy': -1, 'top5_concentration': +1}


def main():
    out = init_run()
    rows = []
    for m in MODEL_ORDER:
        d = config.TRANSFORMER_ATTENTION_DIR / 'llm_attention' / m
        s = pd.read_csv(d / 'shapessocial_sparsity.csv')
        p = pd.read_csv(d / 'shapesphysical_sparsity.csv')
        mg = s.merge(p, on=['layer', 'head'], suffixes=('_s', '_p'))
        for metric, sign in METRICS.items():
            for unit, frame in [('head', mg),
                                ('layer', mg.groupby('layer')[[f'{metric}_s', f'{metric}_p']].mean())]:
                a, b = frame[f'{metric}_s'].values, frame[f'{metric}_p'].values
                diff = a - b
                rows.append({'model': m, 'metric': metric, 'unit': unit, 'n': len(diff),
                             'social_mean': a.mean(), 'physical_mean': b.mean(),
                             'mean_diff': diff.mean(),
                             'dz': diff.mean() / diff.std(ddof=1),
                             'pct_sparser_social': float(np.mean(sign * diff > 0)),
                             'p_wilcoxon': stats.wilcoxon(a, b).pvalue,
                             'p_ttest': stats.ttest_rel(a, b).pvalue})
    df = pd.DataFrame(rows)
    for unit in ['head', 'layer']:
        mask = df.unit == unit
        df.loc[mask, 'p_fdr'] = bh_fdr(df.loc[mask, 'p_wilcoxon'])
    df.to_csv(out / 'sparsity_stats.csv', index=False)
    print(df.round(4).to_string())


if __name__ == '__main__':
    main()
