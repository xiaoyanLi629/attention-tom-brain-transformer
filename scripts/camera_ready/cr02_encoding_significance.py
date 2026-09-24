#!/usr/bin/env python3
"""
Reviewer 4, point 4: is the best-layer encoding r itself above zero?

Picking the best layer and then testing it on the same subjects is biased
upward, so the layer is chosen leave-one-subject-out: for each held-out
subject, the best layer is the one with the highest mean r over the other
57 subjects, and the held-out subject contributes its r at that layer.
The resulting 58 values get a one-sided one-sample t-test, a Wilcoxon
signed-rank test and a bootstrap 95% CI of the mean. BH-FDR is applied
over all model x ROI x condition cells.

The group-best-layer mean (what Fig. 3C and the heatmap display) is
reported alongside for reference.

Input:  <run>/cross_domain/encoding_results.csv (Stage 3)
Output: <run>/camera_ready/encoding_significance.csv
"""

from cr_common import init_run, bh_fdr, config
import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10000


def main():
    out = init_run()
    df = pd.read_csv(config.CROSS_DOMAIN_DIR / 'encoding_results.csv')
    rng = np.random.default_rng(0)
    rows = []
    for (model, roi, task), g in df.groupby(['model', 'roi', 'task']):
        P = g.pivot_table(index='subject', columns='layer', values='encoding_r')
        R = P.values
        n = len(R)
        loso = np.empty(n)
        for i in range(n):
            loso[i] = R[i, np.nanargmax(np.nanmean(np.delete(R, i, 0), 0))]
        best_col = int(np.nanargmax(np.nanmean(R, 0)))
        boot = rng.choice(loso, size=(N_BOOT, n)).mean(1)
        t, p_t = stats.ttest_1samp(loso, 0, alternative='greater')
        _, p_w = stats.wilcoxon(loso, alternative='greater')
        rows.append({'model': model, 'roi': roi, 'task': task, 'n_subjects': n,
                     'group_best_layer': int(P.columns[best_col]),
                     'group_best_mean_r': float(np.nanmean(R[:, best_col])),
                     'loso_mean_r': float(loso.mean()),
                     'loso_ci_low': float(np.percentile(boot, 2.5)),
                     'loso_ci_high': float(np.percentile(boot, 97.5)),
                     't': float(t), 'p_ttest': float(p_t), 'p_wilcoxon': float(p_w),
                     'n_subjects_r_gt_0': int((loso > 0).sum())})
    res = pd.DataFrame(rows)
    res['p_fdr'] = bh_fdr(res['p_ttest'])
    res.to_csv(out / 'encoding_significance.csv', index=False)
    print(res.sort_values(['roi', 'task', 'model']).round(4).to_string())


if __name__ == '__main__':
    main()
