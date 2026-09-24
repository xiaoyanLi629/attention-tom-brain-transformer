#!/usr/bin/env python3
"""
Reviewer 1, point 1: is the ROI signal reliable enough to interpret?

For each ROI x condition: leave-one-out ISC per subject (the conservative
noise-ceiling bound used in the paper), a one-sample t-test of subject ISC
against zero, and a circular-shift null for the group-mean ISC (each
subject's series is independently rotated by a random offset of at least
20 TRs, which preserves autocorrelation but breaks stimulus locking).

Output: <run>/camera_ready/isc_reliability.csv
"""

from cr_common import init_run, load_brain, TASKS, ROI_NAMES
import numpy as np
import pandas as pd
from scipy import stats

N_PERM = 5000
MIN_SHIFT = 20


def loo_isc(D):
    """D: (n_sub, n_trs). Pearson r of each subject with the mean of the others."""
    Z = (D - D.mean(1, keepdims=True)) / D.std(1, keepdims=True)
    total = Z.sum(0)
    out = np.empty(len(Z))
    for i in range(len(Z)):
        others = (total - Z[i]) / (len(Z) - 1)
        out[i] = np.corrcoef(Z[i], others)[0, 1]
    return out


def main():
    out = init_run()
    subjects, brain, n_trs = load_brain()
    rng = np.random.default_rng(0)
    rows = []
    for task in TASKS:
        for ri, roi in enumerate(ROI_NAMES):
            D = brain[task][:, :, ri]
            D = D[D.std(1) > 1e-10]
            isc = loo_isc(D)
            t, p_t = stats.ttest_1samp(isc, 0, alternative='greater')
            obs = isc.mean()
            null = np.empty(N_PERM)
            for k in range(N_PERM):
                shifts = rng.integers(MIN_SHIFT, n_trs - MIN_SHIFT, size=len(D))
                Ds = np.stack([np.roll(d, s) for d, s in zip(D, shifts)])
                null[k] = loo_isc(Ds).mean()
            rows.append({'task': task, 'roi': roi, 'n_subjects': len(D),
                         'loo_isc_mean': obs, 'loo_isc_median': float(np.median(isc)),
                         't': t, 'p_ttest': p_t,
                         'p_circshift': (1 + (null >= obs).sum()) / (1 + N_PERM),
                         'null_95pct': float(np.percentile(null, 95))})
            print(rows[-1])
    pd.DataFrame(rows).to_csv(out / 'isc_reliability.csv', index=False)


if __name__ == '__main__':
    main()
