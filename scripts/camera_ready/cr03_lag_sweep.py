#!/usr/bin/env python3
"""
Reviewer 1, point 3: robustness of encoding to the HRF shift.

Re-runs the Stage 3 encoding at feature lags of 0-6 TRs (0-9 s) for every
model, layer, ROI, condition and subject. Lag 3 (4.5 s) is the value used in
the paper.

Output: <run>/camera_ready/lag_sweep.csv  (subject-mean encoding r)
        <run>/camera_ready/lag_sweep_best.csv  (best layer per model x ROI x task x lag)
"""

from cr_common import (init_run, load_brain, load_features, lagged_pca, encode_multi,
                       brain_matrix, unpack, model_layers, TASKS, MODEL_ORDER, ROI_NAMES)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

LAGS = range(0, 7)


def job(model, task, layer, brain_task, n_trs, n_sub):
    feats = load_features(model, task, layer, n_trs)
    rows = []
    for lag in LAGS:
        X, br = lagged_pca(feats, lag=lag)
        r = unpack(encode_multi(X, brain_matrix(brain_task, br)), n_sub)
        for ri, roi in enumerate(ROI_NAMES):
            rows.append({'model': model, 'task': task, 'layer': layer, 'lag': lag,
                         'roi': roi, 'mean_r': float(np.nanmean(r[:, ri]))})
    return rows


def main():
    out = init_run()
    subjects, brain, n_trs = load_brain()
    jobs = [(m, t, l) for m in MODEL_ORDER for t in TASKS for l in model_layers(m)]
    res = Parallel(n_jobs=8, verbose=2)(
        delayed(job)(m, t, l, brain[t], n_trs, len(subjects)) for m, t, l in jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(out / 'lag_sweep.csv', index=False)
    best = (df.loc[df.groupby(['model', 'task', 'roi', 'lag'])['mean_r'].idxmax()]
              .sort_values(['roi', 'task', 'model', 'lag']))
    best.to_csv(out / 'lag_sweep_best.csv', index=False)
    print(best[best.roi.isin(['mPFC', 'rTPJ'])].pivot_table(
        index=['roi', 'task', 'model'], columns='lag', values='mean_r').round(3).to_string())


if __name__ == '__main__':
    main()
