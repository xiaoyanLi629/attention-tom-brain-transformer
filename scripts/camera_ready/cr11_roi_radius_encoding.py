#!/usr/bin/env python3
"""
Reviewer 1, point 2 (second half): encoding at smaller ROI radii.

Uses the time series from cr10 (6 mm, 8 mm and the original radii
re-extracted) and re-runs the Stage 3 encoding for every model, layer,
ROI, condition and subject. Also reports the LOO-ISC ceiling per radius,
since smaller spheres average fewer voxels and are noisier.

Output: <run>/camera_ready/roi_radius_encoding.csv  (subject-mean r per layer)
        <run>/camera_ready/roi_radius_summary.csv   (best layer, LOO ISC)
"""

from cr_common import (init_run, load_brain, load_features, lagged_pca, encode_multi,
                       brain_matrix, unpack, model_layers, TASKS, MODEL_ORDER, ROI_NAMES)
from cr01_isc_reliability import loo_isc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

TAGS = ['orig', 'r8', 'r6']


def job(model, task, layer, brains, n_trs, n_sub):
    feats = load_features(model, task, layer, n_trs)
    X, br = lagged_pca(feats)
    rows = []
    for tag, b in brains.items():
        r = unpack(encode_multi(X, brain_matrix(b[task], br)), n_sub)
        for ri, roi in enumerate(ROI_NAMES):
            rows.append({'radius': tag, 'model': model, 'task': task, 'layer': layer,
                         'roi': roi, 'mean_r': float(np.nanmean(r[:, ri]))})
    return rows


def main():
    out = init_run()
    brains, n_trs = {}, None
    for tag in TAGS:
        subjects, b, n = load_brain(out / 'roi_radius' / tag)
        brains[tag], n_trs = b, n if n_trs is None else min(n_trs, n)
    brains = {k: {t: v[t][:, :n_trs] for t in TASKS} for k, v in brains.items()}

    jobs = [(m, t, l) for m in MODEL_ORDER for t in TASKS for l in model_layers(m)]
    res = Parallel(n_jobs=8, verbose=2)(
        delayed(job)(m, t, l, brains, n_trs, len(subjects)) for m, t, l in jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(out / 'roi_radius_encoding.csv', index=False)

    best = df.loc[df.groupby(['radius', 'model', 'task', 'roi'])['mean_r'].idxmax()]
    def ceiling(D):
        return float(np.mean(loo_isc(D[D.std(1) > 1e-10])))

    isc = pd.DataFrame([{'radius': tag, 'task': t, 'roi': roi,
                         'loo_isc': ceiling(brains[tag][t][:, :, ri])}
                        for tag in TAGS for t in TASKS for ri, roi in enumerate(ROI_NAMES)])
    summ = best.merge(isc, on=['radius', 'task', 'roi'])
    summ.to_csv(out / 'roi_radius_summary.csv', index=False)
    print(summ.pivot_table(index=['roi', 'task', 'model'], columns='radius',
                           values='mean_r').round(4).to_string())
    print(isc.pivot_table(index=['roi', 'task'], columns='radius', values='loo_isc').round(3))


if __name__ == '__main__':
    main()
