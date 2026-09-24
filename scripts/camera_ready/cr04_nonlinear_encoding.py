#!/usr/bin/env python3
"""
Reviewer 4, point 1: is the weak alignment a limitation of linear ridge?

At each model's group-best layer (per ROI x condition, from cr02), compare
under the identical 5-fold temporal-block CV and 3-TR lag:
  ridge_pca50   the paper's encoder (reference)
  ridge_pca200  linear, 4x more PCA components (is PCA-50 the bottleneck?)
  krr_rbf       kernel ridge, RBF kernel, alpha/gamma tuned by inner
                contiguous-block CV on the training folds only
  mlp           one hidden layer (64 units), L2 tuned by inner CV,
                mPFC and rTPJ only (slow)

Output: <run>/camera_ready/nonlinear_encoding.csv  (per subject)
        <run>/camera_ready/nonlinear_encoding_summary.csv
"""

from cr_common import (init_run, load_brain, load_features, lagged_pca, temporal_block_cv,
                       ROI_NAMES, TASKS, MODEL_ORDER, RIDGE_ALPHAS, config)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MLP_ROIS = ['mPFC', 'rTPJ']
INNER = KFold(n_splits=4, shuffle=False)  # contiguous blocks inside the training folds


def cv_predict(make, X, y):
    pred = np.zeros_like(y)
    for tr, te in temporal_block_cv(len(y)):
        pred[te] = make(X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
    return float(np.corrcoef(pred, y)[0, 1])


def ridge(d):
    return RidgeCV(alphas=RIDGE_ALPHAS)


def krr(d):
    grid = {'kernelridge__alpha': [0.1, 1, 10, 100],
            'kernelridge__gamma': [0.05 / d, 0.2 / d, 1.0 / d]}
    return GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel='rbf')),
                        grid, cv=INNER, scoring='r2')


def mlp(d):
    grid = {'mlpregressor__alpha': [0.1, 1.0, 10.0]}
    net = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, early_stopping=False,
                       random_state=0)
    return GridSearchCV(make_pipeline(StandardScaler(), net), grid, cv=INNER, scoring='r2')


def job(model, task, roi, layer, brain_task, subjects, n_trs):
    feats = load_features(model, task, layer, n_trs)
    X50, rows = lagged_pca(feats, n_components=50)
    X200, _ = lagged_pca(feats, n_components=200)
    ri = ROI_NAMES.index(roi)
    out = []
    for si, sub in enumerate(subjects):
        y = brain_task[si, rows, ri]
        if y.std() < 1e-10:
            continue
        rec = {'model': model, 'task': task, 'roi': roi, 'layer': layer, 'subject': sub,
               'ridge_pca50': cv_predict(ridge, X50, y),
               'ridge_pca200': cv_predict(ridge, X200, y),
               'krr_rbf': cv_predict(krr, X50, y)}
        if roi in MLP_ROIS:
            rec['mlp'] = cv_predict(mlp, X50, y)
        out.append(rec)
    return out


def main():
    out = init_run()
    subjects, brain, n_trs = load_brain()
    sig = pd.read_csv(out / 'encoding_significance.csv')
    jobs = [(r.model, r.task, r.roi, int(r.group_best_layer)) for r in sig.itertuples()]
    res = Parallel(n_jobs=8, verbose=5)(
        delayed(job)(m, t, roi, l, brain[t], subjects, n_trs) for m, t, roi, l in jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(out / 'nonlinear_encoding.csv', index=False)

    methods = ['ridge_pca50', 'ridge_pca200', 'krr_rbf', 'mlp']
    summ = []
    for (m, t, roi), g in df.groupby(['model', 'task', 'roi']):
        rec = {'model': m, 'task': t, 'roi': roi, 'layer': int(g.layer.iloc[0])}
        for k in methods:
            if k in g and g[k].notna().any():
                rec[k] = g[k].mean()
                if k != 'ridge_pca50':
                    d = (g[k] - g['ridge_pca50']).dropna()
                    rec[f'{k}_minus_ridge_p'] = stats.wilcoxon(d).pvalue
        summ.append(rec)
    s = pd.DataFrame(summ)
    s.to_csv(out / 'nonlinear_encoding_summary.csv', index=False)
    print(s.round(4).to_string())


if __name__ == '__main__':
    main()
