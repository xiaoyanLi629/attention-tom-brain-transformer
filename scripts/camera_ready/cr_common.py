"""
Shared helpers for the camera-ready (reviewer-response) analyses.

All analyses read from one run directory, selected with the
CURRENT_RUN_TIMESTAMP environment variable exactly like the main pipeline,
and write to <run>/camera_ready/.

The encoder here is a vectorised version of s03.fit_encoding_model: one
RidgeCV with alpha_per_target=True fits every subject x ROI column at once.
Because RidgeCV selects alpha independently per target by efficient
leave-one-out GCV, each column's prediction is identical to fitting that
column on its own (checked by cr00_check_encoder.py).
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from configs import config
from scripts.s03_encoding_models import (temporal_block_cv, ROI_NAMES, HRF_DELAY_TRS,
                                         PCA_COMPONENTS, RIDGE_ALPHAS, N_FOLDS)

TASKS = ['shapessocial', 'shapesphysical']

# Resolve run directories at import so joblib worker processes (which import
# this module afresh) see the same paths as the parent.
config.ensure_run_directories()
MODEL_ORDER = ['GPT-2-Medium', 'GPT-2-XL', 'Mistral-7B', 'Qwen2-7B']


def init_run():
    config.ensure_run_directories()
    out = config.CURRENT_RUN_DIR / 'camera_ready'
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_brain(roi_dir=None):
    """Return subjects, {task: (n_sub, n_trs, n_roi)} truncated to the common TR count."""
    roi_dir = Path(roi_dir) if roi_dir else config.BRAIN_ATTENTION_DIR / 'roi_timeseries'
    subjects = config.get_subject_list()
    raw = {t: [np.load(str(roi_dir / f'{s}_{t}.npy')) for s in subjects] for t in TASKS}
    min_trs = min(a.shape[0] for t in TASKS for a in raw[t])
    brain = {t: np.stack([a[:min_trs] for a in raw[t]]) for t in TASKS}
    return subjects, brain, min_trs


def model_layers(model):
    d = config.TRANSFORMER_ATTENTION_DIR / 'llm_features' / model
    return sorted({int(p.stem.split('layer')[1]) for p in d.glob('shapessocial_layer*.npy')})


def load_features(model, task, layer, n_trs):
    f = config.TRANSFORMER_ATTENTION_DIR / 'llm_features' / model / f'{task}_layer{layer}.npy'
    return np.load(str(f))[:n_trs]


def lagged_pca(X, lag=HRF_DELAY_TRS, n_components=PCA_COMPONENTS, keep=None):
    """Shift features by `lag` TRs (feature row i predicts brain row i+lag), then PCA.

    keep: optional boolean mask over brain rows (length n_trs) selecting which
    (feature, brain) pairs to use, applied after the shift.
    Returns X_pca and the brain-row index array to pair with it.
    """
    n = X.shape[0]
    brain_rows = np.arange(lag, n)
    Xs = X[:n - lag] if lag > 0 else X
    if keep is not None:
        sel = keep[brain_rows]
        Xs, brain_rows = Xs[sel], brain_rows[sel]
    k = min(n_components, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=k, svd_solver="full").fit_transform(Xs), brain_rows


def encode_multi(X, Y, n_folds=N_FOLDS):
    """Temporal-block-CV ridge for many targets at once. Y: (n, T). Returns r per target."""
    pred = np.zeros_like(Y, dtype=float)
    for tr, te in temporal_block_cv(len(Y), n_folds):
        m = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
    return columnwise_r(pred, Y)


def columnwise_r(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(0) / den


def brain_matrix(brain_task, rows):
    """(n_sub, n_trs, n_roi) -> (len(rows), n_sub*n_roi), column = sub*n_roi + roi."""
    b = brain_task[:, rows, :]
    return b.transpose(1, 0, 2).reshape(len(rows), -1)


def unpack(r, n_sub, n_roi=len(ROI_NAMES)):
    return r.reshape(n_sub, n_roi)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out
