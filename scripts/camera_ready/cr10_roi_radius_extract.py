#!/usr/bin/env python3
"""
Reviewer 1, point 2: ROI-size sensitivity.

Re-runs the Stage 1 preprocessing (MNI resampling, 6 mm smoothing, high-pass,
detrend, z-score, intro trim) for every subject x task and extracts the six
social-brain ROIs at several sphere radii in one pass. The original radii
(12 mm; 10 mm for STS) are extracted too, so the output can be checked
against the Stage 1 time series.

Outputs: <run>/camera_ready/roi_radius/r{radius}/{sub}_{task}.npy  (n_trs, 6)
         <run>/camera_ready/roi_radius/voxel_counts.csv
"""

import argparse
import os

from cr_common import init_run, TASKS, config

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.s01_fmri_preprocessing import (SOCIAL_BRAIN_ROIS, INTRO_TRS, create_spherical_roi,
                                            extract_roi_timeseries, preprocess_bold)
from nilearn import image, datasets

# 'orig' reproduces Stage 1 (12 mm, STS 10 mm)
RADII = {'r6': 6, 'r8': 8, 'orig': None}


def masks_for(template):
    shape, affine = template.shape[:3], template.affine
    out = {}
    for tag, rad in RADII.items():
        out[tag] = {name: create_spherical_roi(shape, affine, info['center'],
                                               rad if rad is not None else info['radius'])
                    for name, info in SOCIAL_BRAIN_ROIS.items()}
    return out


def one(sub, task, out_dir):
    targets = {tag: out_dir / tag / f'{sub}_{task}.npy' for tag in RADII}
    if all(p.exists() for p in targets.values()):
        return sub, task, 'cached'
    template = datasets.load_mni152_template(resolution=2)
    masks = masks_for(template)
    bold = config.DS002345_FMRI / sub / 'func' / f'{sub}_task-{task}_bold.nii.gz'
    img = preprocess_bold(bold, template)
    img = image.index_img(img, slice(INTRO_TRS, None))
    for tag, m in masks.items():
        np.save(str(targets[tag]), extract_roi_timeseries(img, m))
    return sub, task, 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=8)
    args = ap.parse_args()

    out_dir = init_run() / 'roi_radius'
    for tag in RADII:
        (out_dir / tag).mkdir(parents=True, exist_ok=True)

    template = datasets.load_mni152_template(resolution=2)
    rows = [{'radius': tag, 'roi': n, 'n_voxels': int(m.sum())}
            for tag, ms in masks_for(template).items() for n, m in ms.items()]
    pd.DataFrame(rows).to_csv(out_dir / 'voxel_counts.csv', index=False)

    jobs = [(s, t) for s in config.get_subject_list() for t in TASKS]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one)(s, t, out_dir) for s, t in jobs)
    print(pd.Series([r[2] for r in res]).value_counts())


if __name__ == '__main__':
    main()
