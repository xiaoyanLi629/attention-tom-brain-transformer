#!/usr/bin/env python3
"""
Parallel re-run of Stage 1 MNI resampling.
Skips already-completed scans. Uses joblib for multiprocessing.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets as ni_datasets
import json
import logging
from joblib import Parallel, delayed

from configs import config
from scripts.s01_fmri_preprocessing import (
    SOCIAL_BRAIN_ROIS, ROI_NAMES, TR, INTRO_TRS, HIGH_PASS_FREQ, SMOOTH_FWHM,
    create_spherical_roi, extract_roi_timeseries, build_word_to_tr_mapping,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')
logger = logging.getLogger(__name__)

N_JOBS = 8


def process_one(sub_id, task_name, roi_masks, output_dir, mni_template):
    """Process a single subject × task with MNI resampling."""
    out_path = output_dir / f"{sub_id}_{task_name}.npy"
    if out_path.exists():
        return None  # skip already done

    bold_path = (config.DS002345_FMRI / sub_id / "func" /
                 f"{sub_id}_task-{task_name}_bold.nii.gz")
    if not bold_path.exists():
        return None

    try:
        img = nib.load(str(bold_path))
        img_mni = image.resample_to_img(img, mni_template, interpolation='continuous')
        img_smooth = image.smooth_img(img_mni, fwhm=SMOOTH_FWHM)
        img_clean = image.clean_img(
            img_smooth, detrend=True, standardize='zscore_sample',
            high_pass=HIGH_PASS_FREQ, t_r=TR)

        total_trs = img_clean.shape[3]
        if total_trs <= INTRO_TRS:
            return None

        img_trimmed = image.index_img(img_clean, slice(INTRO_TRS, None))
        ts = extract_roi_timeseries(img_trimmed, roi_masks)
        np.save(str(out_path), ts)

        story_trs = img_trimmed.shape[3]
        logger.info(f"{sub_id} {task_name}: {total_trs}→{story_trs} TRs OK")

        qc = {'subject': sub_id, 'task': task_name,
              'total_trs_raw': total_trs, 'story_trs': story_trs}
        for i, name in enumerate(ROI_NAMES):
            qc[f'{name}_mean'] = float(np.nanmean(ts[:, i]))
            qc[f'{name}_std'] = float(np.nanstd(ts[:, i]))
        return qc

    except Exception as e:
        logger.error(f"{sub_id} {task_name}: FAILED - {e}")
        return None


def main():
    config.initialize_run_directories('20260415_stage1')
    output_dir = config.BRAIN_ATTENTION_DIR / "roi_timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = config.get_subject_list()
    tasks = config.SHAPES_TASK['task_names']
    logger.info(f"Subjects: {len(subjects)}, Tasks: {tasks}")

    # Count remaining
    total_jobs = []
    for sub in subjects:
        for task in tasks:
            out = output_dir / f"{sub}_{task}.npy"
            if not out.exists():
                total_jobs.append((sub, task))
    logger.info(f"Remaining: {len(total_jobs)}/{len(subjects)*len(tasks)}")

    if not total_jobs:
        logger.info("All done, nothing to process!")
        return

    # Load MNI template
    mni_template = ni_datasets.load_mni152_template(resolution=2)

    # Build ROI masks in MNI space
    roi_masks = {}
    for roi_name, roi_info in SOCIAL_BRAIN_ROIS.items():
        mask = create_spherical_roi(mni_template.shape[:3], mni_template.affine,
                                    roi_info['center'], roi_info['radius'])
        roi_masks[roi_name] = mask
        logger.info(f"  ROI {roi_name}: {mask.sum()} voxels")

    # Parallel processing
    logger.info(f"Starting {N_JOBS} parallel workers...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_one)(sub, task, roi_masks, output_dir, mni_template)
        for sub, task in total_jobs
    )

    # Save QC
    qc_records = [r for r in results if r is not None]
    if qc_records:
        # Merge with existing QC
        qc_path = config.BRAIN_ATTENTION_DIR / "preprocessing_qc.csv"
        if qc_path.exists():
            old_qc = pd.read_csv(str(qc_path))
            new_qc = pd.DataFrame(qc_records)
            qc_df = pd.concat([old_qc, new_qc], ignore_index=True)
            qc_df = qc_df.drop_duplicates(subset=['subject', 'task'], keep='last')
        else:
            qc_df = pd.DataFrame(qc_records)
        qc_df.to_csv(str(qc_path), index=False)

    done = len(list(output_dir.glob("*.npy")))
    logger.info(f"Complete: {done}/118 scans")


if __name__ == '__main__':
    main()
