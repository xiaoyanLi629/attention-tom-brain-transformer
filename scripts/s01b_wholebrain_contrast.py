#!/usr/bin/env python3
"""
=============================================================================
Stage 1b: Whole-Brain Social > Physical Contrast (group-level t-map)
=============================================================================

Reprocesses the ds002345 Shapes task BOLD data to produce a voxel-wise
group-level t-map of social > physical activation across subjects.

Pipeline per subject:
  1. Load BOLD -> resample to MNI -> smooth -> clean -> trim intro TRs
  2. Compute mean activation image per condition (voxel-wise time-average)
  3. Save subject-level contrast map (social_mean - physical_mean)

Group level:
  - One-sample t-test across subjects on contrast maps (voxel-wise)
  - Output NIfTI t-map used by s07 plot_fig1 glass brain.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import nibabel as nib
from nilearn import datasets as ni_datasets
from scipy import stats
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed

from configs import config
from scripts.s01_fmri_preprocessing import preprocess_bold, INTRO_TRS
from nilearn import image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _subject_contrast_to_disk(sub_id, output_dir, mni_template_path):
    """Worker: preprocess one subject and write contrast NIfTI. Returns (sub_id, success)."""
    out_path = output_dir / f"{sub_id}_contrast.nii.gz"
    if out_path.exists():
        return sub_id, True

    # Limit BLAS threads inside each worker to avoid oversubscription
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(var, '2')

    mni = nib.load(str(mni_template_path))

    cond_means = {}
    for task in ['shapessocial', 'shapesphysical']:
        bold_path = config.get_fmri_path(sub_id, task=task)
        if not bold_path.exists():
            return sub_id, False
        img = preprocess_bold(bold_path, mni)
        if img is None or img.shape[3] <= INTRO_TRS:
            return sub_id, False
        img_trimmed = image.index_img(img, slice(INTRO_TRS, None))
        cond_means[task] = np.nanmean(img_trimmed.get_fdata(), axis=3)

    arr = cond_means['shapessocial'] - cond_means['shapesphysical']
    nib.save(nib.Nifti1Image(arr.astype(np.float32), mni.affine), str(out_path))
    return sub_id, True


def run_stage1b(n_jobs=8):
    logger.info("=" * 70)
    logger.info(f"STAGE 1b: Whole-Brain Social > Physical Contrast (n_jobs={n_jobs})")
    logger.info("=" * 70)

    config.ensure_run_directories()
    output_dir = config.BRAIN_ATTENTION_DIR / "wholebrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and cache MNI template once, then pass its path to workers
    mni = ni_datasets.load_mni152_template(resolution=2)
    mni_path = output_dir / "_mni152_2mm.nii.gz"
    if not mni_path.exists():
        nib.save(mni, str(mni_path))
    ref_affine = mni.affine

    subjects = config.SUBJECTS
    pending = [s for s in subjects
               if not (output_dir / f"{s}_contrast.nii.gz").exists()]
    logger.info(f"Subjects: total={len(subjects)}, cached={len(subjects)-len(pending)}, to-process={len(pending)}")

    if pending:
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_subject_contrast_to_disk)(sub_id, output_dir, mni_path)
            for sub_id in pending
        )
        failed = [s for s, ok in results if not ok]
        if failed:
            logger.warning(f"  Failed subjects: {failed}")

    contrast_stack = []
    kept_subjects = []
    for sub_id in tqdm(subjects, desc="Loading contrasts"):
        out_path = output_dir / f"{sub_id}_contrast.nii.gz"
        if not out_path.exists():
            continue
        contrast_stack.append(nib.load(str(out_path)).get_fdata())
        kept_subjects.append(sub_id)

    if not contrast_stack:
        logger.error("No subject contrasts computed — aborting group analysis")
        return

    stack = np.stack(contrast_stack, axis=0)  # (n_subjects, X, Y, Z)
    logger.info(f"Group stack: {stack.shape}")

    # One-sample t-test per voxel (H0: mean diff = 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t_stat, p_val = stats.ttest_1samp(stack, popmean=0.0, axis=0, nan_policy='omit')

    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)
    p_val = np.nan_to_num(p_val, nan=1.0)

    t_img = nib.Nifti1Image(t_stat.astype(np.float32), ref_affine)
    p_img = nib.Nifti1Image(p_val.astype(np.float32), ref_affine)
    nib.save(t_img, str(output_dir / "social_vs_physical_tmap.nii.gz"))
    nib.save(p_img, str(output_dir / "social_vs_physical_pmap.nii.gz"))

    # Also save a brain-masked version (restricts t-map to voxels inside MNI brain)
    mask_img = ni_datasets.load_mni152_brain_mask(resolution=2)
    mask_resampled = image.resample_to_img(mask_img, t_img, interpolation='nearest',
                                           force_resample=True, copy_header=True)
    mask_arr = mask_resampled.get_fdata() > 0.5
    t_masked = t_stat.copy()
    t_masked[~mask_arr] = 0
    nib.save(nib.Nifti1Image(t_masked.astype(np.float32), ref_affine),
             str(output_dir / "social_vs_physical_tmap_masked.nii.gz"))

    logger.info(f"Saved t-map: {output_dir / 'social_vs_physical_tmap.nii.gz'}")
    logger.info(f"  |t| range: [{np.abs(t_stat).min():.2f}, {np.abs(t_stat).max():.2f}]")
    logger.info(f"  N subjects used: {len(kept_subjects)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-jobs', type=int, default=6,
                        help='Parallel workers for subject-level preprocessing '
                             '(default: 6; autodl cgroup cap ~25 cores, each worker uses ~2GB + 2 BLAS threads)')
    args = parser.parse_args()
    run_stage1b(n_jobs=args.n_jobs)
