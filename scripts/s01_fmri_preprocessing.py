#!/usr/bin/env python3
"""
=============================================================================
Stage 1: fMRI Preprocessing and ROI Time Series Extraction
=============================================================================

Process ds002345 Shapes task fMRI data (BIDS format):
  - Load volumetric NIfTI BOLD data
  - Trim introductory music period (first 33 TRs)
  - Confound regression (high-pass filter, detrend)
  - Spatial smoothing (6mm FWHM)
  - Z-score normalization
  - Extract ROI time series (6 social brain ROIs)
  - Build word-to-TR alignment from whisper transcripts

Outputs:
  - roi_timeseries/{sub}_{task}.npy   — (n_TRs, n_ROIs)
  - word_to_tr_mapping.json           — word index → TR index
  - preprocessing_qc.csv              — quality control per subject
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, signal, masking
from scipy import stats
import json
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# =============================================================================
# CONSTANTS
# =============================================================================

SOCIAL_BRAIN_ROIS = {
    'rTPJ': {'center': (54, -48, 24), 'radius': 12},
    'lTPJ': {'center': (-54, -48, 24), 'radius': 12},
    'mPFC': {'center': (0, 54, 18), 'radius': 12},
    'PC':   {'center': (0, -54, 36), 'radius': 12},
    'rSTS': {'center': (54, -42, 6), 'radius': 10},
    'lSTS': {'center': (-54, -42, 6), 'radius': 10},
}

ROI_NAMES = list(SOCIAL_BRAIN_ROIS.keys())

TR = config.SHAPES_TASK['tr']                    # 1.5s
INTRO_TRS = config.SHAPES_TASK['intro_trs']      # 33 TRs to trim
STORY_ONSET = config.SHAPES_TASK['story_onset']  # 49.5s
SMOOTH_FWHM = 6  # mm
HIGH_PASS_FREQ = 0.01  # Hz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ROI EXTRACTION
# =============================================================================

def create_spherical_roi(shape, affine, center, radius):
    """Create spherical ROI mask in MNI space."""
    i, j, k = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    ijk = np.column_stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    mni = (affine @ ijk.T).T[:, :3]
    distances = np.sqrt(np.sum((mni - np.array(center))**2, axis=1))
    mask = (distances <= radius).reshape(shape)
    return mask


def extract_roi_timeseries(img_4d, roi_masks):
    """Extract mean time series for each ROI from a 4D image.

    Args:
        img_4d: nibabel 4D NIfTI image (already preprocessed)
        roi_masks: dict {roi_name: 3D boolean mask array}

    Returns:
        np.ndarray of shape (n_trs, n_rois)
    """
    data = img_4d.get_fdata()
    n_trs = data.shape[3]
    n_rois = len(roi_masks)
    timeseries = np.zeros((n_trs, n_rois))

    for i, (roi_name, mask) in enumerate(roi_masks.items()):
        voxels = data[mask]  # (n_voxels, n_trs)
        if voxels.size == 0:
            logger.warning(f"ROI {roi_name}: no voxels in mask, filling with NaN")
            timeseries[:, i] = np.nan
        else:
            timeseries[:, i] = np.nanmean(voxels, axis=0)

    return timeseries


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_bold(bold_path, mni_template, smooth_fwhm=SMOOTH_FWHM):
    """Load and preprocess a BOLD NIfTI file.

    Steps:
      1. Load 4D NIfTI
      2. Resample to MNI space (critical for ROI alignment)
      3. Spatial smoothing (6mm FWHM)
      4. Confound regression: high-pass filter + detrend
      5. Z-score each voxel

    Args:
        bold_path: Path to _bold.nii.gz file
        mni_template: MNI152 template image (target space)
        smooth_fwhm: smoothing kernel FWHM in mm

    Returns:
        nibabel 4D image (preprocessed, in MNI space), or None on failure
    """
    bold_path = Path(bold_path)
    if not bold_path.exists():
        logger.error(f"BOLD file not found: {bold_path}")
        return None

    logger.info(f"Loading {bold_path.name}...")
    img = nib.load(str(bold_path))

    # Resample to MNI space
    logger.info(f"  Resampling to MNI space...")
    img_mni = image.resample_to_img(img, mni_template, interpolation='continuous')

    # Smooth
    logger.info(f"  Smoothing ({smooth_fwhm}mm FWHM)...")
    img_smooth = image.smooth_img(img_mni, fwhm=smooth_fwhm)

    # Clean: high-pass filter + detrend + z-score
    logger.info(f"  Cleaning (high-pass={HIGH_PASS_FREQ}Hz, detrend, standardize)...")
    img_clean = image.clean_img(
        img_smooth,
        detrend=True,
        standardize='zscore_sample',
        high_pass=HIGH_PASS_FREQ,
        t_r=TR,
    )

    return img_clean


def process_subject(sub_id, task_name, roi_masks, output_dir, mni_template):
    """Process a single subject x task combination.

    Args:
        sub_id: e.g. 'sub-206'
        task_name: 'shapessocial' or 'shapesphysical'
        roi_masks: dict of precomputed ROI masks
        output_dir: directory to save outputs
        mni_template: MNI152 template for spatial normalization

    Returns:
        dict with QC metrics, or None on failure
    """
    bold_path = (config.DS002345_FMRI / sub_id / "func" /
                 f"{sub_id}_task-{task_name}_bold.nii.gz")

    if not bold_path.exists():
        logger.warning(f"Missing: {bold_path}")
        return None

    # Preprocess (with MNI resampling)
    img_clean = preprocess_bold(bold_path, mni_template)
    if img_clean is None:
        return None

    # Get total TRs
    total_trs = img_clean.shape[3]

    # Trim intro music (first INTRO_TRS)
    if total_trs <= INTRO_TRS:
        logger.error(f"{sub_id} {task_name}: only {total_trs} TRs, cannot trim {INTRO_TRS}")
        return None

    img_trimmed = image.index_img(img_clean, slice(INTRO_TRS, None))
    story_trs = img_trimmed.shape[3]
    logger.info(f"  Trimmed: {total_trs} → {story_trs} TRs (removed {INTRO_TRS} intro TRs)")

    # Extract ROI time series
    ts = extract_roi_timeseries(img_trimmed, roi_masks)

    # Save
    out_path = output_dir / f"{sub_id}_{task_name}.npy"
    np.save(str(out_path), ts)

    # QC metrics
    qc = {
        'subject': sub_id,
        'task': task_name,
        'total_trs_raw': total_trs,
        'story_trs': story_trs,
        'mean_fd': np.nan,  # Would need confounds file
    }
    for i, name in enumerate(ROI_NAMES):
        roi_ts = ts[:, i]
        qc[f'{name}_mean'] = float(np.nanmean(roi_ts))
        qc[f'{name}_std'] = float(np.nanstd(roi_ts))
        qc[f'{name}_tsnr'] = float(np.nanmean(roi_ts) / np.nanstd(roi_ts)) if np.nanstd(roi_ts) > 0 else 0

    return qc


# =============================================================================
# WORD-TO-TR MAPPING
# =============================================================================

def build_word_to_tr_mapping(transcript_dir):
    """Build mapping from word index to TR index for each task.

    Uses whisper word-level timestamps. TR indexing starts from 0
    after intro trimming (i.e., TR 0 = story onset).

    Returns:
        dict: {task_name: [{word, start, end, tr_idx}, ...]}
    """
    mapping = {}

    for task in ['shapessocial', 'shapesphysical']:
        words_file = transcript_dir / f"{task}_words.txt"
        if not words_file.exists():
            logger.warning(f"Word timestamps not found: {words_file}")
            continue

        words = []
        with open(words_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    word = parts[2]

                    # Skip intro music words (before story onset)
                    if start < STORY_ONSET:
                        continue

                    # Compute TR index relative to story onset
                    word_time_relative = start - STORY_ONSET
                    tr_idx = int(word_time_relative / TR)

                    words.append({
                        'word': word,
                        'start': start,
                        'end': end,
                        'start_relative': word_time_relative,
                        'tr_idx': tr_idx,
                    })

        mapping[task] = words
        logger.info(f"  {task}: {len(words)} words mapped to TRs "
                    f"(TR range: 0-{words[-1]['tr_idx'] if words else '?'})")

    return mapping


# =============================================================================
# MAIN
# =============================================================================

def run_stage1():
    """Run Stage 1: fMRI preprocessing and ROI extraction."""
    logger.info("=" * 70)
    logger.info("STAGE 1: fMRI Preprocessing and ROI Time Series Extraction")
    logger.info("=" * 70)

    # Setup output directory
    config.ensure_run_directories()
    output_dir = config.BRAIN_ATTENTION_DIR / "roi_timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get subject list
    subjects = config.get_subject_list()
    if not subjects:
        logger.error("No subjects found! Check DS002345_FMRI path.")
        return
    logger.info(f"Found {len(subjects)} subjects")

    # Load MNI template (all data will be resampled to this space)
    from nilearn import datasets as ni_datasets
    mni_template = ni_datasets.load_mni152_template(resolution=2)
    logger.info(f"MNI template: shape={mni_template.shape}, voxel=2mm")

    # Build ROI masks in MNI space
    ref_shape = mni_template.shape[:3]
    ref_affine = mni_template.affine

    logger.info(f"Reference image shape: {ref_shape}, voxel size: {np.abs(np.diag(ref_affine)[:3])}")

    roi_masks = {}
    for roi_name, roi_info in SOCIAL_BRAIN_ROIS.items():
        mask = create_spherical_roi(ref_shape, ref_affine,
                                    roi_info['center'], roi_info['radius'])
        n_voxels = mask.sum()
        roi_masks[roi_name] = mask
        logger.info(f"  ROI {roi_name}: {n_voxels} voxels")

    if any(m.sum() == 0 for m in roi_masks.values()):
        logger.error("Some ROIs have 0 voxels! Check image space (must be MNI).")
        return

    # Process all subjects
    qc_records = []
    tasks = config.SHAPES_TASK['task_names']

    for sub in tqdm(subjects, desc="Subjects"):
        for task in tasks:
            qc = process_subject(sub, task, roi_masks, output_dir, mni_template)
            if qc is not None:
                qc_records.append(qc)

    # Save QC
    qc_df = pd.DataFrame(qc_records)
    qc_path = config.BRAIN_ATTENTION_DIR / "preprocessing_qc.csv"
    qc_df.to_csv(str(qc_path), index=False)
    logger.info(f"QC saved: {qc_path} ({len(qc_df)} scans)")

    # Build word-to-TR mapping
    logger.info("Building word-to-TR mapping...")
    word_mapping = build_word_to_tr_mapping(config.DS002345_TRANSCRIPTS)
    mapping_path = config.BRAIN_ATTENTION_DIR / "word_to_tr_mapping.json"
    with open(str(mapping_path), 'w') as f:
        json.dump(word_mapping, f, indent=2)
    logger.info(f"Word-to-TR mapping saved: {mapping_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("Stage 1 Complete")
    logger.info(f"  Subjects processed: {len(qc_df['subject'].unique())}")
    logger.info(f"  Scans processed: {len(qc_df)}")
    for task in tasks:
        task_qc = qc_df[qc_df['task'] == task]
        if len(task_qc) > 0:
            mean_trs = task_qc['story_trs'].mean()
            logger.info(f"  {task}: {len(task_qc)} scans, mean story TRs = {mean_trs:.0f}")
    logger.info(f"  ROI time series saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_stage1()
