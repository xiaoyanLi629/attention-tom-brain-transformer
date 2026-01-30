#!/usr/bin/env python3
"""
=============================================================================
Project 3 Analysis Pipeline: Adaptive Heuristics in Social Cognition
CogSci 2026 Submission
=============================================================================

"When Fast is Better: Neural Signatures of Adaptive Heuristics in Social Cognition"

Complete pipeline for analyzing brain-Transformer alignment in social cognition,
with deep mechanistic analysis of attention patterns.

Stages:
    1. Behavioral Analysis - Human ToM task performance
    2. Brain Activation - fMRI activation extraction  
    3. Transformer Extraction - Model attention & hidden states
    4. Probing Analysis - Layer-wise information content
    5. Causal Analysis - Activation patching & ablation
    6. RSA Analysis - Brain-model alignment
    7. Encoding Models - Predicting brain from model
    8. Cross-Project Integration - Unified framework
    9. Basic Visualization - Publication figures
    10. Advanced Analysis - Geometry, clustering, CKA
    11. Advanced Visualization - Scientific figures
    12. Temporal Analysis - Dynamic connectivity, phase synchrony
    13. Temporal Visualization - Temporal dynamics figures

Usage:
    python run_pipeline.py                    # Run all stages
    python run_pipeline.py --stage 1 2 3      # Run specific stages
    python run_pipeline.py --from 5           # Run from stage 5 onwards
    python run_pipeline.py --to 9             # Run up to stage 9
    python run_pipeline.py --from 5 --to 9    # Run stages 5-9
    python run_pipeline.py --visualize        # Only run visualization stages (9, 11, 13)
    python run_pipeline.py --list             # List all stages
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from configs import config

# =============================================================================
# PIPELINE STAGES
# =============================================================================

STAGES = {
    1: {
        'name': 'Behavioral Analysis',
        'script': 's01_behavioral_analysis.py',
        'description': 'Extract RT and accuracy, compute heuristic index',
    },
    2: {
        'name': 'Brain Activation Analysis',
        'script': 's02_brain_activation.py',
        'description': 'Analyze ToM ROI activation and brain sparsity',
    },
    3: {
        'name': 'Transformer Extraction',
        'script': 's03_transformer_extraction.py',
        'description': 'Extract attention weights from transformer models',
    },
    4: {
        'name': 'Probing Analysis',
        'script': 's04_probing_analysis.py',
        'description': 'Train probing classifiers on hidden states',
    },
    5: {
        'name': 'Causal Analysis',
        'script': 's05_causal_analysis.py',
        'description': 'Identify causal circuits via ablation',
    },
    6: {
        'name': 'RSA Alignment',
        'script': 's06_rsa_alignment.py',
        'description': 'Brain-model representational similarity analysis',
    },
    7: {
        'name': 'Encoding Models',
        'script': 's07_encoding_models.py',
        'description': 'Predict brain activity from model layers',
    },
    8: {
        'name': 'Cross-Project Integration',
        'script': 's08_cross_project.py',
        'description': 'Integrate findings across P1, P2, and P3',
    },
    9: {
        'name': 'Basic Visualization',
        'script': 's09_visualization.py',
        'description': 'Generate basic publication figures (PNG + SVG)',
    },
    10: {
        'name': 'Advanced Analysis',
        'script': 's10_advanced_analysis.py',
        'description': 'Geometry, attention clustering, CKA, circuit discovery',
    },
    11: {
        'name': 'Advanced Visualization',
        'script': 's11_advanced_visualization.py',
        'description': 'Generate advanced scientific figures (PNG + SVG)',
    },
    12: {
        'name': 'Temporal Dynamics Analysis',
        'script': 's12_temporal_analysis.py',
        'description': 'Dynamic connectivity, phase synchrony, attention evolution',
    },
    13: {
        'name': 'Temporal Visualization',
        'script': 's13_temporal_visualization.py',
        'description': 'Generate temporal dynamics figures (PNG + SVG)',
    },
    14: {
        'name': 'Glass Brain Visualization',
        'script': 's14_glass_brain_visualization.py',
        'description': 'Generate glass brain figures for ToM activation (PNG + SVG)',
    },
}

# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_stage(stage_num, timestamp=None):
    """Run a single pipeline stage"""
    stage = STAGES[stage_num]
    script_path = PROJECT_DIR / "scripts" / stage['script']
    
    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: {stage['name']}")
    print(f"Description: {stage['description']}")
    print(f"{'='*60}\n")
    
    # Prepare environment with timestamp
    env = os.environ.copy()
    if timestamp:
        env['CURRENT_RUN_TIMESTAMP'] = timestamp
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(PROJECT_DIR),
            env=env
        )
        print(f"\n  ✅ Stage {stage_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Stage {stage_num} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n  ❌ Stage {stage_num} failed: {str(e)}")
        return False


def run_pipeline(stages=None, start_stage=1, end_stage=14):
    """Run the full pipeline or specified stages"""
    
    # Setup logging
    logger = config.setup_logging('pipeline')
    
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    # Initialize timestamped directories
    config.initialize_run_directories(timestamp)
    
    print("\n" + "="*70)
    print("  PROJECT 3: ADAPTIVE HEURISTICS IN SOCIAL COGNITION")
    print("  CogSci 2026 Analysis Pipeline")
    print("="*70)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {config.CURRENT_RUN_DIR}")
    print(f"Data directory: {config.DATA_ROOT}")
    print(f"Number of subjects: {len(config.SUBJECTS)}")
    
    # Determine which stages to run
    if stages is None:
        stages = [s for s in STAGES.keys() if start_stage <= s <= end_stage]
    
    print(f"\nStages to run: {stages}")
    
    # Run each stage
    results = {}
    stage_times = {}
    
    for stage_num in stages:
        if stage_num not in STAGES:
            print(f"⚠️  Unknown stage: {stage_num}")
            continue
        
        stage_start = time.time()
        success = run_stage(stage_num, timestamp=timestamp)
        stage_times[stage_num] = time.time() - stage_start
        results[stage_num] = success
        
        if not success:
            logger.error(f"Stage {stage_num} failed")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("  PIPELINE SUMMARY")
    print("="*70)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nResults:")
    
    for stage_num, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        elapsed = stage_times.get(stage_num, 0)
        print(f"  Stage {stage_num:2d} ({STAGES[stage_num]['name']:25s}): {status} ({elapsed:.1f}s)")
    
    all_passed = all(results.values()) if results else False
    if all_passed:
        print(f"\n🎉 All stages completed successfully!")
        logger.info("Pipeline completed successfully")
    else:
        failed = [s for s, r in results.items() if not r]
        print(f"\n⚠️  Some stages failed: {failed}")
        logger.warning(f"Pipeline completed with failures: {failed}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Run Project 3 Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                    # Run all stages
    python run_pipeline.py --stage 1 2 3      # Run specific stages
    python run_pipeline.py --from 5           # Run from stage 5 onwards
    python run_pipeline.py --from 5 --to 9    # Run stages 5-9
    python run_pipeline.py --visualize        # Only run visualization stages
        """
    )
    parser.add_argument(
        '--stage', '-s',
        type=int,
        nargs='+',
        help='Specific stages to run (e.g., --stage 1 2 3)'
    )
    parser.add_argument(
        '--from', dest='start',
        type=int,
        default=1,
        help='Start from this stage (default: 1)'
    )
    parser.add_argument(
        '--to', dest='end',
        type=int,
        default=14,
        help='End at this stage (default: 14)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Only run visualization stages (9, 11, 13)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available stages'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "="*60)
        print("  Available Pipeline Stages")
        print("="*60)
        for num, stage in STAGES.items():
            print(f"\n  Stage {num:2d}: {stage['name']}")
            print(f"           {stage['description']}")
        print()
        return
    
    if args.visualize:
        # Run only visualization stages
        success = run_pipeline(stages=[9, 11, 13])
    elif args.stage:
        # Run specific stages
        success = run_pipeline(stages=args.stage)
    else:
        # Run range of stages
        success = run_pipeline(start_stage=args.start, end_stage=args.end)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

