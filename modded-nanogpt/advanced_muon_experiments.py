#!/usr/bin/env python3
"""
Advanced Muon Features - Experimental Comparison

Tests advanced features ported from modded-nanogpt:
1. Variance Reduction (NorMuon)
2. Cautious Weight Decay

All experiments use Polar Express orthogonalization as the base, 
given its superior convergence quality.
"""

import subprocess
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Experiments to test
EXPERIMENTS = [
    {'name': 'base_polar', 'orthog': 'polar_express', 'vr': False, 'cwd': False},
    {'name': 'vr_only', 'orthog': 'polar_express', 'vr': True, 'cwd': False},
    {'name': 'cwd_only', 'orthog': 'polar_express', 'vr': False, 'cwd': True, 'wd': 0.01},
    {'name': 'vr_cwd_all', 'orthog': 'polar_express', 'vr': True, 'cwd': True, 'wd': 0.01},
]

def run_experiment(exp, output_file):
    """Run single experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"Experiment: {exp['name']}")
    print(f"{'='*80}")

    cmd = [
        sys.executable, 'train_architectural.py',
        '--name', f"adv_{exp['name']}",
        '--optimizer', 'muon',
        '--lr', '0.003',
        '--momentum', '0.95',
        '--orthog-method', exp['orthog'],
        '--alpha', '0.15',
        '--gated-residuals',
        '--batch-size', '32',
        '--steps', '400',
        '--output', output_file
    ]

    if exp['vr']:
        cmd.append('--variance-reduction')
    
    if exp['cwd']:
        cmd.append('--cautious-wd')
        # Need to provide weight decay for cautious WD to have an effect
        # although nGPT weights are normalized, cautious WD might still act as a regularizer
        # or help the 2D weights stay on the sphere better.
        # However, nGPT usually uses WD=0. Let's test if WD>0 helps with CWD.

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✓ Experiment completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ Experiment failed with return code {result.returncode}")
        return False

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'advanced_muon_results_{timestamp}.jsonl'
    summary_file = f'advanced_muon_results_{timestamp}_summary.json'

    print("\n" + "="*80)
    print("ADVANCED MUON FEATURES EXPERIMENT")
    print("="*80)
    print(f"Objective: Test Variance Reduction and Cautious WD on nGPT")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("="*80)

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[Experiment {i}/{len(EXPERIMENTS)}]")
        run_experiment(exp, results_file)

    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to {results_file}")
    print("="*80)

if __name__ == '__main__':
    main()
