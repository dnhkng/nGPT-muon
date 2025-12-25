#!/usr/bin/env python3
"""
Production Comparison Runner: modded-nanogpt vs Optimized Gated nGPT
Runs both systems with their optimal configurations and compares results.
"""

import subprocess
import json
import time
import os
from pathlib import Path

def clear_results():
    """Clear previous comparison results."""
    results_file = Path('comparison_results.jsonl')
    if results_file.exists():
        results_file.unlink()
    print("Cleared previous results\n")

def run_baseline():
    """Run production modded-nanogpt baseline with FP8."""
    print("="*80)
    print("EXPERIMENT 1: Production Baseline (modded-nanogpt)")
    print("="*80)
    print("Configuration:")
    print("  - Architecture: Standard GPT 11×768 (155M params)")
    print("  - Optimizers: Muon (lr=0.023) + DistAdam (lr=0.008)")
    print("  - FP8 quantization: Enabled")
    print("  - Batch size: 32 (constant)")
    print("  - Training: 4 shards, 1000 steps")
    print("-"*80)

    cmd = ['.venv/bin/torchrun', '--standalone', '--nproc_per_node=1', 'train_baseline_production.py']

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✓ Baseline completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ Baseline failed with return code {result.returncode}")
        return False

    return True

def run_gated_ngpt():
    """Run optimized gated nGPT (without FP8)."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Optimized Gated nGPT")
    print("="*80)
    print("Configuration:")
    print("  - Architecture: Gated Residual nGPT 11×768 (155M params)")
    print("  - Optimizer: Adam (lr=0.0005)")
    print("  - Alpha: 0.15 (optimal from hyperparameter sweep)")
    print("  - Projection frequency: 7")
    print("  - FP8 quantization: Disabled")
    print("  - Batch size: 32")
    print("  - Training: 4 shards, 1000 steps")
    print("-"*80)

    cmd = [
        '.venv/bin/python3', 'train_architectural.py',
        '--name', 'gated_optimal_comparison',
        '--gated-residuals',
        '--alpha', '0.15',
        '--lr', '0.0005',
        '--batch-size', '32',
        '--lazy-proj-freq', '7',
        '--steps', '1000',
        '--output', 'comparison_results.jsonl'
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✓ Gated nGPT completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ Gated nGPT failed with return code {result.returncode}")
        return False

    return True

def compare_results():
    """Load and compare results from both experiments."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    results_file = Path('comparison_results.jsonl')
    if not results_file.exists():
        print("Error: No results file found!")
        return

    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    if len(results) < 2:
        print(f"Error: Expected 2 results, got {len(results)}")
        return

    baseline = results[0]
    gated = results[1]

    print("\nBASELINE (modded-nanogpt):")
    print(f"  Architecture: {baseline['architecture']}")
    print(f"  Optimizer: {baseline['optimizer']}")
    print(f"  FP8: {baseline.get('use_fp8', False)}")
    print(f"  Validation Loss: {baseline['val_loss']:.4f}")
    if 'training_time_ms' in baseline:
        print(f"  Training Time: {baseline['training_time_ms']/1000:.1f}s")

    print("\nGATED nGPT (optimized):")
    print(f"  Architecture: {gated.get('architecture', 'gated_ngpt_11x768')}")
    print(f"  Optimizer: {gated.get('optimizer', 'Adam')}")
    print(f"  FP8: {gated.get('use_fp8', False)}")
    print(f"  Validation Loss: {gated['val_loss']:.4f}")
    if 'time_seconds' in gated:
        print(f"  Training Time: {gated['time_seconds']:.1f}s")
    if 'tokens_per_sec' in gated:
        print(f"  Throughput: {gated['tokens_per_sec']:,.0f} tokens/sec")

    # Calculate improvement
    improvement = (baseline['val_loss'] - gated['val_loss']) / baseline['val_loss'] * 100

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline validation loss:     {baseline['val_loss']:.4f}")
    print(f"Gated nGPT validation loss:   {gated['val_loss']:.4f}")
    print(f"Improvement:                  {improvement:+.2f}%")

    if improvement > 0:
        print(f"\n✓ Gated nGPT WINS by {improvement:.2f}%")
        print("  Despite baseline having FP8 advantage, gated architecture wins!")
    elif improvement < 0:
        print(f"\n✗ Baseline wins by {-improvement:.2f}%")
        print("  Note: Baseline has FP8 advantage")
    else:
        print("\n= Results are tied")

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Save summary
    summary = {
        'baseline': baseline,
        'gated_ngpt': gated,
        'improvement_pct': improvement,
        'winner': 'gated_ngpt' if improvement > 0 else 'baseline' if improvement < 0 else 'tie'
    }

    with open('comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: comparison_summary.json\n")

def main():
    print("\n" + "="*80)
    print("PRODUCTION COMPARISON: modded-nanogpt vs Optimized Gated nGPT")
    print("="*80)
    print("Dataset: FineWeb10B (4 shards)")
    print("Steps: 1000")
    print("Model: 11 layers × 768 dim (155M parameters)")
    print("="*80 + "\n")

    # Clear previous results
    clear_results()

    # Run baseline
    if not run_baseline():
        print("Baseline failed, aborting comparison")
        return 1

    # Run gated nGPT
    if not run_gated_ngpt():
        print("Gated nGPT failed, aborting comparison")
        return 1

    # Compare results
    compare_results()

    return 0

if __name__ == '__main__':
    exit(main())
