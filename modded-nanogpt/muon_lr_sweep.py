#!/usr/bin/env python3
"""
Muon Optimizer Learning Rate Sweep

Systematically tests Muon optimizer with different learning rates to find
the optimal LR for nGPT's normalized geometry.

Hypothesis: Optimal Muon LR for nGPT differs from standard GPT
due to unit norm constraints.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Learning rates to test
# 0.023 is modded-nanogpt's optimal LR for standard GPT
LR_VALUES = [0.001, 0.003, 0.01, 0.023, 0.05, 0.1]

def run_experiment(lr, output_file):
    """Run single Muon experiment with given learning rate."""
    print(f"\n{'='*80}")
    print(f"Experiment: Muon LR = {lr}")
    print(f"{'='*80}")

    cmd = [
        'python3', 'train_architectural.py',
        '--name', f'muon_lr_{lr}',
        '--optimizer', 'muon',
        '--lr', str(lr),
        '--momentum', '0.95',
        '--alpha', '0.15',           # Optimal from previous experiments
        '--gated-residuals',          # Best architecture
        '--batch-size', '32',
        '--lazy-proj-freq', '7',      # Optimal projection frequency
        '--steps', '400',             # Full experiment
        '--output', output_file
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✓ Experiment completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ Experiment failed with return code {result.returncode}")
        return False


def generate_report(results_file, summary_file):
    """Generate comprehensive report from results."""
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print(f"\n{'='*80}")
    print("MUON LEARNING RATE SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"Experiments: {len(results)}")
    print(f"Configuration: Gated nGPT, alpha=0.15, batch=32, 400 steps")
    print(f"{'='*80}\n")

    # Sort by validation loss
    results_sorted = sorted(results, key=lambda x: x['val_loss'])
    best = results_sorted[0]

    # Print results table
    print(f"{'LR':<10} {'Val Loss':<12} {'Final Train':<12} {'Tokens/sec':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        lr = r.get('lr', 'N/A')
        val_loss = r.get('val_loss', 999.0)
        final_train = r.get('final_train_loss', 999.0)
        tokens_per_sec = r.get('tokens_per_sec', 0)
        is_best = '⭐ BEST' if r['lr'] == best['lr'] else ''

        print(f"{lr:<10.4f} {val_loss:<12.4f} {final_train:<12.4f} {tokens_per_sec:<12.0f} {is_best}")

    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGURATION")
    print(f"{'='*80}")
    print(f"Best Learning Rate: {best['lr']}")
    print(f"Validation Loss: {best['val_loss']:.4f}")
    print(f"Final Train Loss: {best.get('final_train_loss', 'N/A'):.4f}")
    if 'tokens_per_sec' in best:
        print(f"Throughput: {best['tokens_per_sec']:,.0f} tokens/sec")

    # Compare with Adam baseline (if we have it)
    print(f"\n{'='*80}")
    print("COMPARISON WITH ADAM BASELINE")
    print(f"{'='*80}")
    print("Adam baseline (from previous experiments):")
    print("  - LR: 0.0005")
    print("  - Val Loss: 6.775")
    print(f"\nBest Muon result:")
    print(f"  - LR: {best['lr']}")
    print(f"  - Val Loss: {best['val_loss']:.4f}")

    improvement = (6.775 - best['val_loss']) / 6.775 * 100
    if improvement > 0:
        print(f"\n✓ Muon WINS by {improvement:.2f}%")
        print("  Muon optimizer outperforms Adam for nGPT!")
    elif improvement < 0:
        print(f"\n⚠ Muon underperforms by {-improvement:.2f}%")
        print("  Consider:")
        print("  - Testing more LR values")
        print("  - Implementing Polar Express orthogonalization")
        print("  - Adding NorMuon variance reduction")
    else:
        print("\n= Results are tied")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'muon_lr_sweep',
        'num_experiments': len(results),
        'lr_values_tested': LR_VALUES,
        'best_lr': best['lr'],
        'best_val_loss': best['val_loss'],
        'all_results': results_sorted,
        'adam_baseline_val_loss': 6.775,
        'improvement_vs_adam_pct': improvement
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'muon_lr_sweep_{timestamp}.jsonl'
    summary_file = f'muon_lr_sweep_{timestamp}_summary.json'

    print("\n" + "="*80)
    print("MUON LEARNING RATE SWEEP")
    print("="*80)
    print(f"Objective: Find optimal Muon LR for nGPT")
    print(f"LR values: {LR_VALUES}")
    print(f"Total experiments: {len(LR_VALUES)}")
    print(f"Estimated time: ~{len(LR_VALUES) * 40 / 60:.1f} minutes")
    print("="*80)

    # Clear previous results
    if Path(results_file).exists():
        Path(results_file).unlink()

    # Run all experiments
    start_time = time.time()
    successful = 0
    failed = 0

    for i, lr in enumerate(LR_VALUES, 1):
        print(f"\n[Experiment {i}/{len(LR_VALUES)}]")
        if run_experiment(lr, results_file):
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    # Generate report
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {successful}/{len(LR_VALUES)}")
    print(f"Failed: {failed}/{len(LR_VALUES)}")
    print(f"{'='*80}")

    if successful > 0:
        generate_report(results_file, summary_file)
        return 0
    else:
        print("\n✗ All experiments failed!")
        return 1


if __name__ == '__main__':
    exit(main())
