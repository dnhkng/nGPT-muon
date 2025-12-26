#!/usr/bin/env python3
"""
Polar Express Orthogonalization - Experimental Comparison

Tests two orthogonalization methods for Muon optimizer on nGPT:
1. Newton-Schulz (baseline): 5 iterations of Newton-Schulz method
2. Polar Express: Advanced method from https://arxiv.org/pdf/2505.16932

Hypothesis: Polar Express is 10-20% faster and potentially more accurate than Newton-Schulz.
"""

import subprocess
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Orthogonalization methods to test
METHODS = ['newton_schulz', 'polar_express']

METHOD_DESCRIPTIONS = {
    'newton_schulz': 'Newton-Schulz iteration (baseline, 5 iters)',
    'polar_express': 'Polar Express Sign Method (advanced)'
}

def run_experiment(method, output_file):
    """Run single experiment with given orthogonalization method."""
    print(f"\n{'='*80}")
    print(f"Experiment: {METHOD_DESCRIPTIONS[method]}")
    print(f"{'='*80}")

    cmd = [
        sys.executable, 'train_architectural.py',
        '--name', f'orthog_{method}',
        '--optimizer', 'muon',
        '--lr', '0.003',            # Optimal from Phase 2
        '--momentum', '0.95',
        '--orthog-method', method,
        '--geodesic-mode', 'baseline',  # Use baseline projection (optimal from Phase 3)
        '--alpha', '0.15',          # Optimal alpha
        '--gated-residuals',        # Best architecture
        '--batch-size', '32',
        '--lazy-proj-freq', '7',
        '--steps', '400',
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
    """Generate comprehensive comparison report."""
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print(f"\n{'='*80}")
    print("POLAR EXPRESS ORTHOGONALIZATION - RESULTS")
    print(f"{'='*80}")
    print(f"Experiments: {len(results)}")
    print(f"Configuration: Gated nGPT, Muon (lr=0.003), alpha=0.15, batch=32, 400 steps")
    print(f"{'='*80}\n")

    # Sort by validation loss
    results_sorted = sorted(results, key=lambda x: x['val_loss'])
    best = results_sorted[0]

    # Print results table
    print(f"{'Method':<25} {'Val Loss':<12} {'Final Train':<12} {'Tokens/sec':<12} {'Speedup':<12} {'Status'}")
    print("-" * 90)

    baseline_result = next((r for r in results if 'newton_schulz' in r.get('name', '')), None)
    baseline_throughput = baseline_result.get('tokens_per_sec', 0) if baseline_result else 1

    for r in results:
        method = r.get('name', '').replace('orthog_', '')
        val_loss = r.get('val_loss', 999.0)
        final_train = r.get('final_train_loss', 999.0)
        tokens_per_sec = r.get('tokens_per_sec', 0)
        speedup = (tokens_per_sec / baseline_throughput - 1) * 100 if baseline_throughput > 0 else 0
        is_best = '⭐ BEST' if r['name'] == best['name'] else ''

        method_desc = METHOD_DESCRIPTIONS.get(method, method)
        speedup_str = f"+{speedup:.1f}%" if speedup > 0 else f"{speedup:.1f}%"
        print(f"{method_desc:<25} {val_loss:<12.4f} {final_train:<12.4f} {tokens_per_sec:<12.0f} {speedup_str:<12} {is_best}")

    # Analysis
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}")

    if baseline_result:
        print(f"\nBaseline (Newton-Schulz):")
        print(f"  Val Loss: {baseline_result['val_loss']:.4f}")
        print(f"  Throughput: {baseline_result['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Method: 5 iterations of Newton-Schulz")
        print(f"  Complexity: O(d³) per iteration, 5 iterations total")

        for r in results:
            if 'polar_express' in r.get('name', ''):
                improvement_loss = (baseline_result['val_loss'] - r['val_loss']) / baseline_result['val_loss'] * 100
                improvement_speed = (r['tokens_per_sec'] - baseline_result['tokens_per_sec']) / baseline_result['tokens_per_sec'] * 100

                print(f"\n{METHOD_DESCRIPTIONS['polar_express']}:")
                print(f"  Val Loss: {r['val_loss']:.4f}")
                print(f"  Throughput: {r['tokens_per_sec']:.0f} tokens/sec")
                print(f"  Method: 5 iterations with optimized coefficients")
                print(f"  vs Baseline (loss): {improvement_loss:+.2f}%")
                print(f"  vs Baseline (speed): {improvement_speed:+.2f}%")

                if improvement_loss > 1.0:
                    print(f"  → ✓ Significant convergence improvement!")
                elif improvement_loss > 0.1:
                    print(f"  → ✓ Modest convergence improvement")
                elif improvement_loss < -1.0:
                    print(f"  → ✗ Convergence degradation")
                else:
                    print(f"  → ≈ Similar convergence quality")

                if improvement_speed > 10.0:
                    print(f"  → ✓ Significant throughput improvement!")
                elif improvement_speed > 5.0:
                    print(f"  → ✓ Modest throughput improvement")
                elif improvement_speed < -5.0:
                    print(f"  → ✗ Throughput degradation")
                else:
                    print(f"  → ≈ Similar throughput")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")

    best_method = best.get('name', '').replace('orthog_', '')
    print(f"Best performing method: {METHOD_DESCRIPTIONS.get(best_method, best_method)}")
    print(f"Best validation loss: {best['val_loss']:.4f}")

    if baseline_result:
        if best['name'] != baseline_result['name']:
            improvement = (baseline_result['val_loss'] - best['val_loss']) / baseline_result['val_loss'] * 100
            print(f"Improvement over baseline: {improvement:+.2f}%")

            if improvement > 1.0:
                print("\n✓ SIGNIFICANT: Polar Express provides measurable improvement!")
                print("  Advanced orthogonalization is superior to Newton-Schulz.")
            elif improvement > 0.1:
                print("\n✓ MODEST: Polar Express shows slight improvement.")
                print("  The benefit is small but consistent.")
            else:
                print("\n≈ NEUTRAL: Both methods perform similarly.")
                print("  Both orthogonalization methods are effectively equivalent.")
        else:
            print("\n= Newton-Schulz baseline is optimal.")
            print("  Polar Express doesn't provide additional benefit.")

    print(f"\n{'='*80}")
    print("THEORETICAL IMPLICATIONS")
    print(f"{'='*80}")

    print("""
Polar Express is an advanced orthogonalization method that uses:
- Optimized iteration coefficients (precomputed for 5 iterations)
- Matrix operations: X @ X.T and scaled additions
- Claimed benefits: Faster convergence, better accuracy

If Polar Express improves performance, it suggests:
1. The iteration coefficients are better tuned than Newton-Schulz
2. The method's numerical stability advantages matter
3. Worth using for production nGPT training

If both perform similarly, it suggests:
1. Newton-Schulz 5 iterations is already sufficient
2. The differences are masked by other factors (e.g., lazy projection)
3. Either method is acceptable for nGPT
""")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'polar_express_orthogonalization',
        'num_experiments': len(results),
        'methods_tested': METHODS,
        'best_method': best.get('name', ''),
        'best_val_loss': best['val_loss'],
        'baseline_val_loss': baseline_result['val_loss'] if baseline_result else None,
        'all_results': results_sorted
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'polar_express_experiments_{timestamp}.jsonl'
    summary_file = f'polar_express_experiments_{timestamp}_summary.json'

    print("\n" + "="*80)
    print("POLAR EXPRESS ORTHOGONALIZATION EXPERIMENT")
    print("="*80)
    print(f"Objective: Test Polar Express vs Newton-Schulz orthogonalization")
    print(f"Methods: {len(METHODS)} variants")
    print(f"Estimated time: ~{len(METHODS) * 50 / 60:.1f} minutes")
    print("="*80)

    # Clear previous results
    if Path(results_file).exists():
        Path(results_file).unlink()

    # Run all experiments
    start_time = time.time()
    successful = 0
    failed = 0

    for i, method in enumerate(METHODS, 1):
        print(f"\n[Experiment {i}/{len(METHODS)}]")
        if run_experiment(method, results_file):
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    # Generate report
    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {successful}/{len(METHODS)}")
    print(f"Failed: {failed}/{len(METHODS)}")
    print(f"{'='*80}")

    if successful > 0:
        generate_report(results_file, summary_file)
        return 0
    else:
        print("\n✗ All experiments failed!")
        return 1


if __name__ == '__main__':
    exit(main())
