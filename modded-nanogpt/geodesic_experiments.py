#!/usr/bin/env python3
"""
Geodesic Muon Updates - Experimental Comparison

Tests three geodesic update methods for Muon optimizer on nGPT:
1. Baseline: W_new = normalize(W - lr * U) [projection]
2. Geodesic (theta=lr): W_new = W * cos(lr) + U * sin(lr) [exact manifold]
3. Geodesic (theta=lr*||U||): Scaled version [adaptive]

Hypothesis: Exact geodesic movement converges faster than approximate projection.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Geodesic modes to test
MODES = ['baseline', 'geodesic_lr', 'geodesic_scaled']

MODE_DESCRIPTIONS = {
    'baseline': 'Projection-based (W - lr*U then normalize)',
    'geodesic_lr': 'Exact geodesic (theta = lr)',
    'geodesic_scaled': 'Scaled geodesic (theta = lr * ||U||)'
}

def run_experiment(mode, output_file):
    """Run single experiment with given geodesic mode."""
    print(f"\n{'='*80}")
    print(f"Experiment: {MODE_DESCRIPTIONS[mode]}")
    print(f"{'='*80}")

    cmd = [
        'python3', 'train_architectural.py',
        '--name', f'geodesic_{mode}',
        '--optimizer', 'muon',
        '--lr', '0.003',            # Optimal from Phase 2
        '--momentum', '0.95',
        '--geodesic-mode', mode,
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
    print("GEODESIC MUON UPDATES - RESULTS")
    print(f"{'='*80}")
    print(f"Experiments: {len(results)}")
    print(f"Configuration: Gated nGPT, Muon (lr=0.003), alpha=0.15, batch=32, 400 steps")
    print(f"{'='*80}\n")

    # Sort by validation loss
    results_sorted = sorted(results, key=lambda x: x['val_loss'])
    best = results_sorted[0]

    # Print results table
    print(f"{'Mode':<20} {'Val Loss':<12} {'Final Train':<12} {'Tokens/sec':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        mode = r.get('name', '').replace('geodesic_', '')
        val_loss = r.get('val_loss', 999.0)
        final_train = r.get('final_train_loss', 999.0)
        tokens_per_sec = r.get('tokens_per_sec', 0)
        is_best = '⭐ BEST' if r['name'] == best['name'] else ''

        mode_desc = MODE_DESCRIPTIONS.get(mode, mode)
        print(f"{mode_desc:<20} {val_loss:<12.4f} {final_train:<12.4f} {tokens_per_sec:<12.0f} {is_best}")

    # Analysis
    baseline_result = next((r for r in results if 'baseline' in r.get('name', '')), None)

    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}")

    if baseline_result:
        print(f"\nBaseline (Projection):")
        print(f"  Val Loss: {baseline_result['val_loss']:.4f}")
        print(f"  Method: W_new = normalize(W - lr * U)")
        print(f"  Characteristics: Approximate manifold movement")

        for r in results:
            if 'geodesic' in r.get('name', ''):
                mode = r.get('name', '').replace('geodesic_', '')
                improvement = (baseline_result['val_loss'] - r['val_loss']) / baseline_result['val_loss'] * 100

                print(f"\n{MODE_DESCRIPTIONS[mode]}:")
                print(f"  Val Loss: {r['val_loss']:.4f}")
                if 'geodesic_lr' in r.get('name', ''):
                    print(f"  Method: W_new = W * cos(lr) + U * sin(lr)")
                elif 'geodesic_scaled' in r.get('name', ''):
                    print(f"  Method: W_new = W * cos(lr*||U||) + U * sin(lr*||U||)")
                print(f"  Characteristics: Exact geodesic movement on hypersphere")
                print(f"  vs Baseline: {improvement:+.2f}%")

                if improvement > 0:
                    print(f"  → ✓ Improvement!")
                elif improvement < -0.5:
                    print(f"  → ✗ Degradation")
                else:
                    print(f"  → ≈ Similar performance")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")

    best_mode = best.get('name', '').replace('geodesic_', '')
    print(f"Best performing mode: {MODE_DESCRIPTIONS.get(best_mode, best_mode)}")
    print(f"Best validation loss: {best['val_loss']:.4f}")

    if baseline_result:
        if best['name'] != baseline_result['name']:
            improvement = (baseline_result['val_loss'] - best['val_loss']) / baseline_result['val_loss'] * 100
            print(f"Improvement over baseline: {improvement:+.2f}%")

            if improvement > 1.0:
                print("\n✓ SIGNIFICANT: Geodesic updates provide measurable improvement!")
                print("  Exact manifold movement is superior to projection approximation.")
            elif improvement > 0.1:
                print("\n✓ MODEST: Geodesic updates show slight improvement.")
                print("  The benefit is small but consistent.")
            else:
                print("\n≈ NEUTRAL: Geodesic and projection perform similarly.")
                print("  Both methods are effectively equivalent for this problem.")
        else:
            print("\n= Baseline projection is optimal.")
            print("  Exact geodesic movement doesn't provide additional benefit.")

    print(f"\n{'='*80}")
    print("THEORETICAL IMPLICATIONS")
    print(f"{'='*80}")

    print("""
Geodesic updates implement exact movement along the hypersphere:
- Baseline (projection): Linear step in ambient space + snap to manifold
- Geodesic: Direct rotation along manifold curvature

If geodesic updates improve performance, it suggests:
1. The projection approximation introduces error
2. Exact manifold movement is important for convergence
3. nGPT benefits from respecting geometric structure

If both perform similarly, it suggests:
1. The projection approximation is sufficiently accurate
2. Small LR makes the distinction negligible
3. Lazy projection (every 7 steps) masks the difference
""")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'geodesic_muon_updates',
        'num_experiments': len(results),
        'modes_tested': MODES,
        'best_mode': best.get('name', ''),
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
    results_file = f'geodesic_experiments_{timestamp}.jsonl'
    summary_file = f'geodesic_experiments_{timestamp}_summary.json'

    print("\n" + "="*80)
    print("GEODESIC MUON UPDATES EXPERIMENT")
    print("="*80)
    print(f"Objective: Test exact geodesic movement vs projection")
    print(f"Modes: {len(MODES)} variants")
    print(f"Estimated time: ~{len(MODES) * 50 / 60:.1f} minutes")
    print("="*80)

    # Clear previous results
    if Path(results_file).exists():
        Path(results_file).unlink()

    # Run all experiments
    start_time = time.time()
    successful = 0
    failed = 0

    for i, mode in enumerate(MODES, 1):
        print(f"\n[Experiment {i}/{len(MODES)}]")
        if run_experiment(mode, results_file):
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    # Generate report
    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {successful}/{len(MODES)}")
    print(f"Failed: {failed}/{len(MODES)}")
    print(f"{'='*80}")

    if successful > 0:
        generate_report(results_file, summary_file)
        return 0
    else:
        print("\n✗ All experiments failed!")
        return 1


if __name__ == '__main__':
    exit(main())
