#!/usr/bin/env python3
"""
Hyperparameter Optimization for Gated Residual nGPT
Optimizes parameters specific to the breakthrough gated architecture
"""

import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

class GatedHyperparamOptimizer:
    """Systematic hyperparameter optimization for gated residuals."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f'gated_hyperparam_{self.timestamp}.jsonl'
        self.results = []

    def define_experiments(self):
        """Define all hyperparameter experiments for gated architecture."""

        # Baseline gated config (from architectural experiments)
        base_config = {
            'name': 'gated_baseline',
            'args': [
                '--name', 'gated_baseline',
                '--gated-residuals',
                '--alpha', '0.28',
                '--logit-scale', '15.0',
                '--batch-size', '32',
                '--lr', '0.001',
                '--steps', '400',  # Slightly longer for better signal
                '--lazy-proj-freq', '7',
                '--output', self.output_file,
            ]
        }

        experiments = [base_config]

        # ========== GATE INITIALIZATION SWEEP ==========
        # Hypothesis: Gate initialization affects learning dynamics
        # σ(x) = 0.5 at x=0, but what about x = -2, -1, +1, +2?
        for gate_init in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            exp = {
                'name': f'gate_init_{gate_init}',
                'args': [
                    '--name', f'gate_init_{gate_init}',
                    '--gated-residuals',
                    '--alpha', '0.28',
                    '--gate-init', str(gate_init),
                    '--batch-size', '32',
                    '--steps', '400',
                    '--output', self.output_file,
                ],
                'category': 'gate_init',
            }
            experiments.append(exp)

        # ========== LEARNING RATE SWEEP ==========
        # Hypothesis: Gated architecture may need different LR
        for lr in [0.0005, 0.001, 0.0015, 0.002]:
            exp = {
                'name': f'lr_{lr}',
                'args': [
                    '--name', f'lr_{lr}',
                    '--gated-residuals',
                    '--lr', str(lr),
                    '--batch-size', '32',
                    '--steps', '400',
                    '--output', self.output_file,
                ],
                'category': 'learning_rate',
            }
            experiments.append(exp)

        # ========== BASE ALPHA SWEEP ==========
        # Hypothesis: Gated arch may need different base alpha
        for alpha in [0.15, 0.20, 0.28, 0.35, 0.40]:
            exp = {
                'name': f'alpha_{alpha}',
                'args': [
                    '--name', f'alpha_{alpha}',
                    '--gated-residuals',
                    '--alpha', str(alpha),
                    '--batch-size', '32',
                    '--steps', '400',
                    '--output', self.output_file,
                ],
                'category': 'alpha',
            }
            experiments.append(exp)

        # ========== BATCH SIZE SWEEP ==========
        # Hypothesis: Gates may interact with batch size
        for bs in [16, 24, 32, 48]:
            exp = {
                'name': f'batch_{bs}',
                'args': [
                    '--name', f'batch_{bs}',
                    '--gated-residuals',
                    '--batch-size', str(bs),
                    '--steps', '400',
                    '--output', self.output_file,
                ],
                'category': 'batch_size',
            }
            experiments.append(exp)

        # ========== LAZY PROJECTION FREQUENCY ==========
        # Hypothesis: Gates may change optimal projection frequency
        for freq in [5, 7, 10, 15]:
            exp = {
                'name': f'proj_freq_{freq}',
                'args': [
                    '--name', f'proj_freq_{freq}',
                    '--gated-residuals',
                    '--lazy-proj-freq', str(freq),
                    '--batch-size', '32',
                    '--steps', '400',
                    '--output', self.output_file,
                ],
                'category': 'proj_freq',
            }
            experiments.append(exp)

        # ========== COMBINED OPTIMIZATIONS ==========
        # Test combinations of best settings from each category
        # Will be populated after initial sweeps

        return experiments

    def run_experiment(self, exp, i, total):
        """Run a single experiment."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{total}: {exp['name']}")
        if 'category' in exp:
            print(f"Category: {exp['category']}")
        print(f"{'='*80}\n")

        cmd = ['python3', 'train_architectural.py'] + exp['args']

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0

        if result.returncode == 0:
            # Parse result
            with open(self.output_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_result = json.loads(lines[-1])
                    last_result['category'] = exp.get('category', 'baseline')
                    self.results.append(last_result)

                    print(f"\n✓ Completed in {elapsed:.1f}s")
                    print(f"  Val loss: {last_result['val_loss']:.4f}")
                    print(f"  Train loss: {last_result['final_train_loss']:.4f}")

                    return last_result
        else:
            print(f"\n✗ Failed with return code {result.returncode}")
            return None

        time.sleep(1)

    def run_all(self):
        """Run all experiments."""
        experiments = self.define_experiments()

        print(f"{'='*80}")
        print(f"GATED RESIDUAL HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Expected time: ~{len(experiments) * 35 / 60:.1f} minutes")
        print(f"{'='*80}\n")

        for i, exp in enumerate(experiments, 1):
            self.run_experiment(exp, i, len(experiments))

        # Analyze by category
        self.analyze_results()

        # Save summary
        self.save_summary()

    def analyze_results(self):
        """Analyze results by category."""
        print(f"\n{'='*80}")
        print("CATEGORY ANALYSIS")
        print(f"{'='*80}\n")

        categories = {}
        for result in self.results:
            cat = result.get('category', 'baseline')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        for cat, results in categories.items():
            sorted_results = sorted(results, key=lambda x: x['val_loss'])
            print(f"\n{cat.upper()}:")
            print(f"{'='*60}")
            for i, r in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {r['name']}: {r['val_loss']:.4f}")

    def save_summary(self):
        """Save comprehensive summary."""
        sorted_results = sorted(self.results, key=lambda x: x['val_loss'])

        summary = {
            'timestamp': self.timestamp,
            'total_experiments': len(self.results),
            'best_result': sorted_results[0] if sorted_results else None,
            'top_10': sorted_results[:10],
            'all_results_ranked': sorted_results,
            'category_analysis': self._analyze_by_category(),
        }

        summary_file = self.output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("HYPERPARAMETER OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest configuration: {sorted_results[0]['name']}")
        print(f"Validation loss: {sorted_results[0]['val_loss']:.4f}")
        print(f"\nTop 5 configurations:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['name']}: {r['val_loss']:.4f}")

        print(f"\nResults saved to: {self.output_file}")
        print(f"Summary saved to: {summary_file}")

        return summary

    def _analyze_by_category(self):
        """Analyze best config for each category."""
        categories = {}
        for result in self.results:
            cat = result.get('category', 'baseline')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        analysis = {}
        for cat, results in categories.items():
            best = min(results, key=lambda x: x['val_loss'])
            analysis[cat] = {
                'best_config': best['name'],
                'best_val_loss': best['val_loss'],
                'num_tested': len(results),
            }

        return analysis


def main():
    optimizer = GatedHyperparamOptimizer()
    optimizer.run_all()


if __name__ == '__main__':
    main()
