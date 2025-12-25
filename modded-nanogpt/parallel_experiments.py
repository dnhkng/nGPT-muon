#!/usr/bin/env python3
"""
Parallel nGPT Experiment Runner - Uses Both GPUs Simultaneously
Runs 2 experiments in parallel on 2x GH200 GPUs for 2x speedup
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

class ParallelExperimentRunner:
    """Manages parallel hyperparameter exploration across multiple GPUs."""

    def __init__(self, num_gpus=2, output_dir="parallel_experiments"):
        self.num_gpus = num_gpus
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"experiments_{self.timestamp}.jsonl"
        self.all_results = []
        self.lock = threading.Lock()

    def run_single_experiment(self, gpu_id, round_num, config, description):
        """Run a single experiment on a specific GPU."""
        print(f"\n[GPU {gpu_id}] {'='*70}")
        print(f"[GPU {gpu_id}] ROUND {round_num}: {description}")
        print(f"[GPU {gpu_id}] {'='*70}\n")

        # Set CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Build command
        cmd = [
            "python3", "train_h100_optimized.py",
            "--steps", str(config.get('steps', 500)),
            "--alpha", str(config.get('alpha', 0.28)),
            "--logit-scale", str(config.get('logit_scale', 10.0)),
            "--batch-size", str(config.get('batch_size', 16)),
            "--lr", str(config.get('lr', 0.001)),
            "--n-layer", str(config.get('n_layer', 8)),
            "--n-embd", str(config.get('n_embd', 384)),
            "--lazy-proj-freq", str(config.get('lazy_proj_freq', 10)),
            "--output", str(self.results_file),
        ]

        if config.get('scalar_alpha', True):
            cmd.append("--scalar-alpha")

        # Run experiment
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   check=True, env=env)
            elapsed = time.time() - t0

            # Parse result from last line of JSONL file
            with self.lock:
                with open(self.results_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_result = json.loads(lines[-1])
                        last_result['round'] = round_num
                        last_result['description'] = description
                        last_result['config'] = config
                        last_result['gpu_id'] = gpu_id
                        self.all_results.append(last_result)

                        print(f"\n[GPU {gpu_id}] {'='*70}")
                        print(f"[GPU {gpu_id}] ROUND {round_num} COMPLETE")
                        print(f"[GPU {gpu_id}] {'='*70}")
                        print(f"[GPU {gpu_id}] Validation Loss: {last_result['val_loss']:.4f}")
                        print(f"[GPU {gpu_id}] Train Loss: {last_result['final_train_loss']:.4f}")
                        print(f"[GPU {gpu_id}] Throughput: {last_result['tokens_per_sec']:,.0f} tok/s")
                        print(f"[GPU {gpu_id}] Time: {elapsed:.1f}s")
                        print(f"[GPU {gpu_id}] {'='*70}\n")

                        return last_result
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] ERROR in round {round_num}: {e}")
            print(f"[GPU {gpu_id}] STDERR: {e.stderr}")
            return None

    def run_experiments_parallel(self, experiments):
        """Run experiments in parallel across available GPUs."""
        print(f"\n{'='*80}")
        print(f"Parallel Experiment Runner - {self.num_gpus} GPUs")
        print(f"{'='*80}\n")
        print(f"Running {len(experiments)} experiments in parallel on {self.num_gpus} GPUs")
        print(f"Expected speedup: {self.num_gpus}x")
        print(f"Expected completion time: ~{len(experiments) // self.num_gpus * 30} seconds\n")

        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit all experiments
            futures = []
            for i, exp in enumerate(experiments):
                gpu_id = i % self.num_gpus
                future = executor.submit(
                    self.run_single_experiment,
                    gpu_id,
                    exp['round'],
                    exp['config'],
                    exp['description']
                )
                futures.append(future)

                # Brief delay to stagger GPU initialization
                time.sleep(0.5)

            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Experiment failed with exception: {e}")

        print(f"\n{'='*80}")
        print(f"All {len(experiments)} experiments completed")
        print(f"{'='*80}\n")

    def save_summary(self):
        """Save comprehensive summary of all experiments."""
        summary_file = self.output_dir / f"experiments_{self.timestamp}_summary.json"

        # Sort by validation loss
        sorted_results = sorted(self.all_results, key=lambda x: x['val_loss'])

        summary = {
            'timestamp': self.timestamp,
            'total_rounds': len(self.all_results),
            'num_gpus': self.num_gpus,
            'best_result': sorted_results[0] if sorted_results else None,
            'top_5_results': sorted_results[:5],
            'all_results_ranked': sorted_results,
            'analysis': {
                'val_loss_range': [
                    min(r['val_loss'] for r in self.all_results),
                    max(r['val_loss'] for r in self.all_results)
                ] if self.all_results else [0, 0],
                'throughput_range': [
                    min(r['tokens_per_sec'] for r in self.all_results),
                    max(r['tokens_per_sec'] for r in self.all_results)
                ] if self.all_results else [0, 0],
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("SUMMARY SAVED")
        print(f"{'='*80}")
        print(f"File: {summary_file}")
        print(f"Best validation loss: {summary['best_result']['val_loss']:.4f}")
        print(f"Best configuration: {summary['best_result']['description']}")
        print(f"{'='*80}\n")

        return summary


def example_experiments():
    """Example: Run a quick test sweep in parallel."""

    # Quick 4-experiment test
    experiments = []

    # Test different batch sizes
    for i, bs in enumerate([16, 24, 32, 48], 1):
        experiments.append({
            'round': i,
            'config': {
                'alpha': 0.28,
                'logit_scale': 15.0,
                'lr': 0.001,
                'batch_size': bs,
                'n_layer': 8,
                'n_embd': 384,
                'lazy_proj_freq': 7,
                'scalar_alpha': True,
                'steps': 500,
            },
            'description': f'Parallel Test: batch_size={bs}'
        })

    return experiments


def main():
    """Example usage of parallel experiment runner."""

    # Check for data
    if not os.path.exists('data/fineweb10B/fineweb_val_000000.bin'):
        print("ERROR: FineWeb data not found!")
        print("Please ensure data is in: data/fineweb10B/")
        return

    # Create runner
    runner = ParallelExperimentRunner(num_gpus=2)

    # Get experiments
    experiments = example_experiments()

    print(f"Running {len(experiments)} experiments in parallel on 2 GPUs")
    print(f"This will be {len(experiments) // 2}x faster than sequential!\n")

    # Run in parallel
    t0 = time.time()
    runner.run_experiments_parallel(experiments)
    elapsed = time.time() - t0

    # Save summary
    summary = runner.save_summary()

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Average per experiment: {elapsed / len(experiments):.1f}s")
    print(f"Speedup from parallel execution: ~2x\n")

    return summary


if __name__ == '__main__':
    # Usage:
    # python3 parallel_experiments.py
    #
    # This will run 4 experiments in parallel on 2 GPUs
    # Experiments 0,2 on GPU 0, experiments 1,3 on GPU 1
    # Total time: ~60s (vs ~120s sequential)

    summary = main()
