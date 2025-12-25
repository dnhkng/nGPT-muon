#!/usr/bin/env python3
"""
Advanced nGPT Experiments - 20 Rounds of Systematic Optimization
Builds on previous results to explore remaining hyperparameter space
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

class ExperimentRunner:
    """Manages systematic hyperparameter exploration across 20 rounds."""

    def __init__(self, output_dir="advanced_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"experiments_{self.timestamp}.jsonl"
        self.all_results = []

    def run_experiment(self, round_num, config, description):
        """Run a single experiment with given configuration."""
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/20: {description}")
        print(f"{'='*80}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")

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
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed = time.time() - t0

            # Parse result from last line of JSONL file
            with open(self.results_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_result = json.loads(lines[-1])
                    last_result['round'] = round_num
                    last_result['description'] = description
                    last_result['config'] = config
                    self.all_results.append(last_result)

                    print(f"\n{'='*80}")
                    print(f"ROUND {round_num} COMPLETE")
                    print(f"{'='*80}")
                    print(f"Validation Loss: {last_result['val_loss']:.4f}")
                    print(f"Train Loss: {last_result['final_train_loss']:.4f}")
                    print(f"Throughput: {last_result['tokens_per_sec']:,.0f} tok/s")
                    print(f"Time: {elapsed:.1f}s")
                    print(f"{'='*80}\n")

                    return last_result
        except subprocess.CalledProcessError as e:
            print(f"ERROR in round {round_num}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None

    def save_summary(self):
        """Save comprehensive summary of all experiments."""
        summary_file = self.output_dir / f"experiments_{self.timestamp}_summary.json"

        # Sort by validation loss
        sorted_results = sorted(self.all_results, key=lambda x: x['val_loss'])

        summary = {
            'timestamp': self.timestamp,
            'total_rounds': len(self.all_results),
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

def define_experiments():
    """Define all 20 experimental rounds."""

    # Baseline config (from previous best)
    baseline = {
        'alpha': 0.28,
        'logit_scale': 10.0,
        'lr': 0.001,
        'batch_size': 16,
        'n_layer': 8,
        'n_embd': 384,
        'lazy_proj_freq': 10,
        'scalar_alpha': True,
        'steps': 500,
    }

    experiments = []

    # ========== ROUNDS 1-5: Learning Rate Sweep ==========
    print("Planning Rounds 1-5: Learning Rate Optimization")
    lr_values = [0.0005, 0.001, 0.0015, 0.002, 0.003]
    for i, lr in enumerate(lr_values, 1):
        config = baseline.copy()
        config['lr'] = lr
        experiments.append({
            'round': i,
            'config': config,
            'description': f'Learning Rate Sweep: lr={lr}'
        })

    # ========== ROUNDS 6-10: Batch Size Scaling ==========
    print("Planning Rounds 6-10: Batch Size Optimization")
    batch_sizes = [8, 16, 24, 32, 48]
    for i, bs in enumerate(batch_sizes, 6):
        config = baseline.copy()
        config['batch_size'] = bs
        experiments.append({
            'round': i,
            'config': config,
            'description': f'Batch Size Scaling: batch_size={bs}'
        })

    # ========== ROUNDS 11-13: Logit Scale Fine-tuning ==========
    print("Planning Rounds 11-13: Logit Scale Fine-tuning")
    logit_scales = [8.0, 12.0, 15.0]
    for i, ls in enumerate(logit_scales, 11):
        config = baseline.copy()
        config['logit_scale'] = ls
        experiments.append({
            'round': i,
            'config': config,
            'description': f'Logit Scale: scale={ls}'
        })

    # ========== ROUNDS 14-16: Lazy Projection Frequency ==========
    print("Planning Rounds 14-16: Lazy Projection Fine-tuning")
    proj_freqs = [7, 15, 20]
    for i, freq in enumerate(proj_freqs, 14):
        config = baseline.copy()
        config['lazy_proj_freq'] = freq
        experiments.append({
            'round': i,
            'config': config,
            'description': f'Lazy Projection: freq={freq}'
        })

    # ========== ROUND 17: Model Scaling (Deeper) ==========
    print("Planning Round 17: Model Depth Scaling")
    config = baseline.copy()
    config['n_layer'] = 10
    experiments.append({
        'round': 17,
        'config': config,
        'description': 'Model Scaling: 10 layers (deeper)'
    })

    # ========== ROUND 18: Model Scaling (Wider) ==========
    print("Planning Round 18: Model Width Scaling")
    config = baseline.copy()
    config['n_embd'] = 512
    experiments.append({
        'round': 18,
        'config': config,
        'description': 'Model Scaling: 512 embedding dim (wider)'
    })

    # ========== ROUND 19: Combined Optimal Settings ==========
    print("Planning Round 19: Combined Best Settings from Previous Rounds")
    # This will be updated after rounds 1-18
    experiments.append({
        'round': 19,
        'config': baseline.copy(),  # Will update based on results
        'description': 'Combined Optimal Settings (placeholder)'
    })

    # ========== ROUND 20: Extended Training ==========
    print("Planning Round 20: Extended Training with Best Config")
    config = baseline.copy()
    config['steps'] = 1500  # 3x longer
    experiments.append({
        'round': 20,
        'config': config,
        'description': 'Extended Training: 1500 steps with best config'
    })

    return experiments

def main():
    """Run all 20 experimental rounds."""
    print(f"\n{'='*80}")
    print("Advanced nGPT Experiments - 20 Rounds")
    print(f"{'='*80}\n")

    # Check for data
    if not os.path.exists('data/fineweb10B/fineweb_val_000000.bin'):
        print("ERROR: FineWeb data not found!")
        print("Please ensure data is in: data/fineweb10B/")
        return

    runner = ExperimentRunner()
    experiments = define_experiments()

    print(f"\nPlanned {len(experiments)} experiments\n")

    # Run rounds 1-18
    for exp in experiments[:18]:
        result = runner.run_experiment(exp['round'], exp['config'], exp['description'])
        if result is None:
            print(f"Stopping due to error in round {exp['round']}")
            break

        # Brief pause between experiments
        time.sleep(2)

    # Update round 19 with best settings from rounds 1-18
    if len(runner.all_results) >= 18:
        print(f"\n{'='*80}")
        print("Analyzing rounds 1-18 to optimize round 19...")
        print(f"{'='*80}\n")

        # Find best settings for each category
        lr_results = [(r['lr'], r['val_loss']) for r in runner.all_results[0:5]]
        best_lr = min(lr_results, key=lambda x: x[1])[0]

        bs_results = [(r['batch_size'], r['val_loss']) for r in runner.all_results[5:10]]
        best_bs = min(bs_results, key=lambda x: x[1])[0]

        ls_results = [(r['logit_scale'], r['val_loss']) for r in runner.all_results[10:13]]
        best_ls = min(ls_results, key=lambda x: x[1])[0]

        pf_results = [(r['lazy_proj_freq'], r['val_loss']) for r in runner.all_results[13:16]]
        best_pf = min(pf_results, key=lambda x: x[1])[0]

        # Check if scaling helped
        layer_result = runner.all_results[16]
        embd_result = runner.all_results[17]
        baseline_val_loss = min(r['val_loss'] for r in runner.all_results[:16])

        best_n_layer = 10 if layer_result['val_loss'] < baseline_val_loss else 8
        best_n_embd = 512 if embd_result['val_loss'] < baseline_val_loss else 384

        # Update round 19 config
        experiments[18]['config'] = {
            'alpha': 0.28,  # Keep from alpha sweep
            'logit_scale': best_ls,
            'lr': best_lr,
            'batch_size': best_bs,
            'n_layer': best_n_layer,
            'n_embd': best_n_embd,
            'lazy_proj_freq': best_pf,
            'scalar_alpha': True,
            'steps': 500,
        }
        experiments[18]['description'] = f'Combined Optimal: lr={best_lr}, bs={best_bs}, ls={best_ls}, pf={best_pf}, layers={best_n_layer}, embd={best_n_embd}'

        print(f"Round 19 optimized configuration:")
        print(json.dumps(experiments[18]['config'], indent=2))
        print()

        # Run round 19
        runner.run_experiment(experiments[18]['round'], experiments[18]['config'], experiments[18]['description'])
        time.sleep(2)

        # Update round 20 with best overall config
        best_overall = min(runner.all_results, key=lambda x: x['val_loss'])
        experiments[19]['config'] = best_overall['config'].copy()
        experiments[19]['config']['steps'] = 1500
        experiments[19]['description'] = f'Extended Training: 1500 steps with best config (round {best_overall["round"]})'

        # Run round 20
        runner.run_experiment(experiments[19]['round'], experiments[19]['config'], experiments[19]['description'])

    # Save comprehensive summary
    summary = runner.save_summary()

    # Print final report
    print(f"\n{'='*80}")
    print("ALL 20 ROUNDS COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Configuration:")
    print(f"  Round: {summary['best_result']['round']}")
    print(f"  Description: {summary['best_result']['description']}")
    print(f"  Validation Loss: {summary['best_result']['val_loss']:.4f}")
    print(f"  Throughput: {summary['best_result']['tokens_per_sec']:,.0f} tok/s")
    print(f"\nTop 5 Configurations:")
    for i, result in enumerate(summary['top_5_results'], 1):
        print(f"  {i}. Round {result['round']}: {result['val_loss']:.4f} - {result['description']}")
    print(f"\n{'='*80}\n")

    return summary

if __name__ == '__main__':
    summary = main()
