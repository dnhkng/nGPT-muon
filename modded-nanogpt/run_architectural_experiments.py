#!/usr/bin/env python3
"""
Run all architectural variation experiments systematically.
"""

import subprocess
import time
import json
from datetime import datetime

# Define all experiments
experiments = [
    # Baseline
    {
        'name': 'baseline',
        'args': ['--name', 'baseline', '--hypothesis', '0'],
    },

    # H1: Layer-specific alpha (3 variants)
    {
        'name': 'layer_alpha_linear',
        'args': ['--name', 'layer_alpha_linear', '--hypothesis', '1',
                 '--layer-alpha-schedule', 'linear'],
    },
    {
        'name': 'layer_alpha_upsidedown_u',
        'args': ['--name', 'layer_alpha_upsidedown_u', '--hypothesis', '1',
                 '--layer-alpha-schedule', 'upsidedown_u'],
    },
    {
        'name': 'layer_alpha_downsideup_u',
        'args': ['--name', 'layer_alpha_downsideup_u', '--hypothesis', '1',
                 '--layer-alpha-schedule', 'downsideup_u'],
    },

    # H2: Learnable logit scale
    {
        'name': 'learnable_logit_scale',
        'args': ['--name', 'learnable_logit_scale', '--hypothesis', '2',
                 '--learnable-logit-scale'],
    },

    # H3: Gated residuals
    {
        'name': 'gated_residuals',
        'args': ['--name', 'gated_residuals', '--hypothesis', '3',
                 '--gated-residuals'],
    },

    # H5: Asymmetric alpha (3 variants)
    {
        'name': 'asymmetric_0.20_0.35',
        'args': ['--name', 'asymmetric_0.20_0.35', '--hypothesis', '5',
                 '--asymmetric-alpha', '--alpha-attn', '0.20', '--alpha-mlp', '0.35'],
    },
    {
        'name': 'asymmetric_0.15_0.40',
        'args': ['--name', 'asymmetric_0.15_0.40', '--hypothesis', '5',
                 '--asymmetric-alpha', '--alpha-attn', '0.15', '--alpha-mlp', '0.40'],
    },
    {
        'name': 'asymmetric_0.25_0.30',
        'args': ['--name', 'asymmetric_0.25_0.30', '--hypothesis', '5',
                 '--asymmetric-alpha', '--alpha-attn', '0.25', '--alpha-mlp', '0.30'],
    },

    # H6: Progressive radius (3 variants)
    {
        'name': 'progressive_radius_0.95',
        'args': ['--name', 'progressive_radius_0.95', '--hypothesis', '6',
                 '--progressive-radius', '--target-radius', '0.95'],
    },
    {
        'name': 'progressive_radius_1.05',
        'args': ['--name', 'progressive_radius_1.05', '--hypothesis', '6',
                 '--progressive-radius', '--target-radius', '1.05'],
    },
    {
        'name': 'progressive_radius_1.10',
        'args': ['--name', 'progressive_radius_1.10', '--hypothesis', '6',
                 '--progressive-radius', '--target-radius', '1.10'],
    },
]

print(f"=" * 80)
print(f"ARCHITECTURAL EXPERIMENTS - {len(experiments)} total")
print(f"=" * 80)
print(f"Model: 11 layers × 768 dim (modded-nanogpt size)")
print(f"Dataset: FineWeb (2 shards, ~400K tokens)")
print(f"Training: 300 steps per experiment")
print(f"Expected time: ~{len(experiments) * 40} seconds (~{len(experiments) * 40 / 60:.1f} minutes)")
print(f"=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'architectural_results_{timestamp}.jsonl'

all_results = []
for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
    print(f"{'='*80}\n")

    # Build command
    cmd = ['python3', 'train_architectural.py'] + exp['args'] + ['--output', output_file]

    # Run experiment
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        # Parse result from last line of JSONL
        with open(output_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_result = json.loads(lines[-1])
                all_results.append(last_result)
                print(f"\n✓ Completed in {elapsed:.1f}s")
                print(f"  Val loss: {last_result['val_loss']:.4f}")
                print(f"  Throughput: {last_result['tokens_per_sec']:,.0f} tok/s")
    else:
        print(f"\n✗ Failed with return code {result.returncode}")

    time.sleep(1)

# Generate summary
print(f"\n{'='*80}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"{'='*80}\n")

# Sort by validation loss
sorted_results = sorted(all_results, key=lambda x: x['val_loss'])

print(f"TOP 5 CONFIGURATIONS:\n")
for i, result in enumerate(sorted_results[:5], 1):
    print(f"{i}. {result['name']}")
    print(f"   Val loss: {result['val_loss']:.4f}")
    print(f"   Hypothesis: H{result['hypothesis']}" if result['hypothesis'] > 0 else "   Baseline")
    print()

# Save summary
summary_file = output_file.replace('.jsonl', '_summary.json')
summary = {
    'timestamp': timestamp,
    'total_experiments': len(all_results),
    'best_result': sorted_results[0] if sorted_results else None,
    'top_5': sorted_results[:5],
    'all_results_ranked': sorted_results,
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Results saved to: {output_file}")
print(f"Summary saved to: {summary_file}")
print(f"\nBest configuration: {sorted_results[0]['name']} (val_loss={sorted_results[0]['val_loss']:.4f})")
