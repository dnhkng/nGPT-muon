#!/usr/bin/env python3
"""
Optimization comparison sweep for nGPT on H100.
Tests different optimization combinations from DESIGN_nGPT_MUON.md
"""

import subprocess
import json
import time
from datetime import datetime

# Optimization configurations to test
configs = [
    # Baseline
    {'name': 'baseline', 'lazy_proj_freq': 0, 'scalar_alpha': False},

    # Lazy projection only (Optimization #1)
    {'name': 'lazy_proj_3', 'lazy_proj_freq': 3, 'scalar_alpha': False},
    {'name': 'lazy_proj_5', 'lazy_proj_freq': 5, 'scalar_alpha': False},
    {'name': 'lazy_proj_10', 'lazy_proj_freq': 10, 'scalar_alpha': False},

    # Scalar alpha only (Optimization #8)
    {'name': 'scalar_alpha', 'lazy_proj_freq': 0, 'scalar_alpha': True},

    # Combined optimizations
    {'name': 'lazy_proj_5_scalar', 'lazy_proj_freq': 5, 'scalar_alpha': True},
    {'name': 'lazy_proj_10_scalar', 'lazy_proj_freq': 10, 'scalar_alpha': True},
]

# Fixed parameters (from alpha sweep)
ALPHA = 0.28
LOGIT_SCALE = 10.0
N_LAYER = 8
N_EMBD = 384
STEPS = 500
BATCH_SIZE = 16
OUTPUT_FILE = f'optimization_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'

print(f"Optimization Comparison Sweep for nGPT on H100")
print(f"=" * 80)
print(f"Configurations to test: {len(configs)}")
print(f"Model: {N_LAYER} layers x {N_EMBD} dim (52.8M params)")
print(f"Alpha: {ALPHA} (optimal from previous sweep)")
print(f"Steps per experiment: {STEPS}")
print(f"Output file: {OUTPUT_FILE}")
print(f"=" * 80)

for i, config in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")
    print(f"  Lazy projection freq: {config['lazy_proj_freq'] if config['lazy_proj_freq'] > 0 else 'every step'}")
    print(f"  Scalar alpha: {config['scalar_alpha']}")
    print("-" * 80)

    cmd = [
        'uv', 'run', 'python', 'train_h100_optimized.py',
        '--alpha', str(ALPHA),
        '--logit-scale', str(LOGIT_SCALE),
        '--n-layer', str(N_LAYER),
        '--n-embd', str(N_EMBD),
        '--steps', str(STEPS),
        '--batch-size', str(BATCH_SIZE),
        '--lazy-proj-freq', str(config['lazy_proj_freq']),
        '--output', OUTPUT_FILE
    ]

    if config['scalar_alpha']:
        cmd.append('--scalar-alpha')

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with error: {e}")
        continue

print(f"\n" + "=" * 80)
print(f"Sweep complete! Results saved to {OUTPUT_FILE}")
print(f"=" * 80)

# Analyze results
print(f"\nAnalyzing results...")
results = []
with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        results.append(json.loads(line))

if not results:
    print("No results to analyze!")
    exit(1)

# Sort by validation loss
results.sort(key=lambda x: x['val_loss'])

print(f"\nAll results ranked by validation loss:")
print("-" * 100)
print(f"{'Rank':<6} {'Config':<20} {'Val Loss':<12} {'Train Loss':<12} {'Reduction':<12} {'Tok/s':<12}")
print("-" * 100)

# Add config name to each result
config_map = {i: c['name'] for i, c in enumerate(configs)}
for result in results:
    # Find matching config
    for idx, cfg in enumerate(configs):
        if (result.get('lazy_proj_freq') == cfg['lazy_proj_freq'] and
            result.get('scalar_alpha') == cfg['scalar_alpha']):
            result['config_name'] = cfg['name']
            break

for i, r in enumerate(results, 1):
    config_name = r.get('config_name', 'unknown')
    print(f"{i:<6} {config_name:<20} {r['val_loss']:<12.4f} {r['final_train_loss']:<12.4f} "
          f"{r['train_loss_reduction']:<12.1f}% {r['tokens_per_sec']:<12,.0f}")

# Performance analysis
print("\n" + "=" * 100)
print(f"PERFORMANCE ANALYSIS:")
print("=" * 100)

best = results[0]
baseline = next((r for r in results if r.get('lazy_proj_freq', 0) == 0 and not r.get('scalar_alpha', False)), results[0])

print(f"\nBest Configuration: {best.get('config_name', 'unknown')}")
print(f"  Validation loss: {best['val_loss']:.4f}")
print(f"  Training loss: {best['final_train_loss']:.4f}")
print(f"  Loss reduction: {best['train_loss_reduction']:.1f}%")
print(f"  Throughput: {best['tokens_per_sec']:,.0f} tokens/sec")
print(f"  Projections: {best.get('projection_count', 'N/A')}/{best['steps']}")

print(f"\nBaseline Configuration:")
print(f"  Validation loss: {baseline['val_loss']:.4f}")
print(f"  Throughput: {baseline['tokens_per_sec']:,.0f} tokens/sec")

# Compute improvements
val_loss_improvement = ((baseline['val_loss'] - best['val_loss']) / baseline['val_loss']) * 100
throughput_improvement = ((best['tokens_per_sec'] - baseline['tokens_per_sec']) / baseline['tokens_per_sec']) * 100

print(f"\nImprovement over baseline:")
print(f"  Validation loss: {val_loss_improvement:+.2f}%")
print(f"  Throughput: {throughput_improvement:+.2f}%")

# Lazy projection analysis
print(f"\nLazy Projection Impact:")
lazy_proj_results = [r for r in results if r.get('lazy_proj_freq', 0) > 0 and not r.get('scalar_alpha', False)]
if lazy_proj_results:
    for r in lazy_proj_results:
        freq = r.get('lazy_proj_freq')
        proj_count = r.get('projection_count', 0)
        proj_reduction = ((baseline.get('projection_count', 500) - proj_count) / baseline.get('projection_count', 500)) * 100
        throughput_change = ((r['tokens_per_sec'] - baseline['tokens_per_sec']) / baseline['tokens_per_sec']) * 100
        val_loss_change = ((r['val_loss'] - baseline['val_loss']) / baseline['val_loss']) * 100

        print(f"  Freq={freq}: {proj_count} projections ({proj_reduction:.0f}% reduction), "
              f"throughput {throughput_change:+.1f}%, val loss {val_loss_change:+.2f}%")

# Scalar alpha analysis
print(f"\nScalar Alpha Impact:")
scalar_results = [r for r in results if r.get('scalar_alpha', False) and r.get('lazy_proj_freq', 0) == 0]
if scalar_results:
    r = scalar_results[0]
    param_reduction = (baseline['n_params'] - r['n_params']) / baseline['n_params'] * 100
    throughput_change = ((r['tokens_per_sec'] - baseline['tokens_per_sec']) / baseline['tokens_per_sec']) * 100
    val_loss_change = ((r['val_loss'] - baseline['val_loss']) / baseline['val_loss']) * 100

    print(f"  Parameters: {param_reduction:.1f}% reduction ({baseline['n_params']:,} → {r['n_params']:,})")
    print(f"  Throughput: {throughput_change:+.1f}%")
    print(f"  Validation loss: {val_loss_change:+.2f}%")

print("=" * 100)

# Save summary
summary = {
    'best_config': best.get('config_name'),
    'best_val_loss': best['val_loss'],
    'baseline_val_loss': baseline['val_loss'],
    'improvement_percent': val_loss_improvement,
    'all_results': results
}

summary_file = OUTPUT_FILE.replace('.jsonl', '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to {summary_file}")
