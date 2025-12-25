#!/usr/bin/env python3
"""
Alpha parameter sweep for nGPT on H100.
Tests 12 different alpha values to find optimal residual scaling.
"""

import subprocess
import json
import numpy as np
from datetime import datetime

# Alpha values to test (12 iterations)
alpha_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]

# Fixed logit scale (from Mac testing)
LOGIT_SCALE = 10.0

# Training config
STEPS = 500
BATCH_SIZE = 16
OUTPUT_FILE = f'alpha_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'

print(f"Alpha Parameter Sweep for nGPT on H100")
print(f"=" * 80)
print(f"Alpha values to test: {alpha_values}")
print(f"Total experiments: {len(alpha_values)}")
print(f"Logit scale (fixed): {LOGIT_SCALE}")
print(f"Steps per experiment: {STEPS}")
print(f"Output file: {OUTPUT_FILE}")
print(f"=" * 80)

for i, alpha in enumerate(alpha_values, 1):
    print(f"\n[{i}/{len(alpha_values)}] Testing alpha={alpha}")
    print("-" * 80)

    cmd = [
        'python', 'train_h100.py',
        '--alpha', str(alpha),
        '--logit-scale', str(LOGIT_SCALE),
        '--steps', str(STEPS),
        '--batch-size', str(BATCH_SIZE),
        '--output', OUTPUT_FILE
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed successfully")
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
print("-" * 80)
print(f"{'Rank':<6} {'Alpha':<8} {'Val Loss':<12} {'Train Loss':<12} {'Reduction':<12}")
print("-" * 80)
for i, r in enumerate(results, 1):
    print(f"{i:<6} {r['alpha']:<8.2f} {r['val_loss']:<12.4f} {r['final_train_loss']:<12.4f} {r['train_loss_reduction']:<12.1f}%")

print("\n" + "=" * 80)
print(f"BEST CONFIGURATION:")
best = results[0]
print(f"  Alpha: {best['alpha']}")
print(f"  Logit scale: {best['logit_scale']}")
print(f"  Validation loss: {best['val_loss']:.4f}")
print(f"  Training loss: {best['final_train_loss']:.4f}")
print(f"  Loss reduction: {best['train_loss_reduction']:.1f}%")
print(f"  Throughput: {best['tokens_per_sec']:,.0f} tokens/sec")
print("=" * 80)

# Save summary
summary = {
    'best_alpha': best['alpha'],
    'best_val_loss': best['val_loss'],
    'all_results': results
}

summary_file = OUTPUT_FILE.replace('.jsonl', '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to {summary_file}")
