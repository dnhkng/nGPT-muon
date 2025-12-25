#!/usr/bin/env python3
"""
Hyperparameter sweep for nGPT on H100.
Tests different alpha and logit_scale values to find optimal configuration.
"""

import subprocess
import json
import numpy as np
from datetime import datetime

# Hyperparameter grid
alpha_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
logit_scale_values = [5.0, 7.5, 10.0, 12.5, 15.0]

# Training config
STEPS = 500
BATCH_SIZE = 16
OUTPUT_FILE = f'sweep_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'

print(f"Hyperparameter Sweep for nGPT")
print(f"=" * 80)
print(f"Alpha values: {alpha_values}")
print(f"Logit scale values: {logit_scale_values}")
print(f"Total experiments: {len(alpha_values) * len(logit_scale_values)}")
print(f"Steps per experiment: {STEPS}")
print(f"Output file: {OUTPUT_FILE}")
print(f"=" * 80)

results = []
exp_num = 0

for alpha in alpha_values:
    for logit_scale in logit_scale_values:
        exp_num += 1
        total_exps = len(alpha_values) * len(logit_scale_values)

        print(f"\n[{exp_num}/{total_exps}] Running: alpha={alpha}, logit_scale={logit_scale}")
        print("-" * 80)

        cmd = [
            'python', 'train_h100.py',
            '--alpha', str(alpha),
            '--logit-scale', str(logit_scale),
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

print(f"\nTop 5 configurations by validation loss:")
print("-" * 80)
for i, r in enumerate(results[:5], 1):
    print(f"{i}. alpha={r['alpha']:.2f}, logit_scale={r['logit_scale']:.1f}")
    print(f"   Val loss: {r['val_loss']:.4f}")
    print(f"   Train loss: {r['final_train_loss']:.4f}")
    print(f"   Loss reduction: {r['train_loss_reduction']:.1f}%")
    print()

print("=" * 80)
print(f"Best configuration:")
best = results[0]
print(f"  Alpha: {best['alpha']}")
print(f"  Logit scale: {best['logit_scale']}")
print(f"  Validation loss: {best['val_loss']:.4f}")
print(f"  Training loss: {best['final_train_loss']:.4f}")
print("=" * 80)
