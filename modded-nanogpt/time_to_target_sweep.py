#!/usr/bin/env python3
"""
Two-Stage Hyperparameter Sweep for Time-to-Target Optimization

Objective: Minimize wall-clock time to reach validation loss < 5.0
Primary metric: seconds to target, NOT iterations or final loss quality

Stage 1: Broad sweep (48 configurations)
Stage 2: High-resolution sweep around optimal region (36 configurations)
"""

import subprocess
import json
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product

# Constants
TARGET_LOSS = 5.0
MAX_STEPS = 5000
VALIDATION_FREQ = 50
TIMEOUT_SECONDS = 600  # 10 minutes per experiment

# Fixed baseline configuration
BASE_CONFIG = {
    'n_layer': 11,  # modded-nanogpt scale (REQUIRED)
    'n_embd': 768,
    'alpha': 0.15,
    'gated_residuals': True,
    'optimizer': 'muon',
}

def create_stage1_configs():
    """Generate all 48 Stage 1 configurations."""
    configs = []

    # Group A: Orthogonalization Method (8 configs)
    # Test both orthog methods with different base configs
    group_a_bases = [
        {'lr': 0.003, 'lazy_proj_freq': 7, 'batch_size': 32, 'momentum': 0.95, 'variance_reduction': False, 'cautious_wd': False},
        {'lr': 0.01, 'lazy_proj_freq': 10, 'batch_size': 32, 'momentum': 0.95, 'variance_reduction': False, 'cautious_wd': False},
        {'lr': 0.003, 'lazy_proj_freq': 7, 'batch_size': 32, 'momentum': 0.95, 'variance_reduction': True, 'cautious_wd': True},
        {'lr': 0.01, 'lazy_proj_freq': 10, 'batch_size': 64, 'momentum': 0.90, 'variance_reduction': False, 'cautious_wd': False},
    ]

    for i, base in enumerate(group_a_bases):
        for orthog in ['newton_schulz', 'polar_express']:
            config = {**BASE_CONFIG, **base, 'orthog_method': orthog}
            config['name'] = f'A{i+1}_{orthog[:2]}'
            configs.append(config)

    # Group B: VR/CWD Combinations (4 configs)
    # With Newton-Schulz orthog
    group_b_vr_cwd = [
        (False, False),  # baseline
        (True, False),   # VR only
        (False, True),   # CWD only
        (True, True),    # VR + CWD
    ]

    for i, (vr, cwd) in enumerate(group_b_vr_cwd):
        config = {
            **BASE_CONFIG,
            'lr': 0.003,
            'lazy_proj_freq': 7,
            'batch_size': 32,
            'momentum': 0.95,
            'orthog_method': 'newton_schulz',
            'variance_reduction': vr,
            'cautious_wd': cwd,
            'name': f'B{i+1}_vr{int(vr)}_cwd{int(cwd)}'
        }
        configs.append(config)

    # Group C: Lazy Projection Frequency (6 configs)
    # Use Newton-Schulz (likely best from Group A)
    for lazy_freq in [3, 5, 7, 10, 15, 20]:
        config = {
            **BASE_CONFIG,
            'lr': 0.003,
            'lazy_proj_freq': lazy_freq,
            'batch_size': 32,
            'momentum': 0.95,
            'orthog_method': 'newton_schulz',
            'variance_reduction': False,
            'cautious_wd': False,
            'name': f'C_lazy{lazy_freq}'
        }
        configs.append(config)

    # Group D: Learning Rate (6 configs)
    for lr in [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]:
        config = {
            **BASE_CONFIG,
            'lr': lr,
            'lazy_proj_freq': 7,
            'batch_size': 32,
            'momentum': 0.95,
            'orthog_method': 'newton_schulz',
            'variance_reduction': False,
            'cautious_wd': False,
            'name': f'D_lr{lr}'
        }
        configs.append(config)

    # Group E: Batch Size (4 configs)
    # Adjust LR proportionally (linear scaling rule)
    for batch_size in [16, 32, 64, 128]:
        lr_scaled = 0.003 * (batch_size / 32)
        config = {
            **BASE_CONFIG,
            'lr': lr_scaled,
            'lazy_proj_freq': 7,
            'batch_size': batch_size,
            'momentum': 0.95,
            'orthog_method': 'newton_schulz',
            'variance_reduction': False,
            'cautious_wd': False,
            'name': f'E_batch{batch_size}'
        }
        configs.append(config)

    # Group F: Momentum (4 configs)
    for momentum in [0.85, 0.90, 0.95, 0.98]:
        config = {
            **BASE_CONFIG,
            'lr': 0.003,
            'lazy_proj_freq': 7,
            'batch_size': 32,
            'momentum': momentum,
            'orthog_method': 'newton_schulz',
            'variance_reduction': False,
            'cautious_wd': False,
            'name': f'F_mom{momentum}'
        }
        configs.append(config)

    # Group G: Promising Combinations (16 configs)
    # Hand-picked based on domain knowledge
    group_g_configs = [
        {'lr': 0.01, 'lazy_proj_freq': 10, 'batch_size': 64, 'momentum': 0.95, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G01_fast'},
        {'lr': 0.01, 'lazy_proj_freq': 7, 'batch_size': 64, 'momentum': 0.90, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G02_fast_safe'},
        {'lr': 0.01, 'lazy_proj_freq': 10, 'batch_size': 32, 'momentum': 0.95, 'orthog_method': 'newton_schulz', 'variance_reduction': True, 'cautious_wd': True, 'name': 'G03_fast_quality'},
        {'lr': 0.003, 'lazy_proj_freq': 5, 'batch_size': 32, 'momentum': 0.95, 'orthog_method': 'polar_express', 'variance_reduction': True, 'cautious_wd': True, 'name': 'G04_pe_quality'},
        {'lr': 0.008, 'lazy_proj_freq': 8, 'batch_size': 48, 'momentum': 0.94, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G05_balanced'},
        {'lr': 0.012, 'lazy_proj_freq': 12, 'batch_size': 64, 'momentum': 0.92, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G06_aggressive'},
        {'lr': 0.005, 'lazy_proj_freq': 7, 'batch_size': 32, 'momentum': 0.96, 'orthog_method': 'newton_schulz', 'variance_reduction': True, 'cautious_wd': False, 'name': 'G07_vr_only'},
        {'lr': 0.005, 'lazy_proj_freq': 7, 'batch_size': 32, 'momentum': 0.96, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': True, 'name': 'G08_cwd_only'},
        {'lr': 0.008, 'lazy_proj_freq': 10, 'batch_size': 64, 'momentum': 0.95, 'orthog_method': 'polar_express', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G09_pe_fast'},
        {'lr': 0.015, 'lazy_proj_freq': 15, 'batch_size': 64, 'momentum': 0.90, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G10_ultra_fast'},
        {'lr': 0.003, 'lazy_proj_freq': 3, 'batch_size': 32, 'momentum': 0.97, 'orthog_method': 'newton_schulz', 'variance_reduction': True, 'cautious_wd': True, 'name': 'G11_conservative'},
        {'lr': 0.007, 'lazy_proj_freq': 7, 'batch_size': 48, 'momentum': 0.95, 'orthog_method': 'newton_schulz', 'variance_reduction': True, 'cautious_wd': True, 'name': 'G12_mid_quality'},
        {'lr': 0.01, 'lazy_proj_freq': 10, 'batch_size': 48, 'momentum': 0.93, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': True, 'name': 'G13_cwd_fast'},
        {'lr': 0.006, 'lazy_proj_freq': 6, 'batch_size': 40, 'momentum': 0.95, 'orthog_method': 'newton_schulz', 'variance_reduction': True, 'cautious_wd': False, 'name': 'G14_vr_mid'},
        {'lr': 0.009, 'lazy_proj_freq': 9, 'batch_size': 56, 'momentum': 0.94, 'orthog_method': 'newton_schulz', 'variance_reduction': False, 'cautious_wd': False, 'name': 'G15_balanced2'},
        {'lr': 0.004, 'lazy_proj_freq': 8, 'batch_size': 32, 'momentum': 0.96, 'orthog_method': 'polar_express', 'variance_reduction': True, 'cautious_wd': True, 'name': 'G16_pe_stable'},
    ]

    for g_config in group_g_configs:
        config = {**BASE_CONFIG, **g_config}
        configs.append(config)

    return configs


def run_experiment(config, output_file, stage_num):
    """Run single experiment with timeout and early stopping."""
    print(f"\n{'='*80}")
    print(f"[Stage {stage_num}] Running: {config['name']}")
    print(f"{'='*80}")
    print(f"  LR: {config['lr']:.4f}, Batch: {config['batch_size']}, Lazy: {config['lazy_proj_freq']}")
    print(f"  Orthog: {config['orthog_method']}, VR: {config['variance_reduction']}, CWD: {config['cautious_wd']}")
    print(f"  Momentum: {config['momentum']:.3f}")

    cmd = [
        'uv', 'run', 'python3', 'train_architectural.py',
        '--name', config['name'],
        '--optimizer', config['optimizer'],
        '--lr', str(config['lr']),
        '--momentum', str(config['momentum']),
        '--orthog-method', config['orthog_method'],
        '--lazy-proj-freq', str(config['lazy_proj_freq']),
        '--batch-size', str(config['batch_size']),
        '--steps', str(MAX_STEPS),
        '--target-loss', str(TARGET_LOSS),
        '--val-freq', str(VALIDATION_FREQ),
        '--alpha', str(config['alpha']),
        '--n-layer', str(config['n_layer']),
        '--n-embd', str(config['n_embd']),
        '--output', output_file
    ]

    if config['gated_residuals']:
        cmd.append('--gated-residuals')
    if config['variance_reduction']:
        cmd.append('--variance-reduction')
    if config['cautious_wd']:
        cmd.append('--cautious-wd')

    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=TIMEOUT_SECONDS, check=True)
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s (hit {TIMEOUT_SECONDS}s limit)")
        return False, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ FAILED with return code {e.returncode} after {elapsed:.1f}s")
        return False, elapsed


def analyze_stage1(results_file):
    """Analyze Stage 1 results and identify optimal parameter region."""
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print(f"\n{'='*80}")
    print("STAGE 1 ANALYSIS")
    print(f"{'='*80}")
    print(f"Total experiments: {len(results)}")

    # Filter successful runs (reached <5.0)
    successful = [r for r in results if r.get('reached_target', False)]
    failed = [r for r in results if not r.get('reached_target', False)]

    print(f"  Reached target (<{TARGET_LOSS}): {len(successful)}")
    print(f"  Failed to reach target: {len(failed)}")

    if len(successful) < 5:
        print(f"\n⚠ WARNING: Only {len(successful)} configs reached target!")
        print("  May need to adjust target loss or extend max steps.")
        if len(successful) == 0:
            print("\n✗ NO CONFIGS REACHED TARGET!")
            print("  Falling back to analysis of best performers...")
            # Fallback: sort by validation loss
            results_sorted = sorted(results, key=lambda x: x.get('val_loss', 999))
            top5 = results_sorted[:5]
        else:
            # Sort by time to target
            successful.sort(key=lambda x: x['time_to_target'])
            top5 = successful[:5]
    else:
        # Sort by time to target
        successful.sort(key=lambda x: x['time_to_target'])
        top5 = successful[:5]

    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Name':<20} {'Time (s)':<12} {'Steps':<8} {'Val Loss':<10} {'Reached'}")
    print("-" * 80)

    for i, config in enumerate(top5, 1):
        time_str = f"{config['time_to_target']:.1f}" if config.get('reached_target') else "TIMEOUT"
        steps = config.get('final_step', config.get('steps', 0))
        val_loss = config.get('val_loss', 999)
        reached = "✓" if config.get('reached_target') else "✗"
        print(f"{i:<6} {config['name']:<20} {time_str:<12} {steps:<8} {val_loss:<10.4f} {reached}")

    # Analyze parameter ranges
    print(f"\n{'='*80}")
    print("OPTIMAL PARAMETER RANGES (from top 5)")
    print(f"{'='*80}")

    optimal_region = {}

    # Extract parameters from top 5
    for param in ['lr', 'momentum', 'lazy_proj_freq', 'batch_size']:
        values = [c.get(param) for c in top5 if param in c]
        if values:
            optimal_region[param] = (min(values), max(values))
            print(f"  {param}: [{min(values)}, {max(values)}]")

    # Check if orthog_method is consistent
    orthog_methods = [c.get('orthog_method') for c in top5]
    orthog_counts = {m: orthog_methods.count(m) for m in set(orthog_methods)}
    print(f"\n  Orthog methods in top 5:")
    for method, count in orthog_counts.items():
        print(f"    {method}: {count}/5")

    # Check VR/CWD usage
    vr_count = sum(1 for c in top5 if c.get('variance_reduction'))
    cwd_count = sum(1 for c in top5 if c.get('cautious_wd'))
    print(f"\n  Advanced features in top 5:")
    print(f"    Variance Reduction: {vr_count}/5")
    print(f"    Cautious WD: {cwd_count}/5")

    return optimal_region, top5, successful


def generate_stage2_grid(optimal_region, top5):
    """Generate high-resolution Stage 2 grid around optimal region."""
    print(f"\n{'='*80}")
    print("GENERATING STAGE 2 GRID")
    print(f"{'='*80}")

    configs = []

    # Lock parameters that are consistent across top 5
    orthog_methods = [c.get('orthog_method') for c in top5]
    if orthog_methods.count(orthog_methods[0]) >= 4:
        # Lock if 4/5 use same method
        locked_orthog = orthog_methods[0]
        print(f"  Locked orthog_method: {locked_orthog} (4+ of top 5)")
    else:
        locked_orthog = None
        print(f"  Orthog_method varies - will test both")

    # Determine if VR/CWD should be tested
    vr_count = sum(1 for c in top5 if c.get('variance_reduction'))
    cwd_count = sum(1 for c in top5 if c.get('cautious_wd'))

    # Create fine grid around optimal ranges
    lr_range = optimal_region.get('lr', (0.003, 0.01))
    lr_values = np.linspace(lr_range[0], lr_range[1], 6)

    momentum_range = optimal_region.get('momentum', (0.93, 0.97))
    momentum_values = np.linspace(momentum_range[0], momentum_range[1], 5)

    lazy_range = optimal_region.get('lazy_proj_freq', (7, 12))
    lazy_values = np.linspace(lazy_range[0], lazy_range[1], 5, dtype=int)

    batch_range = optimal_region.get('batch_size', (32, 64))
    batch_values = np.linspace(batch_range[0], batch_range[1], 5, dtype=int)

    print(f"\n  Parameter grids:")
    print(f"    LR: {lr_values}")
    print(f"    Momentum: {momentum_values}")
    print(f"    Lazy freq: {lazy_values}")
    print(f"    Batch size: {batch_values}")

    # Use fractional factorial sampling (not full grid)
    # Strategy: Systematic sampling to cover parameter space efficiently

    # Sample 36 configurations using Latin hypercube-like approach
    np.random.seed(42)  # For reproducibility

    for i in range(36):
        # Rotate through parameter combinations systematically
        lr = lr_values[i % len(lr_values)]
        momentum = momentum_values[(i // len(lr_values)) % len(momentum_values)]
        lazy = lazy_values[(i // (len(lr_values) * len(momentum_values))) % len(lazy_values)]
        batch = batch_values[(i // 6) % len(batch_values)]

        config = {
            **BASE_CONFIG,
            'lr': float(lr),
            'momentum': float(momentum),
            'lazy_proj_freq': int(lazy),
            'batch_size': int(batch),
            'orthog_method': locked_orthog if locked_orthog else ('newton_schulz' if i < 18 else 'polar_express'),
            'variance_reduction': vr_count >= 3 and (i % 4 == 0),  # Test if popular
            'cautious_wd': cwd_count >= 3 and (i % 4 == 1),
            'name': f'S2_{i+1:02d}'
        }
        configs.append(config)

    print(f"\n  Generated {len(configs)} Stage 2 configurations")
    return configs


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage1_file = f'time_to_target_stage1_{timestamp}.jsonl'
    stage2_file = f'time_to_target_stage2_{timestamp}.jsonl'
    summary_file = f'time_to_target_summary_{timestamp}.json'

    print("="*80)
    print("TWO-STAGE HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"Objective: Minimize wall-clock time to reach val_loss < {TARGET_LOSS}")
    print(f"Model: {BASE_CONFIG['n_layer']}L × {BASE_CONFIG['n_embd']}D (155M params)")
    print(f"Max steps: {MAX_STEPS}, Validation freq: {VALIDATION_FREQ}")
    print(f"Timeout per experiment: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS/60:.1f} min)")
    print("="*80)

    # Stage 1: Broad Sweep
    print(f"\n{'='*80}")
    print("STAGE 1: BROAD SWEEP (48 CONFIGURATIONS)")
    print(f"{'='*80}")

    stage1_configs = create_stage1_configs()
    print(f"Total configurations: {len(stage1_configs)}")
    print(f"Estimated time: {len(stage1_configs) * 3 / 60:.1f} - {len(stage1_configs) * 5 / 60:.1f} hours")

    # Clear previous results
    if Path(stage1_file).exists():
        Path(stage1_file).unlink()

    stage1_start = time.time()
    stage1_successful = 0
    stage1_failed = 0

    for i, config in enumerate(stage1_configs, 1):
        print(f"\n[{i}/{len(stage1_configs)}]", end=' ')
        success, elapsed = run_experiment(config, stage1_file, 1)
        if success:
            stage1_successful += 1
        else:
            stage1_failed += 1

    stage1_elapsed = time.time() - stage1_start

    print(f"\n{'='*80}")
    print("STAGE 1 COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {stage1_elapsed/60:.1f} minutes")
    print(f"Successful: {stage1_successful}/{len(stage1_configs)}")
    print(f"Failed: {stage1_failed}/{len(stage1_configs)}")

    # Analyze Stage 1
    optimal_region, top5, successful_configs = analyze_stage1(stage1_file)

    if len(successful_configs) == 0:
        print("\n✗ No configurations reached target. Stopping sweep.")
        print("  Consider:")
        print("  - Increasing max steps")
        print("  - Relaxing target loss threshold")
        print("  - Checking if target <5.0 is achievable with this model size")
        return 1

    # Stage 2: High-Resolution Sweep
    print(f"\n{'='*80}")
    print("STAGE 2: HIGH-RESOLUTION SWEEP")
    print(f"{'='*80}")

    stage2_configs = generate_stage2_grid(optimal_region, top5)
    print(f"Total configurations: {len(stage2_configs)}")
    print(f"Estimated time: {len(stage2_configs) * 2 / 60:.1f} hours")

    # Clear previous results
    if Path(stage2_file).exists():
        Path(stage2_file).unlink()

    stage2_start = time.time()
    stage2_successful = 0
    stage2_failed = 0

    for i, config in enumerate(stage2_configs, 1):
        print(f"\n[{i}/{len(stage2_configs)}]", end=' ')
        success, elapsed = run_experiment(config, stage2_file, 2)
        if success:
            stage2_successful += 1
        else:
            stage2_failed += 1

    stage2_elapsed = time.time() - stage2_start

    print(f"\n{'='*80}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {stage2_elapsed/60:.1f} minutes")
    print(f"Successful: {stage2_successful}/{len(stage2_configs)}")
    print(f"Failed: {stage2_failed}/{len(stage2_configs)}")

    # Final Analysis
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    # Load all results
    all_results = []
    for filepath in [stage1_file, stage2_file]:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                all_results.extend([json.loads(line) for line in f])

    # Filter and sort
    successful_all = [r for r in all_results if r.get('reached_target', False)]
    successful_all.sort(key=lambda x: x['time_to_target'])

    if len(successful_all) > 0:
        print(f"\n🎯 OPTIMAL CONFIGURATION")
        print("="*80)
        best = successful_all[0]
        print(f"Name: {best['name']}")
        print(f"Time to <{TARGET_LOSS}: {best['time_to_target']:.1f}s")
        print(f"Steps to target: {best['final_step']}")
        print(f"Final val loss: {best['val_loss']:.4f}")
        print(f"\nHyperparameters:")
        print(f"  LR: {best['lr']}")
        print(f"  Momentum: {best.get('momentum', 'N/A')}")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Lazy proj freq: {best['lazy_proj_freq']}")
        print(f"  Orthog method: {best.get('orthog_method', 'N/A')}")
        print(f"  Variance reduction: {best.get('variance_reduction', False)}")
        print(f"  Cautious WD: {best.get('cautious_wd', False)}")

        print(f"\n📊 TOP 10 CONFIGURATIONS")
        print("="*80)
        print(f"{'Rank':<6} {'Name':<20} {'Time (s)':<12} {'Steps':<8} {'Val Loss':<10}")
        print("-" * 80)
        for i, config in enumerate(successful_all[:10], 1):
            print(f"{i:<6} {config['name']:<20} {config['time_to_target']:<12.1f} {config['final_step']:<8} {config['val_loss']:<10.4f}")
    else:
        print("\n✗ NO CONFIGURATIONS REACHED TARGET!")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_loss': TARGET_LOSS,
        'max_steps': MAX_STEPS,
        'stage1': {
            'total_configs': len(stage1_configs),
            'successful': stage1_successful,
            'failed': stage1_failed,
            'time_minutes': stage1_elapsed / 60,
        },
        'stage2': {
            'total_configs': len(stage2_configs),
            'successful': stage2_successful,
            'failed': stage2_failed,
            'time_minutes': stage2_elapsed / 60,
        },
        'total_time_minutes': (stage1_elapsed + stage2_elapsed) / 60,
        'best_config': successful_all[0] if len(successful_all) > 0 else None,
        'top_10': successful_all[:10] if len(successful_all) >= 10 else successful_all,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved:")
    print(f"  Stage 1: {stage1_file}")
    print(f"  Stage 2: {stage2_file}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*80}")

    total_time = stage1_elapsed + stage2_elapsed
    print(f"\nTotal sweep time: {total_time/3600:.2f} hours")

    return 0


if __name__ == '__main__':
    exit(main())
