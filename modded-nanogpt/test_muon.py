#!/usr/bin/env python3
"""
Quick test of SimpleMuon optimizer in train_architectural.py
Runs a short experiment to verify Muon works correctly.
"""

import subprocess
import json
from pathlib import Path

def run_quick_test(optimizer, lr, name):
    """Run a quick 50-step test with given optimizer."""
    print(f"\n{'='*80}")
    print(f"Testing {optimizer.upper()} optimizer (lr={lr})")
    print(f"{'='*80}")

    cmd = [
        'python3', 'train_architectural.py',
        '--name', name,
        '--optimizer', optimizer,
        '--lr', str(lr),
        '--alpha', '0.15',
        '--gated-residuals',
        '--batch-size', '32',
        '--lazy-proj-freq', '7',
        '--steps', '50',  # Short test
        '--output', 'muon_test_results.jsonl'
    ]

    if optimizer == 'muon':
        cmd.extend(['--momentum', '0.95'])

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ {optimizer.upper()} test completed successfully")
    else:
        print(f"\n✗ {optimizer.upper()} test failed with return code {result.returncode}")
        return False

    return True


def main():
    print("\n" + "="*80)
    print("SimpleMuon Optimizer Verification Test")
    print("="*80)
    print("Goal: Verify SimpleMuon runs without errors")
    print("Config: Gated nGPT, alpha=0.15, batch=32, 50 steps")
    print("="*80)

    # Clear previous results
    results_file = Path('muon_test_results.jsonl')
    if results_file.exists():
        results_file.unlink()

    # Test 1: Adam baseline (should work - already tested)
    print("\n[Test 1/2] Running Adam baseline...")
    success_adam = run_quick_test('adam', lr=0.0005, name='test_adam_baseline')

    # Test 2: SimpleMuon
    print("\n[Test 2/2] Running SimpleMuon...")
    success_muon = run_quick_test('muon', lr=0.02, name='test_muon')

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Adam baseline:   {'✓ PASS' if success_adam else '✗ FAIL'}")
    print(f"SimpleMuon:      {'✓ PASS' if success_muon else '✗ FAIL'}")

    if success_adam and success_muon:
        print(f"\n{'='*80}")
        print("✓ ALL TESTS PASSED - SimpleMuon is ready for experiments!")
        print(f"{'='*80}")

        # Load and compare results
        with open('muon_test_results.jsonl', 'r') as f:
            results = [json.loads(line) for line in f]

        if len(results) >= 2:
            adam_result = results[0]
            muon_result = results[1]

            print("\nQuick Comparison (50 steps only - not conclusive):")
            print(f"  Adam val_loss:  {adam_result.get('val_loss', 'N/A'):.4f}")
            print(f"  Muon val_loss:  {muon_result.get('val_loss', 'N/A'):.4f}")
            print("\nNote: This is just a verification test. Use muon_lr_sweep.py for real experiments.")

        return 0
    else:
        print(f"\n{'='*80}")
        print("✗ TESTS FAILED - Check errors above")
        print(f"{'='*80}")
        return 1


if __name__ == '__main__':
    exit(main())
