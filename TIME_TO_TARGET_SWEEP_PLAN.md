# Plan: Time-to-Target Hyperparameter Sweep for nGPT

## Objective
Minimize wall-clock time to reach validation loss < 5.0 through systematic two-stage hyperparameter sweeping. Primary metric: seconds to target, NOT iterations or final loss quality.

## Background

### Current State
- **Best nGPT result:** Muon + Polar Express + VR + CWD → 6.35 val_loss in 400 steps (~50s)
- **Target:** < 5.0 val_loss (significantly lower, requires extended training)
- **Model scale:** 11 layers × 768 dim = 155M params (modded-nanogpt size - REQUIRED)
- **Hardware:** H100 GPU

### Key Findings from Exploration
1. **Speed factors (biggest impact first):**
   - Orthogonalization: Newton-Schulz (32-35K tok/s) vs Polar Express (26K tok/s) - 20-30% difference
   - Variance Reduction: -7% throughput but may improve convergence
   - Learning Rate: LR 0.01 is 12% faster than LR 0.001 (surprising!)
   - Lazy projection freq: every 7 steps is balanced

2. **Critical insight:** VR+CWD may be better by iterations but slower by wall-clock time (needs testing)

3. **Infrastructure gap:** No early stopping implemented - all experiments run fixed steps

## Implementation Strategy

### Phase 1: Add Early Stopping (CRITICAL PREREQUISITE)

**Current problem:** All training runs for fixed steps, no validation during training

**Solution:** Modify training loop to validate every N steps and stop when loss < 5.0

**Files to modify:**
- `/home/grace/Projects/nGPT-muon/modded-nanogpt/train_architectural.py`

**Changes needed:**

1. **Add command-line arguments** (after line 47):
```python
parser.add_argument('--target-loss', type=float, default=None, help='Target validation loss for early stopping')
parser.add_argument('--val-freq', type=int, default=50, help='Validation frequency (steps)')
```

2. **Add validation inside training loop** (around line 720, inside the `for step in range(args.steps):` loop):
```python
# Periodic validation with early stopping
if args.target_loss is not None and step > 0 and step % args.val_freq == 0:
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):  # 20 batches for stable estimate
            x, y = get_batch(val_data, config['block_size'], args.batch_size, device)
            _, loss = model(x, y)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    model.train()

    print(f"Step {step}: train={losses[-1]:.4f}, val={val_loss:.4f}")

    if val_loss < args.target_loss:
        time_to_target = time.time() - t0
        reached_target = True
        print(f"\nTARGET REACHED: {val_loss:.4f} < {args.target_loss} at step {step}")
        print(f"Time to target: {time_to_target:.1f}s")
        break
```

3. **Track results** (add to result dict around line 750):
```python
result.update({
    'target_loss': args.target_loss,
    'reached_target': reached_target,
    'time_to_target': time_to_target if reached_target else float('inf'),
    'final_step': step,
})
```

**Parameters:**
- `--val-freq 50`: Validate every 50 steps (balance between overhead and precision)
- `--steps 5000`: Max steps timeout (prevents infinite runs)
- `--target-loss 5.0`: Stop when validation loss drops below 5.0

---

### Phase 2: Stage 1 - Broad Sweep (48 Configurations)

**Objective:** Cast wide net to identify promising hyperparameter regions

**Baseline configuration:**
- Optimizer: Muon with Polar Express
- Model: 11L × 768D (155M params - modded-nanogpt scale)
- Architecture: Gated residuals, alpha=0.15
- VR/CWD: Parameters to sweep (not fixed)

**Parameter groups to test:**

**Group A: Orthogonalization Method (8 configs)**
Test both orthog methods with different base configs:
- `newton_schulz` vs `polar_express`
- Combined with 4 base configs: (lr=0.003, lazy=7), (lr=0.01, lazy=10), (lr=0.003 + VR + CWD), (lr=0.01, batch=64)

**Group B: VR/CWD Combinations (4 configs)**
With Newton-Schulz orthog:
- No VR/CWD (baseline)
- VR only
- CWD only
- VR + CWD

**Group C: Lazy Projection Frequency (6 configs)**
- Values: [3, 5, 7, 10, 15, 20]
- Use best orthog method from Group A

**Group D: Learning Rate (6 configs)**
- Values: [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]
- Fixed: best orthog, lazy=7, batch=32

**Group E: Batch Size (4 configs)**
- Values: [16, 32, 64, 128]
- Adjust LR proportionally (linear scaling rule)

**Group F: Momentum (4 configs)**
- Values: [0.85, 0.90, 0.95, 0.98]

**Group G: Promising Combinations (16 configs)**
Hand-picked based on domain knowledge:
- newton_schulz + lr=0.01 + lazy=10 + batch=64
- newton_schulz + lr=0.01 + lazy=7 + VR + CWD
- polar_express + lr=0.003 + lazy=5 + VR + CWD
- ... (13 more strategic combinations)

**Total Stage 1: 48 configurations**

**Expected runtime:** ~2.4 hours (avg 3 min per config)
- Fast configs: 60-120s to reach <5.0
- Slow configs: 180-300s
- Failed configs (hit 5000 step limit): 300-360s

---

### Phase 3: Stage 2 - High-Resolution Sweep (36 Configurations)

**Objective:** Fine-tune around optimal region identified in Stage 1

**Trigger:** After Stage 1 completes, analyze top 5 configurations

**Process:**
1. **Identify optimal parameter ranges** from top 5 Stage 1 configs
2. **Lock consistently optimal parameters** (e.g., if all top 5 use newton_schulz, lock it)
3. **Create fine grid around variable parameters** that show sensitivity

**Example Stage 2 Grid** (assuming Stage 1 finds):
- orthog_method: newton_schulz (LOCKED)
- lr: varies between 0.008-0.012 in top 5 configs
- momentum: varies between 0.93-0.97
- lazy_freq: varies between 8-12
- batch_size: varies between 48-80

**High-resolution sampling (36 configs):**
- LR: [0.008, 0.009, 0.01, 0.011, 0.012, 0.013] - 6 values
- Momentum: [0.93, 0.94, 0.95, 0.96, 0.97] - 5 values
- Lazy freq: [8, 9, 10, 11, 12] - 5 values
- Batch size: [48, 56, 64, 72, 80] - 5 values

**Sampling strategy:** Fractional factorial (not full grid) - ~36 strategic combinations

**Expected runtime:** ~1.2 hours (avg 2 min per config, near-optimal region)

---

### Phase 4: Create Sweep Orchestration Script

**File to create:** `/home/grace/Projects/nGPT-muon/modded-nanogpt/time_to_target_sweep.py`

**Structure:**
```python
#!/usr/bin/env python3
"""Two-stage hyperparameter sweep for time-to-target optimization"""

TARGET_LOSS = 5.0
MAX_STEPS = 5000
VALIDATION_FREQ = 50

# Stage 1: 48 configs defined
STAGE1_CONFIGS = [...]

def run_experiment(config, output_file):
    """Run single experiment with timeout and early stopping"""
    cmd = ['python', 'train_architectural.py', ...]
    # Launch with subprocess, handle timeout

def analyze_stage1(results_file):
    """Analyze Stage 1, identify top 5, extract optimal region"""
    # Sort by time_to_target
    # Identify parameter ranges from top 5

def generate_stage2_grid(optimal_region):
    """Create high-resolution grid around optimal region"""
    # Fractional factorial sampling

def main():
    # Run Stage 1 (48 configs)
    # Analyze results
    # Generate Stage 2 grid (36 configs)
    # Run Stage 2
    # Final analysis and report
```

**Key features:**
- Timeout handling (10 min per experiment)
- Automatic Stage 2 grid generation based on Stage 1 results
- JSONL output for easy parsing
- Summary reports after each stage

---

## Critical Files

### To Modify:
1. **`/home/grace/Projects/nGPT-muon/modded-nanogpt/train_architectural.py`** (lines 47, 720-750)
   - Add `--target-loss` and `--val-freq` arguments
   - Add validation inside training loop with early stopping
   - Track `time_to_target` and `reached_target` metrics

### To Create:
2. **`/home/grace/Projects/nGPT-muon/modded-nanogpt/time_to_target_sweep.py`**
   - Orchestrate two-stage sweep (48 + 36 configs)
   - Handle timeouts and failures
   - Auto-generate Stage 2 grid from Stage 1 results

### Reference Files:
- `/home/grace/Projects/nGPT-muon/modded-nanogpt/optimization_sweep.py` - Pattern for sweep infrastructure
- `/home/grace/Projects/nGPT-muon/modded-nanogpt/polar_express_experiments.py` - Subprocess handling pattern

---

## Expected Outcomes

### Predicted Optimal Configuration:
Based on exploration data and speed analysis:
- **orthog_method:** newton_schulz (20-30% faster than Polar Express)
- **lr:** 0.008-0.012 (high for speed)
- **momentum:** 0.94-0.96
- **lazy_proj_freq:** 8-10 (aggressive)
- **batch_size:** 64 (better GPU utilization)
- **variance_reduction:** False (throughput penalty likely not worth it)
- **cautious_wd:** False (same reasoning)

**Expected time to <5.0:** 60-90 seconds (1500-2500 steps at 25-30K tok/s)

### Key Questions to Answer:
1. Does Polar Express's quality advantage compensate for 20-30% speed penalty?
2. Is VR+CWD better by wall-clock time or just by iteration count?
3. What LR achieves fastest convergence (speed × fewer steps)?
4. Can aggressive lazy projection (freq=10-15) maintain convergence while boosting speed?

### Failure Modes:
- **No config reaches <5.0:** Target too aggressive → fallback to "time to 5.5" analysis
- **High variance in times:** GPU throttling → run configs 2-3 times, take median
- **All similar times:** Insufficient differentiation → tighten target to 4.8

---

## Timeline Summary

**Phase 1: Early Stopping Implementation** - 30-60 min
- Modify train_architectural.py
- Test with single run to verify early stopping works

**Phase 2: Stage 1 Broad Sweep** - 2.4 hours
- 48 configurations
- Identify top 5 performers

**Phase 3: Stage 2 High-Resolution Sweep** - 1.2 hours
- 36 configurations around optimal region
- Final optimization

**Phase 4: Analysis & Reporting** - 30 min
- Generate comprehensive report
- Identify production configuration

**Total Estimated Time:** ~4.5-5 hours

---

## Success Criteria

✅ **Primary:** Identify configuration with fastest wall-clock time to val_loss < 5.0

✅ **Secondary:** Understand speed/quality tradeoffs for each parameter

✅ **Deliverable:** Production-ready configuration for full-scale FineWeb100B training

---

## Next Steps

1. **Implement early stopping** in train_architectural.py
2. **Create time_to_target_sweep.py** with 48 Stage 1 configs
3. **Run Stage 1 sweep** (~2.4 hours)
4. **Analyze results** and generate Stage 2 grid
5. **Run Stage 2 sweep** (~1.2 hours)
6. **Final report** with optimal configuration
