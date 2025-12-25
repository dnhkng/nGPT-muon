# Muon Optimizer Optimization for nGPT

**Date:** 2025-12-25
**Goal:** Systematically optimize the Muon optimizer for nGPT's normalized hypersphere geometry
**Hypothesis:** Muon's orthogonal gradient descent should be particularly effective for hypersphere-constrained optimization

---

## Background

### Current State
- **Best nGPT Result:** Gated nGPT with Adam (lr=0.0005) → validation loss 6.775
- **Theory:** Muon optimizer tailored for nGPT geometry should outperform generic Adam
- **modded-nanogpt:** Uses NorMuon + DistAdam with extensive optimizations (53 world records)

### Why Muon for nGPT?
1. **Geometric Match:** Muon does orthogonal updates; nGPT weights live on unit hypersphere
2. **Projection Synergy:** nGPT already projects weights; Muon's Newton-Schulz iteration aligns naturally
3. **Proven Track Record:** modded-nanogpt achieved 53 world records via iterative Muon improvements

---

## Implementation Strategy

### Phase 1: Simplified Muon Implementation ✅ IN PROGRESS

**Approach:** Create a standalone, single-GPU Muon optimizer for rapid experimentation

**Simplified Muon Design:**
```python
class SimpleMuon(torch.optim.Optimizer):
    """
    Muon = MomentUm Orthogonalized by Newton-schulz

    For 2D weights (attn, mlp):
      1. Apply momentum: g_t = momentum * g_{t-1} + grad
      2. Orthogonalize: U = newton_schulz(g_t)
      3. Update: W = W - lr * U
      4. Project: W = normalize(W)  # Already done by nGPT

    For 1D params (biases, alphas):
      Use standard AdamW
    """
```

**Orthogonalization Methods:**
1. **Newton-Schulz (Initial):** Simple iterative method
   - `X_{k+1} = X_k * (3I - X_k^T X_k) / 2`
   - 5 iterations, no Triton kernels

2. **Polar Express (Later):** Advanced method from modded-nanogpt
   - Requires porting Triton kernels
   - Faster convergence, better stability

**Files Modified:**
- `train_architectural.py`: Add `SimpleMuon` class and `--optimizer` flag

---

### Phase 2: Muon Learning Rate Sweep

**Objective:** Find optimal Muon LR for nGPT's normalized geometry

**Experiment Design:**
```python
LR_VALUES = [0.001, 0.003, 0.01, 0.023, 0.05, 0.1]
# Note: 0.023 is modded-nanogpt's optimal LR

for lr in LR_VALUES:
    run_experiment(
        name=f'muon_lr_{lr}',
        optimizer='muon',
        lr=lr,
        alpha=0.15,  # Optimal from previous experiments
        batch_size=32,
        steps=400,
        shards=4
    )
```

**Expected Result:** Optimal LR likely differs from standard GPT due to unit norm constraint

**Runner Script:** `muon_lr_sweep.py`

---

### Phase 3: Geodesic Muon Updates

**Theory:** Exact geodesic movement along hypersphere curvature converges faster than approximate projection

**Current (Approximate):**
```python
W_new = normalize(W - lr * U)  # Linear step + projection
```

**Proposed (Exact Geodesic):**
```python
theta = lr  # Or lr * norm(U)
W_new = W * cos(theta) + U * sin(theta)  # Stays exactly on sphere
```

**Experiment Design:**
1. **Baseline:** Current projection-based update
2. **Treatment 1:** Geodesic with `theta = lr`
3. **Treatment 2:** Geodesic with `theta = lr * norm(U)`

**Implementation:** Add `--geodesic-muon` flag to `train_architectural.py`

---

### Phase 4: Advanced Optimizer Features

**Port from modded-nanogpt:**

1. **Polar Express Orthogonalization**
   - Replacement for Newton-Schulz
   - Faster, more accurate
   - Requires Triton kernels `XXT` and `ba_plus_cAA`

2. **NorMuon Variance Reduction**
   - Low-rank variance estimator (similar to Adafactor)
   - Function: `apply_normuon_variance_reduction()`
   - Adds per-feature adaptive scaling

3. **Cautious Weight Decay**
   - Gated version of decoupled weight decay
   - Only applies decay when `(grad * param) > 0`
   - Function: `cautious_wd_and_update_inplace()`

**Files to Extract From:**
- `train_gpt.py` lines 378-710: Complete NorMuon implementation

---

### Phase 5: Lazy Projection Frequency

**Hypothesis:** Projection frequency interacts with optimizer choice

**Already Tested:**
- Adam with lazy projection: optimal freq = 7

**Need to Test:**
- Muon with lazy projection
- Hypothesis: Muon may benefit from less frequent projection (orthogonal updates naturally preserve norm)

**Frequencies to Test:** [1, 3, 5, 7, 10, 15, 20]

---

## Experimental Configuration

### Common Parameters
```python
MODEL = {
    'n_layer': 11,
    'n_embd': 768,
    'n_params': 155M,
    'architecture': 'gated_ngpt',
    'alpha': 0.15,
    'lazy_proj_freq': 7  # Unless testing this parameter
}

TRAINING = {
    'dataset': 'FineWeb10B',
    'shards': 4,
    'steps': 400,
    'batch_size': 32,
    'duration': '~40 seconds'
}
```

### Success Metrics
1. **Primary:** Validation loss < 6.775 (Adam baseline)
2. **Secondary:** Training stability, tokens/sec, convergence speed

---

## Experiment Tracking

### Campaign 1: Muon Infrastructure ✅ IN PROGRESS
- [x] Read train_gpt.py to understand NorMuon
- [x] Design simplified Muon for single-GPU
- [ ] Implement SimpleMuon in train_architectural.py
- [ ] Add --optimizer flag
- [ ] Verify: Single Muon run completes successfully

### Campaign 2: LR Sweep
- [ ] Create muon_lr_sweep.py
- [ ] Run 6 experiments (LR: 0.001 → 0.1)
- [ ] Identify optimal LR
- [ ] Generate MUON_LR_SWEEP_REPORT.md

### Campaign 3: Geodesic Updates
- [ ] Implement geodesic update math
- [ ] Add --geodesic-muon flag
- [ ] Run 3 experiments (baseline, theta=lr, theta=lr*norm)
- [ ] Generate GEODESIC_MUON_REPORT.md

### Campaign 4: Advanced Features
- [ ] Extract Polar Express from train_gpt.py
- [ ] Port Triton kernels (XXT, ba_plus_cAA)
- [ ] Add NorMuon variance reduction
- [ ] Add cautious weight decay
- [ ] Test combinations systematically

### Campaign 5: Projection Frequency
- [ ] Test frequencies [1, 3, 5, 7, 10, 15, 20] with Muon
- [ ] Compare with Adam results
- [ ] Identify optimal frequency for Muon

---

## Key Code Locations

### train_gpt.py (modded-nanogpt)
- **Lines 121-165:** Triton helper functions
- **Lines 171-228:** XXT kernel (A @ A.T)
- **Lines 264-366:** ba_plus_cAA kernel
- **Lines 369-375:** Polar Express coefficients
- **Lines 378-421:** polar_express() function
- **Lines 428-433:** cautious_wd_and_update_inplace()
- **Lines 437-448:** apply_normuon_variance_reduction()
- **Lines 454-710:** NorMuon class (full implementation)
- **Lines 713-728:** project_weights_to_hypersphere()

### train_architectural.py (our experiments)
- **Lines 40-42:** Added --optimizer, --momentum flags
- **Lines 71-80:** project_weights_to_hypersphere() (already exists)
- **TODO:** Add SimpleMuon class
- **Lines 365-368:** Optimizer creation (need to update)

---

## Expected Outcomes

### Research Questions
1. Does Muon's orthogonal update structure provide advantages for nGPT's hypersphere geometry?
2. What is the optimal Muon learning rate for normalized architectures?
3. Do geodesic updates provide measurable improvements over projection?
4. Which modded-nanogpt optimizer innovations transfer to nGPT?

### Hypothesis
- **Muon beats Adam:** val_loss < 6.775
- **Optimal Muon LR:** Likely higher than standard GPT (due to unit norm constraint)
- **Geodesic advantage:** 1-3% improvement over projection
- **Advanced features:** NorMuon + Polar Express achieve best results

---

## Next Steps

1. ✅ Created comprehensive plan documentation
2. **Immediate:** Finish implementing SimpleMuon in train_architectural.py
3. Run single verification experiment with Muon
4. Create muon_lr_sweep.py and run Campaign 2
5. Implement geodesic updates for Campaign 3

---

## Notes

### Newton-Schulz vs Polar Express
- **Newton-Schulz:** Simple, 5 iterations, pure PyTorch
- **Polar Express:** Advanced, requires Triton, faster convergence
- **Strategy:** Start with Newton-Schulz, upgrade to Polar Express later

### Distributed Training
- **Current:** Single-GPU experiments for speed
- **Later:** Can add distributed support if needed
- **NorMuon:** Has advanced distributed optimizations (not needed initially)

### Lazy Projection Philosophy
- **nGPT:** Projects weights every N steps
- **Muon:** Orthogonalizes gradients every step
- **Synergy:** Both methods work with constrained geometry
- **Question:** Do they interact? Does Muon need less frequent projection?

---

**Last Updated:** 2025-12-25
**Status:** Phase 1 in progress (implementing SimpleMuon)
**Next Milestone:** Complete SimpleMuon implementation and run first test
