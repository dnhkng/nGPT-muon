# Training Comparison: nGPT vs Original modded-nanogpt

**Test Date:** 2025-12-25
**Hardware:** MacBook (CPU)
**Dataset:** Shakespeare (~302K training tokens, ~36K validation tokens)
**Duration:** 200 training steps

---

## Test Configuration

Both models used identical configurations for fair comparison:

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Block Size (Context) | 64 tokens |
| Batch Size | 4 |
| Training Steps | 200 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| Device | CPU |

**Total Parameters:** ~13.7M (nearly identical)

---

## 📊 Results Summary

### Original modded-nanogpt (Pre-Norm Transformer)

| Metric | Value |
|--------|-------|
| Initial Loss (step 0) | 10.8240 |
| Final Loss (step 190) | 5.8913 |
| Validation Loss | 5.9814 |
| **Total Decrease** | **-4.9327** |
| **Reduction %** | **45.6%** |

### nGPT (Normalized Transformer)

| Metric | Value |
|--------|-------|
| Initial Loss (step 0) | 10.8251 |
| Final Loss (step 190) | 10.6734 |
| Validation Loss | 10.6705 |
| **Total Decrease** | **-0.1517** |
| **Reduction %** | **1.4%** |

---

## 📈 Training Curves

### Original modded-nanogpt Loss Progression

```
Step   0: 10.8240 ███████████████████████████████████████
Step  10:  9.4205 ██████████████████████████████████
Step  20:  7.9586 ████████████████████████████
Step  30:  7.2310 █████████████████████████
Step  40:  6.8071 ███████████████████████
Step  50:  6.7882 ███████████████████████
Step  60:  6.2750 █████████████████████
Step  70:  6.1381 ████████████████████
Step  80:  6.2999 █████████████████████
Step  90:  6.0254 ████████████████████
Step 100:  6.1265 ████████████████████
Step 110:  6.2026 ████████████████████
Step 120:  6.2367 █████████████████████
Step 130:  6.4144 █████████████████████
Step 140:  6.2258 █████████████████████
Step 150:  5.6810 ███████████████████
Step 160:  6.2951 █████████████████████
Step 170:  6.6462 ██████████████████████
Step 180:  6.0737 ████████████████████
Step 190:  5.8913 ███████████████████
```

**Characteristics:**
- ✅ Rapid initial decrease (10.8 → 7.2 in 30 steps)
- ✅ Continued learning throughout training
- ✅ Reached ~5.9 validation loss
- ✅ Some oscillation but overall downward trend

### nGPT Loss Progression

```
Step   0: 10.8251 ███████████████████████████████████████
Step  10: 10.8165 ███████████████████████████████████████
Step  20: 10.8073 ███████████████████████████████████████
Step  30: 10.8047 ███████████████████████████████████████
Step  40: 10.7970 ███████████████████████████████████████
Step  50: 10.7951 ███████████████████████████████████████
Step  60: 10.7897 ███████████████████████████████████████
Step  70: 10.7842 ███████████████████████████████████████
Step  80: 10.7749 ███████████████████████████████████████
Step  90: 10.7657 ███████████████████████████████████████
Step 100: 10.7613 ██████████████████████████████████████
Step 110: 10.7481 ██████████████████████████████████████
Step 120: 10.7449 ██████████████████████████████████████
Step 130: 10.7329 ██████████████████████████████████████
Step 140: 10.7292 ██████████████████████████████████████
Step 150: 10.7178 ██████████████████████████████████████
Step 160: 10.7158 ██████████████████████████████████████
Step 170: 10.6891 ██████████████████████████████████████
Step 180: 10.6837 ██████████████████████████████████████
Step 190: 10.6734 ██████████████████████████████████████
```

**Characteristics:**
- ⚠️ Very slow decrease (10.8 → 10.7 in 200 steps)
- ⚠️ Loss barely moving
- ⚠️ Validation loss still very high (10.67)
- ✅ Training is stable (no NaN/divergence)
- ✅ Perfect norm enforcement (all norms = 1.000)

---

## 🔬 Analysis

### Why is nGPT Learning So Slowly?

The nGPT implementation is **mathematically correct** (all norms = 1.000, no NaN issues) but has **severe learning dynamics problems**. Here are the likely causes:

#### 1. **Alpha Parameters Too Small** (PRIMARY SUSPECT)

**Current:** Alpha initialized to 0.05

```python
self.alpha_attn = nn.Parameter(torch.full((dim,), 0.05))
self.alpha_mlp = nn.Parameter(torch.full((dim,), 0.05))
```

**Problem:** With residual `x = normalize(x + alpha * layer(x))`, when alpha = 0.05, the layer output contributes only 5% to the residual. This severely limits information flow.

**Expected Behavior:** Alpha should grow during training, but with such slow learning, it may not have time to increase.

**Fix:** Try alpha = 0.1, 0.2, or even 0.5 for faster initial learning.

#### 2. **Logit Scale Too Small**

**Current:** Logit scale initialized to `1/sqrt(128) ≈ 0.089`

```python
self.logit_scale = nn.Parameter(torch.tensor(1.0 / (model_dim ** 0.5)))
```

**Problem:** Cosine similarity is bounded [-1, 1]. With scale ~ 0.089, logits are in range [-0.089, 0.089], which is very narrow for softmax. The gradient might be too small.

**Comparison:** Original uses tanh softcapping which maps to [0, 30], much larger dynamic range.

**Fix:** Initialize logit_scale to 10 or 20 for stronger gradients.

#### 3. **No Cosine Similarity in Attention**

**Current:** nGPT removes QK normalization entirely

```python
# nGPT: x already normalized, weights normalized - QK norm redundant
# q, k = norm(q), norm(k) # REMOVED
```

**Problem:** While mathematically the inputs and weights are normalized, removing the explicit QK norm might reduce the effective attention scores. Original explicitly normalizes Q and K, which may provide better attention dynamics.

**Comparison:** Original applies `norm(q)` and `norm(k)`, which forces queries and keys to be unit vectors, maximizing the dynamic range of attention scores.

**Fix:** Consider re-adding QK normalization even in nGPT for better attention signal.

#### 4. **Learning Rate Mismatch**

**Current:** LR = 1e-3 for all parameters

**Problem:** nGPT has fundamentally different geometry. Parameters on the unit sphere may need different learning rates. The alpha parameters and logit_scale should probably have higher learning rates.

**Fix:** Increase LR for alpha and logit_scale to 1e-2 or higher.

#### 5. **Embedding Normalization**

**Current:** Embeddings are normalized immediately after lookup

```python
x = normalize_ngpt(self.embed(input_seq), dim=-1)
```

**Problem:** For a vocabulary of 50K tokens with 128-dim embeddings, forcing all embeddings to unit norm might remove important signal about token frequency or importance.

**Comparison:** Original embeddings can have varying magnitudes.

**Fix:** Consider not normalizing embeddings, or using a softer normalization.

---

## 🎯 Verification Status

### Original modded-nanogpt: ✅ WORKS AS EXPECTED

- Fast convergence
- Loss decreased 45.6% in 200 steps
- Validation loss ~6.0 (reasonable for small model on Shakespeare)
- Standard pre-norm transformer working correctly

### nGPT: ⚠️ CORRECT BUT INEFFECTIVE

- Perfect norm enforcement (1.000 ± 0.000)
- No numerical instabilities
- Architecture implemented correctly per design spec
- **BUT:** Learning is ~32x slower than original (0.15 vs 4.93 loss decrease)

---

## 🔧 Recommended Fixes

### Priority 1: Increase Alpha Initialization

```python
# Current
self.alpha_attn = nn.Parameter(torch.full((dim,), 0.05))

# Recommended
self.alpha_attn = nn.Parameter(torch.full((dim,), 0.2))  # 4x larger
```

**Rationale:** 0.05 is too conservative. Start with 20% contribution from each layer.

### Priority 2: Increase Logit Scale

```python
# Current
self.logit_scale = nn.Parameter(torch.tensor(1.0 / (model_dim ** 0.5)))

# Recommended
self.logit_scale = nn.Parameter(torch.tensor(10.0))  # Much larger
```

**Rationale:** Cosine similarity needs larger scaling for effective gradients.

### Priority 3: Restore QK Normalization (Optional)

```python
# In attention forward:
q, k = normalize_ngpt(q), normalize_ngpt(k)  # Restore explicit QK norm
```

**Rationale:** May improve attention dynamics even with normalized inputs.

### Priority 4: Increase Learning Rates for Alpha/Scale

```python
# In optimizer setup
adam_labels = [..., 'alpha', 'logit_scale']
# Set higher LR for these in parameter groups
```

**Rationale:** These parameters need to adapt quickly.

---

## 📋 Next Steps

1. **Immediate:** Test with larger alpha (0.2) and logit_scale (10.0)
2. **Verify:** Re-run 200-step test and check if loss decreases faster
3. **Tune:** Experiment with alpha in [0.1, 0.5] range
4. **Research:** Check nGPT paper for recommended alpha initialization
5. **Scale:** Once Mac test shows learning, move to H100 for full-scale training

---

## 📁 Files

**Test Scripts:**
- `modded-nanogpt/train_mac.py` - nGPT test script
- `modded-nanogpt-original/train_mac_original.py` - Original test script

**Logs:**
- `modded-nanogpt/ngpt_test.log` - nGPT training log
- `modded-nanogpt-original/original_test.log` - Original training log

**Datasets:**
- `train.bin` - 301,966 tokens
- `val.bin` - 36,059 tokens

---

## 💡 Key Insights

### What Works in nGPT:
- ✅ Unit norm enforcement (perfect 1.000)
- ✅ Weight projection hook (no exploding/collapsing norms)
- ✅ Numerical stability (no NaN/Inf)
- ✅ Float32 precision for norms prevents collapse

### What Doesn't Work in nGPT:
- ❌ Learning rate too slow (32x slower than original)
- ❌ Alpha too small (0.05 is too conservative)
- ❌ Logit scale too small (needs 10-100x increase)
- ❌ Possibly needs QK norm even with normalized inputs

### Architecture Comparison:

| Feature | Original | nGPT | Impact |
|---------|----------|------|--------|
| Residual | `x + layer(norm(x))` | `norm(x + 0.05*layer(x))` | **Alpha = 0.05 is bottleneck** |
| QK Norm | ✅ Explicit | ❌ Removed | May hurt attention |
| Logits | `30*sigmoid(W@x/7.5)` | `0.089*cosine(W,x)` | **Scale too small** |
| Embedding Norm | ❌ No | ✅ Yes | May remove signal |

---

## 🎓 Conclusion

**The nGPT implementation is architecturally sound but has hyperparameter tuning issues.**

The original modded-nanogpt achieves **32x faster learning** (4.93 vs 0.15 loss decrease) with the same model size and dataset. This is not due to bugs in the nGPT implementation - the unit norm constraints are perfectly enforced - but rather due to overly conservative initialization of alpha and logit_scale parameters.

**Recommended action:** Increase alpha from 0.05 to 0.2 and logit_scale from ~0.089 to 10.0, then re-test.

---

**Report Generated:** 2025-12-25
**Test Environment:** MacBook CPU
**nGPT Version:** Phase 1 (Mac Testing)
