# Mac Testing Instructions for nGPT

This document describes how to test the nGPT (Normalized Transformer) implementation on MacBook (CPU/MPS) before scaling to H100.

## Overview

The nGPT architecture replaces standard pre-norm residuals with normalized residuals:
- **OLD:** `x = x + layer(norm(x))` with RMSNorm
- **NEW:** `x = normalize(x + alpha * layer(x))` with unit-norm constraint

This Mac test verifies the correctness of the nGPT geometry using a simplified trainer (`train_mac.py`) on the Shakespeare dataset.

## Prerequisites

- Python 3.10+
- `uv` package manager (recommended) or `pip`
- MacBook with CPU or Apple Silicon (MPS)

## Setup

### 1. Install Dependencies

Using `uv` (recommended):
```bash
cd /Users/david/Documents/Projects/nGPT
uv pip install torch numpy tiktoken requests
```

Or using `pip`:
```bash
pip install torch numpy tiktoken requests
```

### 2. Prepare Shakespeare Dataset

The dataset should already be prepared. If not, run:
```bash
cd /Users/david/Documents/Projects/nGPT
python prepare.py
```

Expected output:
```
train has 301,966 tokens
val has 36,059 tokens
```

This creates:
- `input.txt` - Raw Shakespeare text
- `train.bin` - Training data (~302K tokens)
- `val.bin` - Validation data (~36K tokens)

## Running the Mac Test

### Basic Test (CPU)

```bash
cd /Users/david/Documents/Projects/nGPT/modded-nanogpt
python train_mac.py --device cpu --steps 100
```

### Mac with Apple Silicon (MPS)

```bash
python train_mac.py --device mps --steps 100
```

### Custom Configuration

```bash
python train_mac.py \
    --device cpu \
    --steps 100 \
    --batch-size 4 \
    --data-dir /path/to/data
```

### Command-Line Options

- `--device`: Device to use (`cpu`, `mps`, or `cuda`)
- `--steps`: Number of training steps (default: 100)
- `--batch-size`: Batch size (default: 4)
- `--data-dir`: Directory containing train.bin and val.bin (default: `.`)

## Expected Behavior

### Training Progress

You should see output like:
```
Using device: cpu
Training for 100 steps

Loading data from .
  Vocab size: 50257
  Train tokens: 301,966
  Val tokens: 36,059

Model config:
  vocab_size: 50257
  n_layer: 4
  n_embd: 128
  n_head: 4
  block_size: 64

Total parameters: 6,905,728

Optimizer groups:
  Weight params (attn, mlp): 6,619,136
  Other params (embed, alpha, scale): 286,592

Starting training...
--------------------------------------------------------------------------------
Step   0: loss = 10.8234
Step  10: loss = 8.5432
Step  20: loss = 7.2341
...
Step  50: loss = 5.1234

================================================================================
NORM VERIFICATION (Step 50)
================================================================================

Weight norms (should be ~1.000):
  ✓ token_embedding.weight        : 1.000023
  ✓ position_embedding.weight     : 0.999987
  ✓ blocks.0.attn.qkv.weight      : 1.000045
  ✓ blocks.0.attn.out_proj.weight : 0.999912
  ✓ blocks.0.mlp.fc1.weight       : 1.000067
  ✓ blocks.0.mlp.fc2.weight       : 0.999934
  ...
  ✓ lm_head.weight                : 1.000011

Activation norms (should be ~1.000):
  ✓ token_embedding               : 1.000000
  ✓ position_embedding            : 1.000000
  ✓ combined_embedding            : 1.000000
  ✓ block_0_output                : 1.000000
  ✓ block_1_output                : 1.000000
  ✓ block_2_output                : 1.000000
  ✓ block_3_output                : 1.000000
================================================================================

...
Step  90: loss = 4.3210
--------------------------------------------------------------------------------
Training complete!

Final validation...
Validation loss: 4.5678

================================================================================
nGPT VERIFICATION COMPLETE
================================================================================

Pass criteria:
  1. Loss decreased: Check logs above
  2. Weight norms = 1.000 (±0.01): See verification at step 50
  3. Activation norms = 1.000 (±0.01): See verification at step 50
```

## Pass Criteria

From `DESIGN_nGPT_MUON.md`, the test passes if ALL of the following are true:

### 1. Loss Decreases (Not NaN)
- **Check:** Loss at step 100 < Loss at step 0
- **Expected:** Loss should decrease from ~11 to ~4-5
- **Failure:** Loss is NaN, Inf, or increases

### 2. Weight Norms = 1.000 (±0.01)
- **Check:** All 2D weight matrices have norm ~1.000 at step 50
- **Expected:** All weights show ✓ in verification output
- **Failure:** Any weight norm < 0.99 or > 1.01

### 3. Activation Norms = 1.000 (±0.01)
- **Check:** All block outputs have norm ~1.000 at step 50
- **Expected:** All activations show ✓ in verification output
- **Failure:** Any activation norm < 0.99 or > 1.01

## Troubleshooting

### Issue 1: Loss is NaN

**Symptoms:**
```
Step   0: loss = nan
```

**Possible Causes:**
1. Alpha initialization too large
2. Normalization epsilon too small
3. Weight projection not working

**Fixes:**
1. Check `alpha` init is ~0.05 in `SimpleBlock.__init__`
2. Verify `eps=1e-8` in `normalize_ngpt()`
3. Confirm `project_weights_to_hypersphere()` is called after `optimizer.step()`

### Issue 2: Norm Collapse (norms → 0)

**Symptoms:**
```
Weight norms (should be ~1.000):
  ✗ blocks.0.attn.qkv.weight      : 0.000123
```

**Possible Causes:**
1. Projection hook using wrong precision
2. Epsilon too small

**Fixes:**
1. Verify projection uses `float32`: `param_f32 = param.float()`
2. Check `eps=1e-8` (not 1e-10)

### Issue 3: Exploding Norms (norms >> 1.0)

**Symptoms:**
```
Weight norms (should be ~1.000):
  ✗ blocks.0.mlp.fc1.weight       : 12.345678
```

**Possible Causes:**
1. Projection hook not called
2. Projection skipping wrong parameters

**Fixes:**
1. Verify `project_weights_to_hypersphere(model)` is called after `optimizer.step()`
2. Check skip conditions in projection hook don't exclude 2D weights

### Issue 4: MPS Compatibility Errors

**Symptoms:**
```
RuntimeError: The MPS framework is not available
```
or
```
RuntimeError: <operation> not supported on MPS
```

**Fixes:**
1. Fall back to CPU: `--device cpu`
2. Update PyTorch to latest version
3. Some ops may not support MPS yet

### Issue 5: Poor Convergence

**Symptoms:**
- Loss decreases very slowly
- Loss plateaus early

**Possible Causes:**
1. Learning rate too low/high
2. Alpha parameters not learning

**Fixes:**
1. Check optimizer learning rates (1e-3 for weights, 3e-4 for others)
2. Verify alpha parameters have `.label = 'alpha'`
3. Confirm alpha parameters are in AdamW optimizer group

## Model Configuration

The Mac test uses a small model configuration for fast verification:

```python
config = {
    'vocab_size': 50257,  # GPT-2 vocabulary
    'n_layer': 4,         # 4 transformer blocks
    'n_embd': 128,        # 128-dimensional embeddings
    'n_head': 4,          # 4 attention heads
    'block_size': 64,     # 64-token context
}
```

**Total parameters:** ~6.9M

This is intentionally small (~1% of GPT-2) to enable fast iteration on Mac hardware.

## Next Steps

### If Test Passes ✓

1. **Scale to Full Train Script:**
   - Test `train_gpt.py` with full FineWeb dataset
   - Verify Muon optimizer integration
   - Test on H100 with CUDA/Triton

2. **Optimize:**
   - Implement lazy projection (normalize every N steps)
   - Add custom Triton kernels for normalized residuals
   - Implement geodesic Muon updates

### If Test Fails ✗

1. **Debug on Mac First:**
   - Use simplified `train_mac.py` for debugging
   - Add print statements in `normalize_ngpt()` and `project_weights_to_hypersphere()`
   - Check intermediate tensor norms

2. **Verify Architecture:**
   - Review `DESIGN_nGPT_MUON.md` specifications
   - Confirm all changes in `train_gpt.py` match plan
   - Check for missing normalization calls

## Architecture Summary

### nGPT Changes vs. Standard Transformer

| Component | Standard | nGPT |
|-----------|----------|------|
| Residual | `x = x + layer(norm(x))` | `x = normalize(x + alpha * layer(x))` |
| Normalization | RMSNorm (learnable) | L2 norm (fixed, unit sphere) |
| Weights | Unconstrained | Unit norm (via projection) |
| Embeddings | Standard | Normalized after lookup |
| Logits | `W @ x` | `cosine(W, x) * scale` |
| Alpha | N/A | Learnable per sub-layer |
| Logit Scale | N/A | Learnable scalar |

### Key nGPT Invariants

1. **All activations have unit norm:** `||x|| = 1` at every layer
2. **All weights have unit norm:** `||W|| = 1` along output dimension
3. **Residuals are normalized:** Sum first, then normalize
4. **No LayerNorm/RMSNorm:** Only L2 normalization

## Files

- `train_mac.py` - Simplified Mac trainer (this test)
- `train_gpt.py` - Full trainer with Muon/Triton (for H100)
- `prepare.py` - Shakespeare dataset preparation
- `train.bin`, `val.bin` - Tokenized Shakespeare data
- `DESIGN_nGPT_MUON.md` - Full architecture specification

## References

- **Design Doc:** `DESIGN_nGPT_MUON.md`
- **nGPT Paper:** [Normalized Transformer](https://arxiv.org/abs/2410.01131) (arXiv:2410.01131)
- **Muon Optimizer:** NorMuon in `train_gpt.py` lines 454-710

## Contact

For issues or questions, refer to the design document or check the implementation in `train_gpt.py`.
