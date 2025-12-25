# nGPT: Normalized Transformer with Muon Optimizer

**Status:** Phase 1 Complete ✅ | Ready for H100 Testing 🚀

Implementation of Normalized Transformer (nGPT) architecture applied to the `modded-nanogpt` codebase, optimized for training efficiency with the Muon optimizer.

---

## 🎯 What is nGPT?

nGPT is a transformer architecture that enforces **unit-norm constraints** on all activations and weights, replacing traditional LayerNorm/RMSNorm with explicit L2 normalization. Key properties:

- **Residuals:** `x = normalize(x + alpha * layer(x))` instead of `x = x + layer(norm(x))`
- **Weights:** All weight matrices projected to unit hypersphere after each optimizer step
- **Logits:** Cosine similarity instead of standard linear projection
- **Stability:** Perfect norm enforcement (all norms = 1.000) prevents gradient explosion/vanishing

---

## ✅ Phase 1: Complete (Mac Verification)

### Implemented Features

- ✅ **Core Architecture:** Normalized residuals, weight projection, cosine similarity logits
- ✅ **Optimizer Integration:** Muon for 2D weights, AdamW for embeddings/scalars
- ✅ **Hyperparameter Optimization:** Iterative testing found optimal alpha=0.15, logit_scale=10.0
- ✅ **Verification:** Perfect unit norms, stable training, no NaN/Inf issues
- ✅ **Performance:** 42.6% loss reduction (vs 45.6% for original) on Shakespeare dataset

### Test Results

| Metric | Original | nGPT (Optimized) | Difference |
|--------|----------|------------------|------------|
| Final Loss | 5.89 | 6.48 | +10% |
| Validation Loss | 5.98 | 6.43 | +7.5% |
| Loss Reduction | 45.6% | 42.6% | -3.0pp |
| Norm Stability | N/A | 1.000 ± 0.000 | Perfect ✅ |

**Conclusion:** nGPT within 7.5% of original performance with perfect geometric constraints.

---

## 🚀 Phase 2: TODO (H100 Production)

### Ready for Testing

1. **Full-scale training** on FineWeb dataset (10B+ tokens)
2. **Muon optimizer** validation at production scale
3. **Distributed training** across multiple H100 GPUs
4. **Performance benchmarking** vs baseline modded-nanogpt

### Optimization Opportunities

1. **Lazy projection:** Normalize every N steps instead of every step
2. **Fused Triton kernels:** Custom kernel for `normalize(x + alpha*z)`
3. **Geodesic Muon:** Exact manifold updates using rotation matrices
4. **FP8 quantization:** Efficient distributed AllReduce on normalized activations
5. **Dynamic alpha:** Schedule alpha to increase during training

---

## 📁 Repository Structure

```
nGPT/
├── README.md                          # This file
├── DESIGN_nGPT_MUON.md               # Complete design specification
├── FINAL_COMPARISON_REPORT.md        # Iterative optimization analysis
├── COMPARISON_REPORT.md              # Initial baseline comparison
├── prepare.py                         # Shakespeare dataset preparation
├── train.bin, val.bin                # Tokenized Shakespeare data
│
└── modded-nanogpt/
    ├── train_gpt.py                  # Production trainer (optimized nGPT)
    ├── train_mac.py                  # Mac testing harness
    ├── MAC_TESTING.md                # Mac testing instructions
    └── (other modded-nanogpt files)
```

---

## 🔧 Quick Start

### Mac Testing (Phase 1 Verification)

```bash
# Clone repository
git clone <repo-url> nGPT
cd nGPT

# Install dependencies
pip install torch numpy tiktoken requests

# Prepare Shakespeare dataset
python prepare.py

# Run Mac test (200 steps, ~5 minutes on CPU)
cd modded-nanogpt
python train_mac.py --device cpu --steps 200 --data-dir ..
```

**Expected output:**
- Loss decreases from ~11.3 to ~6.5
- All weight norms = 1.000 at step 50
- All activation norms = 1.000 at step 50
- No NaN/Inf values

### H100 Production (Phase 2)

```bash
# Clone on H100 system
git clone <repo-url> nGPT
cd nGPT/modded-nanogpt

# Setup environment
pip install torch numpy tiktoken

# Download FineWeb dataset
# (follow modded-nanogpt data preparation instructions)

# Run full training
python train_gpt.py
```

**Configuration:**
- Alpha: 0.15 (optimized)
- Logit Scale: 10.0 (optimized)
- Muon optimizer for 2D weights
- AdamW for embeddings, alphas, scalars
- Weight projection after every optimizer step

---

## 📊 Key Findings from Optimization

### 1. Alpha Parameter is Critical

| Alpha | Performance | Notes |
|-------|-------------|-------|
| 0.05 (original spec) | 1.4% learning | Too conservative |
| **0.15 (optimized)** | **42.6% learning** | **Optimal** ✅ |
| 0.20 | 42.0% learning | Slightly less stable |

**Insight:** Alpha controls information flow. 0.05 allowed only 5% signal per layer, starving the network. 0.15 provides optimal balance.

### 2. Logit Scale Needs Large Boost

| Scale | Dynamic Range | Effect |
|-------|---------------|--------|
| 0.089 (1/√128) | [-0.09, 0.09] | Negligible gradients |
| **10.0 (optimized)** | **[-10, 10]** | **Strong gradients** ✅ |

**Insight:** Cosine similarity is bounded [-1, 1], requiring significant scaling for effective learning.

### 3. QK Normalization is Harmful

Adding explicit QK normalization (as in original modded-nanogpt) **worsens** validation loss by 2.5% in nGPT.

**Reason:** Inputs are already normalized in nGPT. Additional normalization disrupts the balanced geometry.

---

## 🔬 Architecture Details

### Normalized Residual Block

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = Attention(dim)
        self.mlp = MLP(dim)
        self.alpha_attn = nn.Parameter(torch.full((dim,), 0.15))
        self.alpha_mlp = nn.Parameter(torch.full((dim,), 0.15))

    def forward(self, x):
        # nGPT: Normalize after residual addition
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x
```

### Weight Projection Hook

```python
@torch.no_grad()
def project_weights_to_hypersphere(model):
    """Called after optimizer.step()"""
    for name, param in model.named_parameters():
        if 'alpha' in name or 'bias' in name or 'logit_scale' in name:
            continue
        # Normalize to unit sphere (float32 for precision)
        param_f32 = param.float()
        norm = param_f32.norm(p=2, dim=-1, keepdim=True).add_(1e-8)
        param.copy_((param_f32 / norm).type_as(param))
```

### Cosine Similarity Logits

```python
def forward(self, x):
    # ... transformer blocks ...

    # Final logits via cosine similarity
    x_norm = normalize_ngpt(x, dim=-1)
    w_norm = normalize_ngpt(self.lm_head.weight, dim=-1)
    logits = F.linear(x_norm, w_norm) * self.logit_scale

    return logits
```

---

## 📚 References

### Papers
- **nGPT:** [Normalized Transformer](https://arxiv.org/abs/2410.01131) (arXiv:2410.01131)
- **Muon Optimizer:** Implemented in modded-nanogpt

### Repositories
- **modded-nanogpt:** https://github.com/KellerJordan/modded-nanogpt
- **Original nanoGPT:** https://github.com/karpathy/nanoGPT

### Documentation
- `DESIGN_nGPT_MUON.md` - Complete architecture specification
- `FINAL_COMPARISON_REPORT.md` - Optimization analysis with 4 iterations
- `MAC_TESTING.md` - Mac testing protocol and troubleshooting

---

## 🤝 Contributing

This is a research implementation. For questions or issues:
1. Check `DESIGN_nGPT_MUON.md` for architecture details
2. Review `FINAL_COMPARISON_REPORT.md` for optimization rationale
3. Run Mac test to verify setup: `python train_mac.py --device cpu --steps 100`

---

## 📄 License

This project builds on modded-nanogpt. Please refer to the original repository for licensing.

---

**Status:** Ready for H100 deployment 🚀
**Last Updated:** 2025-12-25
**Contact:** See DESIGN_nGPT_MUON.md for implementation details
