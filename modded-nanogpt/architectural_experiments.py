#!/usr/bin/env python3
"""
Architectural Improvement Experiments for nGPT
Tests novel architectural variations on modded-nanogpt size model (11L x 768D)
Uses smaller dataset for rapid iteration
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# New architectural hypotheses to test
# -----------------------------------------------------------------------------

HYPOTHESES = """
ARCHITECTURAL IMPROVEMENT HYPOTHESES FOR nGPT

Baseline: Best config from previous experiments
- Model: 11 layers × 768 dim (matching modded-nanogpt)
- Alpha: 0.28, Logit scale: 15.0, Batch size: 48
- Lazy projection freq: 7, Scalar alpha: True

NEW HYPOTHESES TO TEST:

1. LAYER-SPECIFIC ALPHA SCALING
   Hypothesis: Different layers benefit from different alpha values
   - Early layers: Lower alpha (more conservative residuals)
   - Middle layers: Higher alpha (stronger signal propagation)
   - Late layers: Medium alpha (balanced)
   Implementation: alpha_i = base_alpha * scale_func(layer_idx)
   Expected: 1-2% improvement from layer-specialized signal flow

2. LEARNABLE LOGIT TEMPERATURE
   Hypothesis: Fixed logit scale is suboptimal, should adapt during training
   - Start at 15.0, allow gradient-based adjustment
   - Per-position temperature for long-range vs short-range predictions
   Implementation: logit_temperature as nn.Parameter with constraints
   Expected: 0.5-1% improvement from adaptive scaling

3. GATED RESIDUAL CONNECTIONS
   Hypothesis: Learn which residuals to emphasize
   - Gate: g = sigmoid(linear(x))
   - Residual: norm(x + g * alpha * layer(x))
   Implementation: Add learnable gate before each residual
   Expected: 1-3% improvement from selective residual learning

4. DUAL-SCALE NORMALIZATION
   Hypothesis: Track both local and global normalization statistics
   - Local: normalize per example (current approach)
   - Global: track running statistics like BatchNorm
   - Combine with learned weighting
   Implementation: Mix local and EMA global norms
   Expected: 0.5-1.5% improvement from stabler normalization

5. ATTENTION-SPECIFIC ALPHA
   Hypothesis: Attention and MLP residuals need different scaling
   - Attention: Lower alpha (attention is powerful, don't overweight)
   - MLP: Higher alpha (MLP needs stronger signal)
   Implementation: alpha_attn ≠ alpha_mlp per layer
   Expected: 1-2% improvement from operation-specific tuning

6. PROGRESSIVE HYPERSPHERE RADIUS
   Hypothesis: Fixed unit norm too restrictive, allow slight radius variation
   - Start strict (radius=1.0), gradually allow 0.95-1.05
   - Learn optimal radius per layer
   Implementation: Normalize to learnable radius instead of 1.0
   Expected: 0.5-1% improvement from relaxed geometry

7. MULTI-HEAD ALPHA
   Hypothesis: Different attention heads need different residual strengths
   - Per-head alpha values for attention
   - Allows specialization (some heads strong, some weak)
   Implementation: alpha_attn as vector[num_heads]
   Expected: 1-2% improvement from head specialization

8. ADAPTIVE PROJECTION FREQUENCY
   Hypothesis: Different training phases need different projection rates
   - Early training: Frequent projection (every 5 steps)
   - Late training: Infrequent projection (every 15 steps)
   Implementation: Schedule based on step or loss gradient
   Expected: 0.5-1% improvement + speedup

9. HIERARCHICAL NORMALIZATION
   Hypothesis: Normalize at multiple granularities
   - Token-level: Current approach
   - Head-level: Normalize within each attention head
   - Layer-level: Normalize across layer outputs
   Implementation: Multi-level normalization with learned mixing
   Expected: 1-2% improvement from richer normalization

10. RESIDUAL DROPOUT ON HYPERSPHERE
    Hypothesis: Dropout adapted for hypersphere geometry
    - Randomly zero out dimensions, then renormalize
    - Drop residual connections probabilistically
    Implementation: Structured dropout before normalization
    Expected: 1-3% improvement from better regularization
"""

# -----------------------------------------------------------------------------
# nGPT base primitives
# -----------------------------------------------------------------------------

def normalize_ngpt(x: Tensor, dim=-1, eps=1e-8):
    """nGPT normalization: projects x onto unit hypersphere."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    norm_sq = x_f32.pow(2).sum(dim=dim, keepdim=True).add_(eps)
    scale = norm_sq.rsqrt()
    result = x_f32 * scale
    return result.to(orig_dtype)

def normalize_to_radius(x: Tensor, radius: float, dim=-1, eps=1e-8):
    """Normalize to specific radius (for hypothesis 6)."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    norm_sq = x_f32.pow(2).sum(dim=dim, keepdim=True).add_(eps)
    scale = (radius * radius) / norm_sq
    scale = scale.sqrt()
    result = x_f32 * scale
    return result.to(orig_dtype)

@torch.no_grad()
def project_weights_to_hypersphere(model):
    """Project all 2D weights to unit hypersphere."""
    for name, param in model.named_parameters():
        label = getattr(param, 'label', None)
        if label in ['attn', 'mlp', 'lm_head', 'embed']:
            if param.ndim >= 2:
                param_f32 = param.float()
                norm = param_f32.norm(p=2, dim=-1, keepdim=True).add_(1e-8)
                param.copy_((param_f32 / norm).type_as(param))

# -----------------------------------------------------------------------------
# Hypothesis 1: Layer-Specific Alpha Scaling
# -----------------------------------------------------------------------------

def get_layer_alpha_schedule(num_layers, base_alpha=0.28, schedule_type='linear'):
    """Generate layer-specific alpha values."""
    if schedule_type == 'linear':
        # Linearly increase from 0.7x to 1.3x base_alpha
        alphas = torch.linspace(0.7 * base_alpha, 1.3 * base_alpha, num_layers)
    elif schedule_type == 'upsidedown_u':
        # U-shape: low-high-low (middle layers get higher alpha)
        t = torch.linspace(0, 1, num_layers)
        alphas = base_alpha * (1.0 + 0.5 * torch.sin(t * 3.14159))
    elif schedule_type == 'downsideup_u':
        # Inverted U: high-low-high (middle layers get lower alpha)
        t = torch.linspace(0, 1, num_layers)
        alphas = base_alpha * (1.0 - 0.3 * torch.sin(t * 3.14159))
    else:
        alphas = torch.full((num_layers,), base_alpha)
    return alphas

# -----------------------------------------------------------------------------
# Hypothesis 3: Gated Residual Block
# -----------------------------------------------------------------------------

class GatedResidualBlock(nn.Module):
    """Block with learned gates for residual connections."""
    def __init__(self, n_embd, n_head, alpha_init=0.28):
        super().__init__()
        from train_h100_optimized import SimpleAttention, SimpleMLP

        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        # Learnable gates
        self.gate_attn = nn.Parameter(torch.ones(n_embd))
        self.gate_mlp = nn.Parameter(torch.ones(n_embd))

        # Scalar alpha
        self.alpha_attn = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_mlp = nn.Parameter(torch.tensor(alpha_init))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'
        self.gate_attn.label = 'gate'
        self.gate_mlp.label = 'gate'

    def forward(self, x):
        # Gated residual for attention
        attn_out = self.attn(x)
        gate_a = torch.sigmoid(self.gate_attn)
        x = normalize_ngpt(x + gate_a * self.alpha_attn * attn_out)

        # Gated residual for MLP
        mlp_out = self.mlp(x)
        gate_m = torch.sigmoid(self.gate_mlp)
        x = normalize_ngpt(x + gate_m * self.alpha_mlp * mlp_out)

        return x

# -----------------------------------------------------------------------------
# Hypothesis 5: Attention-Specific Alpha (asymmetric)
# -----------------------------------------------------------------------------

class AsymmetricAlphaBlock(nn.Module):
    """Block with different alphas for attention vs MLP."""
    def __init__(self, n_embd, n_head, alpha_attn=0.20, alpha_mlp=0.35):
        super().__init__()
        from train_h100_optimized import SimpleAttention, SimpleMLP

        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        self.alpha_attn = nn.Parameter(torch.tensor(alpha_attn))
        self.alpha_mlp = nn.Parameter(torch.tensor(alpha_mlp))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'

    def forward(self, x):
        # Attention with lower alpha
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        # MLP with higher alpha
        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x

# -----------------------------------------------------------------------------
# Experiment configurations
# -----------------------------------------------------------------------------

def get_architectural_experiments():
    """Define all architectural variation experiments."""

    # Use smaller dataset for faster iteration
    # Use Shakespeare for quick testing (300K tokens vs 1B)
    base_config = {
        'n_layer': 11,
        'n_embd': 768,
        'n_head': 6,
        'block_size': 128,
        'steps': 300,  # Shorter for rapid iteration
        'batch_size': 32,  # Smaller for Shakespeare dataset
        'lr': 0.001,
        'alpha': 0.28,
        'logit_scale': 15.0,
        'lazy_proj_freq': 7,
        'scalar_alpha': True,
        'dataset': 'shakespeare',  # Smaller dataset
    }

    experiments = []

    # Baseline: Best previous config
    experiments.append({
        'name': 'baseline',
        'description': 'Baseline: Best config from previous experiments',
        'config': base_config.copy(),
        'hypothesis': None,
    })

    # Hypothesis 1: Layer-specific alpha (3 variants)
    for schedule in ['linear', 'upsidedown_u', 'downsideup_u']:
        exp = base_config.copy()
        exp['layer_alpha_schedule'] = schedule
        experiments.append({
            'name': f'layer_alpha_{schedule}',
            'description': f'H1: Layer-specific alpha ({schedule})',
            'config': exp,
            'hypothesis': 1,
        })

    # Hypothesis 2: Learnable logit temperature
    exp = base_config.copy()
    exp['learnable_logit_scale'] = True
    experiments.append({
        'name': 'learnable_logit_temp',
        'description': 'H2: Learnable logit temperature',
        'config': exp,
        'hypothesis': 2,
    })

    # Hypothesis 3: Gated residuals
    exp = base_config.copy()
    exp['gated_residuals'] = True
    experiments.append({
        'name': 'gated_residuals',
        'description': 'H3: Gated residual connections',
        'config': exp,
        'hypothesis': 3,
    })

    # Hypothesis 5: Attention-specific alpha (asymmetric)
    for attn_a, mlp_a in [(0.20, 0.35), (0.15, 0.40), (0.25, 0.30)]:
        exp = base_config.copy()
        exp['alpha_attn'] = attn_a
        exp['alpha_mlp'] = mlp_a
        exp['asymmetric_alpha'] = True
        experiments.append({
            'name': f'asymmetric_alpha_{attn_a}_{mlp_a}',
            'description': f'H5: Asymmetric alpha (attn={attn_a}, mlp={mlp_a})',
            'config': exp,
            'hypothesis': 5,
        })

    # Hypothesis 6: Progressive hypersphere radius
    for target_radius in [0.95, 1.0, 1.05]:
        exp = base_config.copy()
        exp['target_radius'] = target_radius
        exp['progressive_radius'] = True
        experiments.append({
            'name': f'progressive_radius_{target_radius}',
            'description': f'H6: Progressive radius (r={target_radius})',
            'config': exp,
            'hypothesis': 6,
        })

    # Hypothesis 8: Adaptive projection frequency
    exp = base_config.copy()
    exp['adaptive_proj_freq'] = True
    exp['proj_freq_range'] = (5, 15)
    experiments.append({
        'name': 'adaptive_proj_freq',
        'description': 'H8: Adaptive projection frequency',
        'config': exp,
        'hypothesis': 8,
    })

    return experiments

# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------

def print_hypotheses():
    """Print all hypotheses."""
    print(HYPOTHESES)

if __name__ == '__main__':
    print_hypotheses()
    print("\n" + "="*80)
    print("PLANNED EXPERIMENTS")
    print("="*80)

    experiments = get_architectural_experiments()
    for i, exp in enumerate(experiments, 1):
        print(f"\n{i}. {exp['name']}")
        print(f"   {exp['description']}")
        if exp['hypothesis']:
            print(f"   Hypothesis: #{exp['hypothesis']}")

    print(f"\nTotal: {len(experiments)} experiments planned")
    print("="*80)
