#!/usr/bin/env python3
"""
Optimized nGPT trainer for H100 - implements improvements from DESIGN_nGPT_MUON.md
Includes: lazy projection, scalar alpha, scaled-up model
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from dataclasses import dataclass

# Parse args
parser = argparse.ArgumentParser(description='Optimized nGPT H100 Training')
parser.add_argument('--alpha', type=float, default=0.28, help='Alpha parameter (from sweep)')
parser.add_argument('--logit-scale', type=float, default=10.0, help='Logit scale')
parser.add_argument('--steps', type=int, default=1000, help='Training steps')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--n-layer', type=int, default=8, help='Number of layers (scaled up)')
parser.add_argument('--n-embd', type=int, default=384, help='Embedding dimension (scaled up)')
parser.add_argument('--lazy-proj-freq', type=int, default=0, help='Lazy projection frequency (0=every step)')
parser.add_argument('--scalar-alpha', action='store_true', help='Use scalar alpha per layer (not vector)')
parser.add_argument('--output', type=str, default='optimizations.jsonl', help='Output file')
args = parser.parse_args()

device = torch.device('cuda')
print(f"Optimized nGPT Training on H100")
print(f"=" * 80)
print(f"Model: {args.n_layer} layers x {args.n_embd} dim")
print(f"Alpha: {args.alpha} ({'scalar' if args.scalar_alpha else 'vector'})")
print(f"Lazy projection: {'every ' + str(args.lazy_proj_freq) + ' steps' if args.lazy_proj_freq > 0 else 'every step'}")
print(f"=" * 80)

# -----------------------------------------------------------------------------
# nGPT normalization primitives

def normalize_ngpt(x: Tensor, dim=-1, eps=1e-8):
    """nGPT normalization: projects x onto unit hypersphere."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    norm_sq = x_f32.pow(2).sum(dim=dim, keepdim=True).add_(eps)
    scale = norm_sq.rsqrt()
    result = x_f32 * scale
    return result.to(orig_dtype)

@torch.no_grad()
def project_weights_to_hypersphere(model):
    """
    Project all 2D weights to unit hypersphere.
    Optimization #1: Lazy Projection - can be called every N steps instead of every step.
    """
    for name, param in model.named_parameters():
        label = getattr(param, 'label', None)
        if label in ['attn', 'mlp', 'lm_head', 'embed']:
            if param.ndim >= 2:
                param_f32 = param.float()
                norm = param_f32.norm(p=2, dim=-1, keepdim=True).add_(1e-8)
                param.copy_((param_f32 / norm).type_as(param))

# -----------------------------------------------------------------------------
# Model components

class SimpleAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.qkv.weight.label = 'attn'
        self.out_proj.weight.label = 'attn'

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y

class SimpleMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

        self.fc1.weight.label = 'mlp'
        self.fc2.weight.label = 'mlp'

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class SimpleBlock(nn.Module):
    """
    nGPT Block with optimization support.
    Optimization #8: Scalar vs Vector Alpha - controlled by scalar_alpha flag.
    """
    def __init__(self, n_embd, n_head, alpha_init=0.28, scalar_alpha=False):
        super().__init__()
        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        # Optimization #8: Scalar alpha (single value) vs Vector alpha (per-channel)
        if scalar_alpha:
            self.alpha_attn = nn.Parameter(torch.tensor(alpha_init))
            self.alpha_mlp = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.alpha_attn = nn.Parameter(torch.full((n_embd,), alpha_init))
            self.alpha_mlp = nn.Parameter(torch.full((n_embd,), alpha_init))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'

    def forward(self, x):
        # nGPT: normalize(x + alpha * sublayer(x))
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x

class SimpleGPT(nn.Module):
    def __init__(self, config, alpha_init=0.28, logit_scale_init=10.0, scalar_alpha=False):
        super().__init__()
        self.block_size = config['block_size']
        self.n_embd = config['n_embd']

        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.ModuleList([
            SimpleBlock(config['n_embd'], config['n_head'], alpha_init, scalar_alpha)
            for _ in range(config['n_layer'])
        ])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

        self.token_embedding.weight.label = 'embed'
        self.position_embedding.weight.label = 'embed'
        self.lm_head.weight.label = 'lm_head'
        self.logit_scale.label = 'logit_scale'

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        # nGPT: normalize embeddings
        tok_emb = normalize_ngpt(self.token_embedding(idx), dim=-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = normalize_ngpt(self.position_embedding(pos), dim=-1)
        x = normalize_ngpt(tok_emb + pos_emb, dim=-1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # nGPT: cosine similarity logits
        x_norm = normalize_ngpt(x, dim=-1)
        w_norm = normalize_ngpt(self.lm_head.weight, dim=-1)
        if x_norm.dtype != w_norm.dtype:
            w_norm = w_norm.to(x_norm.dtype)
        logits = F.linear(x_norm, w_norm) * self.logit_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# -----------------------------------------------------------------------------
# Data loading

def load_fineweb_shard(shard_path):
    """Load a single FineWeb binary shard."""
    with open(shard_path, 'rb') as f:
        f.read(256)  # Skip header
        tokens = np.fromfile(f, dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int64))

def load_fineweb_data(data_dir, num_train_shards=10):
    """Load FineWeb data from binary shards."""
    train_shards = []
    for i in range(1, num_train_shards + 1):
        path = os.path.join(data_dir, f'fineweb_train_{i:06d}.bin')
        if os.path.exists(path):
            print(f"  Loading {path}...")
            train_shards.append(load_fineweb_shard(path))

    val_path = os.path.join(data_dir, 'fineweb_val_000000.bin')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    train_data = torch.cat(train_shards) if train_shards else None
    val_data = load_fineweb_shard(val_path)

    return train_data, val_data

def get_batch(data, block_size, batch_size, device):
    """Get a random batch from the dataset."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# -----------------------------------------------------------------------------
# Main training

def main():
    t0 = time.time()

    # Load data
    print("\nLoading FineWeb data...")
    data_dir = 'data/fineweb10B'
    train_data, val_data = load_fineweb_data(data_dir, num_train_shards=10)
    vocab_size = 50257
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")

    # Model configuration - SCALED UP
    config = {
        'vocab_size': vocab_size,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': 8,
        'block_size': 128,
    }

    print(f"\nModel config (SCALED UP):")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = SimpleGPT(config, alpha_init=args.alpha, logit_scale_init=args.logit_scale,
                      scalar_alpha=args.scalar_alpha).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Optimizer
    weight_params = [p for p in model.parameters() if getattr(p, 'label', None) in ['attn', 'mlp']]
    other_params = [p for p in model.parameters() if getattr(p, 'label', None) not in ['attn', 'mlp']]

    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr * 0.3}
    ])

    print(f"\nOptimizer: AdamW (lr={args.lr})")
    print(f"  Weight params: {sum(p.numel() for p in weight_params):,}")
    print(f"  Other params: {sum(p.numel() for p in other_params):,}")

    # Optimization info
    print(f"\nOptimizations enabled:")
    print(f"  Lazy projection: {'Yes (every ' + str(args.lazy_proj_freq) + ' steps)' if args.lazy_proj_freq > 0 else 'No (every step)'}")
    print(f"  Scalar alpha: {'Yes' if args.scalar_alpha else 'No (vector alpha)'}")

    # Training
    print(f"\nStarting training...")
    print("-" * 80)

    losses = []
    proj_count = 0

    for step in range(args.steps):
        # Training step
        x, y = get_batch(train_data, config['block_size'], args.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optimization #1: Lazy Projection
        if args.lazy_proj_freq == 0 or step % args.lazy_proj_freq == 0:
            project_weights_to_hypersphere(model)
            proj_count += 1

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Step {step:4d}: loss = {loss.item():.4f}")

    # Final validation
    print("-" * 80)
    print("Computing final validation loss...")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(val_data, config['block_size'], args.batch_size, device)
            _, loss = model(x, y)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    print(f"Validation loss: {val_loss:.4f}")

    # Save results
    elapsed = time.time() - t0
    result = {
        'alpha': args.alpha,
        'logit_scale': args.logit_scale,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_params': n_params,
        'lr': args.lr,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'lazy_proj_freq': args.lazy_proj_freq,
        'scalar_alpha': args.scalar_alpha,
        'projection_count': proj_count,
        'final_train_loss': losses[-1],
        'val_loss': val_loss,
        'train_loss_reduction': (losses[0] - losses[-1]) / losses[0] * 100,
        'time_seconds': elapsed,
        'tokens_per_sec': (args.steps * args.batch_size * config['block_size']) / elapsed
    }

    # Append to JSONL file
    with open(args.output, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"\nResults saved to {args.output}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Throughput: {result['tokens_per_sec']:,.0f} tokens/sec")
    print(f"Projections performed: {proj_count}/{args.steps}")

    return result

if __name__ == '__main__':
    result = main()
