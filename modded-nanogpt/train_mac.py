#!/usr/bin/env python3
"""
Simplified nGPT trainer for Mac (CPU/MPS) testing.
Stripped-down version without Triton/flash-attn/distributed dependencies.

Test config: n_layer=4, n_head=4, n_embd=128, block_size=64
Purpose: Verify nGPT architecture correctness on Shakespeare dataset.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from dataclasses import dataclass

# Parse args
parser = argparse.ArgumentParser(description='nGPT Mac Testing')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'mps', 'cpu'],
                    help='Device to use for training')
parser.add_argument('--steps', type=int, default=100,
                    help='Number of training steps')
parser.add_argument('--data-dir', type=str, default='.',
                    help='Directory containing train.bin and val.bin')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size for training')
args = parser.parse_args()

device = torch.device(args.device)
print(f"Using device: {device}")
print(f"Training for {args.steps} steps")

# -----------------------------------------------------------------------------
# nGPT normalization primitives

def normalize_ngpt(x: Tensor, dim=-1, eps=1e-8):
    """
    nGPT normalization: projects x onto unit hypersphere.
    CRITICAL: Forces float32 precision for numerical stability.
    """
    x_f32 = x.float()
    norm_sq = x_f32.pow(2).sum(dim=dim, keepdim=True).add_(eps)
    scale = norm_sq.rsqrt()
    return x.mul(scale.type_as(x))

@torch.no_grad()
def project_weights_to_hypersphere(model):
    """
    Project all weight parameters to unit hypersphere.
    Called immediately after optimizer.step().
    """
    for name, param in model.named_parameters():
        # Skip scalars and special parameters
        if any(skip in name for skip in ['alpha', 'bias', 'scale', 'logit_scale']):
            continue
        if param.ndim < 1:
            continue

        # Normalize along last dimension (output features)
        param_f32 = param.float()
        norm = param_f32.norm(p=2, dim=-1, keepdim=True).add_(1e-8)
        param.copy_((param_f32 / norm).type_as(param))

# -----------------------------------------------------------------------------
# Simple attention (no flash-attn, no RoPE for simplicity)

def simple_causal_attention(q, k, v):
    """
    Simple causal self-attention without flash attention.
    q, k, v: [B, num_heads, seq_len, head_dim]
    """
    B, num_heads, T, head_dim = q.shape

    # Scaled dot-product attention
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    return out

# -----------------------------------------------------------------------------
# Simplified Model Components

class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Initialize
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.out_proj.weight)

        # Labels for optimizer
        self.qkv.weight.label = 'attn'
        self.out_proj.weight.label = 'attn'

    def forward(self, x):
        B, T, C = x.shape

        # Compute QKV
        qkv = self.qkv(x)  # [B, T, 3*C]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Iteration 4: QK norm removed (hurts performance), inputs already normalized

        # Attention
        attn_out = simple_causal_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(attn_out)

        return out

class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

        # Initialize
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

        # Labels for optimizer
        self.fc1.weight.label = 'mlp'
        self.fc2.weight.label = 'mlp'

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()  # ReLU^2 activation (matches modded-nanogpt)
        x = self.fc2(x)
        return x

class SimpleBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.attn = SimpleAttention(dim, num_heads)
        self.mlp = SimpleMLP(dim)

        # nGPT: learnable alpha parameters for residual scaling
        # OPTIMAL: alpha=0.15 (best validation loss from iterative testing)
        self.alpha_attn = nn.Parameter(torch.full((dim,), 0.15))
        self.alpha_mlp = nn.Parameter(torch.full((dim,), 0.15))
        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'

    def forward(self, x):
        # nGPT: normalized residuals (no pre-norm!)
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Blocks
        self.blocks = nn.ModuleList([SimpleBlock(n_embd, n_head) for _ in range(n_layer)])

        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # nGPT: learnable logit temperature/scale
        # Iteration 1: Increased from ~0.089 to 10.0 for stronger gradients
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

        # Initialize
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

        # Labels for optimizer
        self.token_embedding.weight.label = 'embed'
        self.position_embedding.weight.label = 'embed'
        self.lm_head.weight.label = 'lm_head'
        self.logit_scale.label = 'logit_scale'

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        # nGPT: normalize embeddings immediately after lookup
        tok_emb = normalize_ngpt(self.token_embedding(idx), dim=-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = normalize_ngpt(self.position_embedding(pos), dim=-1)

        # Combine embeddings and normalize
        x = normalize_ngpt(tok_emb + pos_emb, dim=-1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # nGPT: cosine similarity logits
        x_norm = normalize_ngpt(x, dim=-1)
        w_norm = normalize_ngpt(self.lm_head.weight, dim=-1)
        logits = F.linear(x_norm, w_norm) * self.logit_scale

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# -----------------------------------------------------------------------------
# Optimizer setup

def get_optimizer_groups(model):
    """
    Separate parameters into groups:
    - weight_params: 2D weights (would use Muon on CUDA)
    - other_params: embeddings, alphas, scalars (use AdamW)
    """
    weight_params = []
    other_params = []

    for name, param in model.named_parameters():
        label = getattr(param, 'label', None)
        if label in ['attn', 'mlp']:
            weight_params.append(param)
        else:
            other_params.append(param)

    return weight_params, other_params

# -----------------------------------------------------------------------------
# Data loading

def load_data(data_dir):
    """Load Shakespeare dataset from binary files."""
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)

    train_data = torch.from_numpy(train_data.astype(np.int64))
    val_data = torch.from_numpy(val_data.astype(np.int64))

    return train_data, val_data

def get_batch(data, block_size, batch_size, device):
    """Get a random batch from the dataset."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# -----------------------------------------------------------------------------
# Main training loop

def main():
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_data, val_data = load_data(args.data_dir)
    vocab_size = int(max(train_data.max(), val_data.max())) + 1
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")

    # Model configuration (small for Mac testing)
    config = {
        'vocab_size': vocab_size,
        'n_layer': 4,
        'n_embd': 128,
        'n_head': 4,
        'block_size': 64,
    }
    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = SimpleGPT(**config).to(device)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer setup
    weight_params, other_params = get_optimizer_groups(model)
    print(f"\nOptimizer groups:")
    print(f"  Weight params (attn, mlp): {sum(p.numel() for p in weight_params):,}")
    print(f"  Other params (embed, alpha, scale): {sum(p.numel() for p in other_params):,}")

    # Use AdamW for all params (Muon requires CUDA)
    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': 1e-3},
        {'params': other_params, 'lr': 3e-4}
    ], weight_decay=0.01)

    # Training loop
    print(f"\nStarting training...")
    print("-" * 80)

    for step in range(args.steps):
        model.train()

        # Get batch
        x, y = get_batch(train_data, config['block_size'], args.batch_size, device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # nGPT: Project weights to hypersphere
        project_weights_to_hypersphere(model)

        # Logging
        if step % 10 == 0 or step == 50:
            print(f"Step {step:3d}: loss = {loss.item():.4f}")

        # Verification at step 50
        if step == 50:
            print("\n" + "=" * 80)
            print("NORM VERIFICATION (Step 50)")
            print("=" * 80)

            # Check weight norms
            print("\nWeight norms (should be ~1.000):")
            for name, param in model.named_parameters():
                if param.ndim >= 2 and 'alpha' not in name and 'logit_scale' not in name:
                    norm = param.norm(p=2, dim=-1).mean().item()
                    status = "✓" if abs(norm - 1.0) < 0.01 else "✗"
                    print(f"  {status} {name:30s}: {norm:.6f}")

            # Check activation norms
            print("\nActivation norms (should be ~1.000):")
            with torch.no_grad():
                x_test, _ = get_batch(train_data, config['block_size'], 1, device)

                # Token embeddings
                tok_emb = normalize_ngpt(model.token_embedding(x_test), dim=-1)
                emb_norm = tok_emb.norm(p=2, dim=-1).mean().item()
                status = "✓" if abs(emb_norm - 1.0) < 0.01 else "✗"
                print(f"  {status} {'token_embedding':30s}: {emb_norm:.6f}")

                # Position embeddings
                pos = torch.arange(0, config['block_size'], dtype=torch.long, device=device)
                pos_emb = normalize_ngpt(model.position_embedding(pos), dim=-1)
                pos_norm = pos_emb.norm(p=2, dim=-1).mean().item()
                status = "✓" if abs(pos_norm - 1.0) < 0.01 else "✗"
                print(f"  {status} {'position_embedding':30s}: {pos_norm:.6f}")

                # Combined embeddings
                x = normalize_ngpt(tok_emb + pos_emb, dim=-1)
                combined_norm = x.norm(p=2, dim=-1).mean().item()
                status = "✓" if abs(combined_norm - 1.0) < 0.01 else "✗"
                print(f"  {status} {'combined_embedding':30s}: {combined_norm:.6f}")

                # Block outputs
                for i, block in enumerate(model.blocks):
                    x = block(x)
                    block_norm = x.norm(p=2, dim=-1).mean().item()
                    status = "✓" if abs(block_norm - 1.0) < 0.01 else "✗"
                    print(f"  {status} {f'block_{i}_output':30s}: {block_norm:.6f}")

            print("=" * 80 + "\n")

    print("-" * 80)
    print("Training complete!")

    # Final validation
    print("\nFinal validation...")
    model.eval()
    val_loss = 0
    val_steps = min(10, len(val_data) // (config['block_size'] * args.batch_size))

    with torch.no_grad():
        for _ in range(val_steps):
            x, y = get_batch(val_data, config['block_size'], args.batch_size, device)
            _, loss = model(x, y)
            val_loss += loss.item()

    val_loss /= val_steps
    print(f"Validation loss: {val_loss:.4f}")

    print("\n" + "=" * 80)
    print("nGPT VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nPass criteria:")
    print("  1. Loss decreased: Check logs above")
    print("  2. Weight norms = 1.000 (±0.01): See verification at step 50")
    print("  3. Activation norms = 1.000 (±0.01): See verification at step 50")

if __name__ == '__main__':
    main()
