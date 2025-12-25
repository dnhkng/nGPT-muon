#!/usr/bin/env python3
"""
nGPT Architectural Variations Trainer
Tests novel architectural hypotheses on modded-nanogpt size model (11L x 768D)
Uses subset of FineWeb for rapid iteration (~300K tokens)
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

# Parse args
parser = argparse.ArgumentParser(description='nGPT Architectural Experiments')
parser.add_argument('--name', type=str, required=True, help='Experiment name')
parser.add_argument('--hypothesis', type=int, default=0, help='Hypothesis number (0=baseline)')
parser.add_argument('--alpha', type=float, default=0.28, help='Base alpha value')
parser.add_argument('--logit-scale', type=float, default=15.0, help='Logit scale')
parser.add_argument('--steps', type=int, default=300, help='Training steps')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--n-layer', type=int, default=11, help='Number of layers')
parser.add_argument('--n-embd', type=int, default=768, help='Embedding dimension')
parser.add_argument('--lazy-proj-freq', type=int, default=7, help='Lazy projection frequency')
parser.add_argument('--layer-alpha-schedule', type=str, default=None, help='Layer alpha schedule')
parser.add_argument('--gated-residuals', action='store_true', help='Use gated residuals (H3)')
parser.add_argument('--asymmetric-alpha', action='store_true', help='Use asymmetric alpha (H5)')
parser.add_argument('--alpha-attn', type=float, default=0.20, help='Attention alpha (if asymmetric)')
parser.add_argument('--alpha-mlp', type=float, default=0.35, help='MLP alpha (if asymmetric)')
parser.add_argument('--progressive-radius', action='store_true', help='Progressive radius (H6)')
parser.add_argument('--target-radius', type=float, default=1.0, help='Target radius (if progressive)')
parser.add_argument('--learnable-logit-scale', action='store_true', help='Learnable logit scale (H2)')
parser.add_argument('--gate-init', type=float, default=0.0, help='Gate initialization value')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'muon'], help='Optimizer type')
parser.add_argument('--momentum', type=float, default=0.95, help='Muon momentum')
parser.add_argument('--output', type=str, default='architectural_results.jsonl', help='Output file')
args = parser.parse_args()

device = torch.device('cuda')
print(f"nGPT Architectural Experiment: {args.name}")
print(f"=" * 80)
print(f"Model: {args.n_layer} layers x {args.n_embd} dim (matching modded-nanogpt)")
print(f"Hypothesis: {args.hypothesis if args.hypothesis > 0 else 'Baseline'}")
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

def normalize_to_radius(x: Tensor, radius: float, dim=-1, eps=1e-8):
    """Normalize to specific radius."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    norm_sq = x_f32.pow(2).sum(dim=dim, keepdim=True).add_(eps)
    scale = (radius * radius) / norm_sq
    scale = scale.sqrt()
    result = x_f32 * scale
    return result.to(orig_dtype)

@torch.no_grad()
def project_weights_to_hypersphere(model, radius=1.0):
    """Project all 2D weights to hypersphere."""
    for name, param in model.named_parameters():
        label = getattr(param, 'label', None)
        if label in ['attn', 'mlp', 'lm_head', 'embed']:
            if param.ndim >= 2:
                param_f32 = param.float()
                norm = param_f32.norm(p=2, dim=-1, keepdim=True).add_(1e-8)
                param.copy_((param_f32 / norm * radius).type_as(param))

# -----------------------------------------------------------------------------
# Simplified Muon Optimizer for nGPT

def newton_schulz_orthogonalize(G, num_iters=5):
    """
    Orthogonalize matrix G using Newton-Schulz iteration.
    Computes the nearest orthogonal matrix to G.

    Newton-Schulz iteration: X_{k+1} = X_k * (3I - X_k^T X_k) / 2
    """
    # Transpose if more columns than rows (to work with tall matrices)
    if G.size(-2) < G.size(-1):
        G = G.mT
        transposed = True
    else:
        transposed = False

    # Work in bfloat16 for efficiency
    X = G.bfloat16()
    # Normalize to ensure convergence
    X = X / (X.norm() * 1.1 + 1e-7)

    # Identity matrix
    I = torch.eye(X.size(-1), device=X.device, dtype=X.dtype)

    # Iterate
    for _ in range(num_iters):
        # X = X @ (1.5 * I - 0.5 * X.T @ X)
        XTX = X.mT @ X
        A = 1.5 * I - 0.5 * XTX
        X = X @ A

    if transposed:
        X = X.mT

    return X.to(G.dtype)


class SimpleMuon(torch.optim.Optimizer):
    """
    Simplified Muon optimizer for single-GPU nGPT experiments.

    Muon = MomentUm Orthogonalized by Newton-schulz

    For 2D weight matrices (attn, mlp):
      1. Apply momentum: g_t = momentum * g_{t-1} + (1-momentum) * grad
      2. Orthogonalize: U = newton_schulz(g_t)
      3. Update: W = W - lr * U

    For 1D parameters (biases, embeddings, alphas, scalars):
      Use standard AdamW

    Args:
        params: Model parameters
        lr: Learning rate for Muon updates
        momentum: Momentum coefficient for 2D weights
        nesterov: Whether to use Nesterov momentum
        weight_decay: L2 penalty
        adam_lr_ratio: LR multiplier for Adam (relative to Muon LR)
        adam_betas: Beta coefficients for Adam
        adam_eps: Epsilon for Adam
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=False,
                 weight_decay=0.0, adam_lr_ratio=1.0, adam_betas=(0.9, 0.999), adam_eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       weight_decay=weight_decay,
                       adam_lr_ratio=adam_lr_ratio, adam_betas=adam_betas, adam_eps=adam_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # For 2D weights (attn, mlp): use Muon (orthogonalized momentum)
                if p.ndim >= 2 and hasattr(p, 'label') and p.label in ['attn', 'mlp']:
                    # Initialize momentum buffer
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']
                    # Momentum update: buf = momentum * buf + (1 - momentum) * grad
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                    if nesterov:
                        update = grad + momentum * buf
                    else:
                        update = buf

                    # Orthogonalize the update
                    update_ortho = newton_schulz_orthogonalize(update)

                    # Apply weight decay (decoupled, applied before update)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # Apply orthogonalized update
                    p.add_(update_ortho, alpha=-lr)

                # For 1D parameters (biases, alphas, scalars) and embeddings: use AdamW
                else:
                    # Initialize Adam state
                    if 'adam_step' not in state:
                        state['adam_step'] = 0
                        state['adam_exp_avg'] = torch.zeros_like(grad)
                        state['adam_exp_avg_sq'] = torch.zeros_like(grad)

                    adam_exp_avg = state['adam_exp_avg']
                    adam_exp_avg_sq = state['adam_exp_avg_sq']
                    state['adam_step'] += 1

                    beta1, beta2 = group['adam_betas']

                    # Update biased first/second moment
                    adam_exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    adam_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['adam_step']
                    bias_correction2 = 1 - beta2 ** state['adam_step']
                    step_size = (lr * group['adam_lr_ratio']) * (bias_correction2 ** 0.5) / bias_correction1

                    # Weight decay (AdamW style)
                    if weight_decay != 0:
                        p.mul_(1 - (lr * group['adam_lr_ratio']) * weight_decay)

                    # Update parameters
                    denom = adam_exp_avg_sq.sqrt().add_(group['adam_eps'])
                    p.addcdiv_(adam_exp_avg, denom, value=-step_size)

        return loss

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

class BaselineBlock(nn.Module):
    """Baseline nGPT block with scalar alpha."""
    def __init__(self, n_embd, n_head, alpha_init=0.28):
        super().__init__()
        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        self.alpha_attn = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_mlp = nn.Parameter(torch.tensor(alpha_init))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'

    def forward(self, x):
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x

class GatedBlock(nn.Module):
    """H3: Gated residual connections."""
    def __init__(self, n_embd, n_head, alpha_init=0.28, gate_init=0.0):
        super().__init__()
        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        self.gate_attn = nn.Parameter(torch.full((n_embd,), gate_init))
        self.gate_mlp = nn.Parameter(torch.full((n_embd,), gate_init))

        self.alpha_attn = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_mlp = nn.Parameter(torch.tensor(alpha_init))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'
        self.gate_attn.label = 'gate'
        self.gate_mlp.label = 'gate'

    def forward(self, x):
        attn_out = self.attn(x)
        gate_a = torch.sigmoid(self.gate_attn)
        x = normalize_ngpt(x + gate_a * self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        gate_m = torch.sigmoid(self.gate_mlp)
        x = normalize_ngpt(x + gate_m * self.alpha_mlp * mlp_out)

        return x

class AsymmetricAlphaBlock(nn.Module):
    """H5: Different alphas for attention vs MLP."""
    def __init__(self, n_embd, n_head, alpha_attn=0.20, alpha_mlp=0.35):
        super().__init__()
        self.attn = SimpleAttention(n_embd, n_head)
        self.mlp = SimpleMLP(n_embd)

        self.alpha_attn = nn.Parameter(torch.tensor(alpha_attn))
        self.alpha_mlp = nn.Parameter(torch.tensor(alpha_mlp))

        self.alpha_attn.label = 'alpha'
        self.alpha_mlp.label = 'alpha'

    def forward(self, x):
        attn_out = self.attn(x)
        x = normalize_ngpt(x + self.alpha_attn * attn_out)

        mlp_out = self.mlp(x)
        x = normalize_ngpt(x + self.alpha_mlp * mlp_out)

        return x

class SimpleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config['block_size']
        self.n_embd = config['n_embd']

        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])

        # Choose block type based on experiment
        if args.gated_residuals:
            print(f"Using GatedBlock (H3) with gate_init={args.gate_init}")
            BlockClass = GatedBlock
            block_kwargs = {'alpha_init': args.alpha, 'gate_init': args.gate_init}
        elif args.asymmetric_alpha:
            print(f"Using AsymmetricAlphaBlock (H5): attn={args.alpha_attn}, mlp={args.alpha_mlp}")
            BlockClass = AsymmetricAlphaBlock
            block_kwargs = {'alpha_attn': args.alpha_attn, 'alpha_mlp': args.alpha_mlp}
        else:
            BlockClass = BaselineBlock
            block_kwargs = {'alpha_init': args.alpha}

        # Layer-specific alpha (H1)
        if args.layer_alpha_schedule:
            print(f"Using layer-specific alpha schedule: {args.layer_alpha_schedule}")
            alphas = get_layer_alpha_schedule(config['n_layer'], args.alpha, args.layer_alpha_schedule)
            self.blocks = nn.ModuleList([
                BaselineBlock(config['n_embd'], config['n_head'], alpha_init=alphas[i].item())
                for i in range(config['n_layer'])
            ])
        else:
            self.blocks = nn.ModuleList([
                BlockClass(config['n_embd'], config['n_head'], **block_kwargs)
                for _ in range(config['n_layer'])
            ])

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # H2: Learnable logit scale
        if args.learnable_logit_scale:
            print(f"Using learnable logit scale (H2): init={args.logit_scale}")
            self.logit_scale = nn.Parameter(torch.tensor(args.logit_scale))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(args.logit_scale))
            self.logit_scale.requires_grad = False

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

def get_layer_alpha_schedule(num_layers, base_alpha, schedule_type):
    """Generate layer-specific alpha values (H1)."""
    if schedule_type == 'linear':
        alphas = torch.linspace(0.7 * base_alpha, 1.3 * base_alpha, num_layers)
    elif schedule_type == 'upsidedown_u':
        t = torch.linspace(0, 1, num_layers)
        alphas = base_alpha * (1.0 + 0.5 * torch.sin(t * 3.14159))
    elif schedule_type == 'downsideup_u':
        t = torch.linspace(0, 1, num_layers)
        alphas = base_alpha * (1.0 - 0.3 * torch.sin(t * 3.14159))
    else:
        alphas = torch.full((num_layers,), base_alpha)
    return alphas

# -----------------------------------------------------------------------------
# Data loading

def load_fineweb_shard(shard_path):
    """Load a single FineWeb binary shard."""
    with open(shard_path, 'rb') as f:
        f.read(256)  # Skip header
        tokens = np.fromfile(f, dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int64))

def load_fineweb_data(data_dir, num_train_shards=2):
    """Load FineWeb data (subset for faster experiments)."""
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

    # Load data (small subset)
    print("\nLoading FineWeb data (subset for fast experiments)...")
    data_dir = 'data/fineweb10B'
    train_data, val_data = load_fineweb_data(data_dir, num_train_shards=4)  # 4 shards for proper comparison
    vocab_size = 50257
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")

    # Model configuration (modded-nanogpt size)
    config = {
        'vocab_size': vocab_size,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': 6,  # Fixed for 768 dim
        'block_size': 128,
    }

    print(f"\nModel config (modded-nanogpt size):")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = SimpleGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Optimizer
    weight_params = [p for p in model.parameters() if getattr(p, 'label', None) in ['attn', 'mlp']]
    other_params = [p for p in model.parameters() if getattr(p, 'label', None) not in ['attn', 'mlp']]

    if args.optimizer == 'muon':
        # SimpleMuon: handles both 2D weights (Muon) and 1D params (Adam) internally
        all_params = list(model.parameters())
        optimizer = SimpleMuon(
            all_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=0.0,  # No weight decay for nGPT (weights are normalized)
            adam_lr_ratio=0.3   # 1D params use 0.3x learning rate
        )
        print(f"\nOptimizer: SimpleMuon (lr={args.lr}, momentum={args.momentum})")
        print(f"  2D weight params (Muon): {sum(p.numel() for p in weight_params):,}")
        print(f"  1D params (Adam): {sum(p.numel() for p in other_params):,}")
    else:
        # AdamW for baseline
        optimizer = torch.optim.AdamW([
            {'params': weight_params, 'lr': args.lr},
            {'params': other_params, 'lr': args.lr * 0.3}
        ])
        print(f"\nOptimizer: AdamW (lr={args.lr})")
        print(f"  Weight params: {sum(p.numel() for p in weight_params):,}")
        print(f"  Other params: {sum(p.numel() for p in other_params):,}")

    # Training
    print(f"\nStarting training ({args.steps} steps)...")
    print("-" * 80)

    losses = []
    proj_count = 0

    # H6: Progressive radius
    current_radius = 1.0
    target_radius = args.target_radius if args.progressive_radius else 1.0

    for step in range(args.steps):
        # H6: Gradually adjust radius
        if args.progressive_radius and step > 0:
            # Linear interpolation toward target radius
            progress = min(step / args.steps, 1.0)
            current_radius = 1.0 + (target_radius - 1.0) * progress

        # Training step
        x, y = get_batch(train_data, config['block_size'], args.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Lazy projection
        if step % args.lazy_proj_freq == 0:
            project_weights_to_hypersphere(model, radius=current_radius)
            proj_count += 1

        losses.append(loss.item())

        if step % 50 == 0:
            print(f"Step {step:4d}: loss = {loss.item():.4f}", end='')
            if args.progressive_radius:
                print(f", radius = {current_radius:.3f}", end='')
            print()

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
        'name': args.name,
        'hypothesis': args.hypothesis,
        'alpha': args.alpha,
        'logit_scale': args.logit_scale,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_params': n_params,
        'lr': args.lr,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'lazy_proj_freq': args.lazy_proj_freq,
        'layer_alpha_schedule': args.layer_alpha_schedule,
        'gated_residuals': args.gated_residuals,
        'gate_init': args.gate_init if args.gated_residuals else None,
        'asymmetric_alpha': args.asymmetric_alpha,
        'alpha_attn': args.alpha_attn if args.asymmetric_alpha else None,
        'alpha_mlp': args.alpha_mlp if args.asymmetric_alpha else None,
        'progressive_radius': args.progressive_radius,
        'target_radius': target_radius if args.progressive_radius else 1.0,
        'learnable_logit_scale': args.learnable_logit_scale,
        'projection_count': proj_count,
        'final_train_loss': losses[-1],
        'val_loss': val_loss,
        'train_loss_reduction': (losses[0] - losses[-1]) / losses[0] * 100,
        'time_seconds': elapsed,
        'tokens_per_sec': (args.steps * args.batch_size * config['block_size']) / elapsed
    }

    with open(args.output, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"\nResults saved to {args.output}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Throughput: {result['tokens_per_sec']:,.0f} tokens/sec")
    print(f"Projections: {proj_count}/{args.steps}")

    return result

if __name__ == '__main__':
    result = main()
