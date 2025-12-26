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
parser.add_argument('--orthog-method', type=str, default='newton_schulz', choices=['newton_schulz', 'polar_express'], help='Orthogonalization method for Muon')
parser.add_argument('--geodesic-mode', type=str, default='baseline', choices=['baseline', 'geodesic_lr', 'geodesic_scaled'], help='Update mode')
parser.add_argument('--variance-reduction', action='store_true', help='Use NorMuon variance reduction')
parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for variance reduction')
parser.add_argument('--cautious-wd', action='store_true', help='Use cautious weight decay')
parser.add_argument('--target-loss', type=float, default=None, help='Target validation loss for early stopping')
parser.add_argument('--val-freq', type=int, default=50, help='Validation frequency (steps)')
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


# -----------------------------------------------------------------------------
# Polar Express Orthogonalization (PyTorch fallback implementation)

# PyTorch fallback functions (no Triton required)
def XXT_pytorch(A, out):
    """PyTorch fallback: Compute out = A @ A.T"""
    torch.mm(A, A.mT, out=out)
    return out

def ba_plus_cAA_pytorch(A, alpha, beta, out):
    """PyTorch fallback: Compute out = beta * A + alpha * A @ A.T"""
    # Note: A must be square for this operation
    AA = A @ A.mT
    out.copy_(beta * A + alpha * AA)
    return out

# Precomputed coefficients for 5 iterations
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

    PyTorch-only implementation (no Triton kernels required).
    Faster and more accurate than Newton-Schulz for orthogonalization.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    d = X.size(-2)
    A = torch.empty((d, d), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        # A = X @ X.mT
        XXT_pytorch(X, out=A)
        
        # B = b * A + c * A @ A
        ba_plus_cAA_pytorch(A, alpha=c, beta=b, out=B)

        # C = a * X + B @ X
        torch.mm(B, X, out=C)
        C.add_(X, alpha=a)

        X, C = C, X  # Swap references

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


def cautious_wd_and_update_inplace(p, v, wd, lr):
    """Cautious weight decay + parameter update. wd and lr are scalars."""
    mask = (v * p) >= 0
    p.mul_(1.0 - mask * wd * lr)
    p.add_(v, alpha=-lr)


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
                 weight_decay=0.0, adam_lr_ratio=1.0, adam_betas=(0.9, 0.999), adam_eps=1e-8,
                 geodesic_mode='baseline', orthog_method='newton_schulz',
                 variance_reduction=False, beta2=0.95, cautious_wd=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       weight_decay=weight_decay,
                       adam_lr_ratio=adam_lr_ratio, adam_betas=adam_betas, adam_eps=adam_eps,
                       geodesic_mode=geodesic_mode, orthog_method=orthog_method,
                       variance_reduction=variance_reduction, beta2=beta2, cautious_wd=cautious_wd)
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
                    orthog_method = group['orthog_method']
                    if orthog_method == 'polar_express':
                        update_ortho = polar_express(update)
                    else:  # newton_schulz
                        update_ortho = newton_schulz_orthogonalize(update)

                    # Variance reduction
                    if group['variance_reduction']:
                        if 'second_momentum_buffer' not in state:
                            # Low-rank variance buffer
                            # Match the reduction dimension logic
                            red_dim = -1 if update_ortho.size(-2) >= update_ortho.size(-1) else -2
                            # Buffer size matches the other dimension
                            buffer_shape = list(update_ortho.shape)
                            buffer_shape[red_dim] = 1
                            state['second_momentum_buffer'] = torch.zeros(buffer_shape, device=update_ortho.device, dtype=update_ortho.dtype)
                        
                        red_dim = -1 if update_ortho.size(-2) >= update_ortho.size(-1) else -2
                        update_ortho = apply_normuon_variance_reduction(
                            update_ortho, state['second_momentum_buffer'], group['beta2'], red_dim
                        )

                    # Apply update using geodesic or baseline method
                    geodesic_mode = group['geodesic_mode']

                    if group['cautious_wd']:
                        # Cautious weight decay + update (baseline only for now)
                        if geodesic_mode == 'baseline':
                            cautious_wd_and_update_inplace(p, update_ortho, weight_decay, lr)
                        else:
                            # Fallback for geodesic (not typically used with cautious WD in modded-nanogpt)
                            if weight_decay != 0:
                                p.mul_(1 - lr * weight_decay)
                            p.add_(update_ortho, alpha=-lr)
                    else:
                        # Standard update
                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)

                        if geodesic_mode == 'baseline':
                            # Baseline: W_new = W - lr * U (will be projected later)
                            p.add_(update_ortho, alpha=-lr)

                        elif geodesic_mode == 'geodesic_lr':
                            # Geodesic: W_new = W * cos(lr) + U * sin(lr)
                            theta = torch.tensor(lr, dtype=p.dtype, device=p.device)
                            cos_theta = torch.cos(theta)
                            sin_theta = torch.sin(theta)
                            p_new = p * cos_theta + update_ortho * sin_theta
                            p.copy_(p_new)

                        elif geodesic_mode == 'geodesic_scaled':
                            # Geodesic scaled: W_new = W * cos(lr*||U||) + U * sin(lr*||U||)
                            U_norm = update_ortho.norm()
                            theta = lr * U_norm
                            theta_tensor = torch.tensor(theta, dtype=p.dtype, device=p.device)
                            cos_theta = torch.cos(theta_tensor)
                            sin_theta = torch.sin(theta_tensor)
                            p_new = p * cos_theta + update_ortho * sin_theta
                            p.copy_(p_new)

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
            adam_lr_ratio=0.3,   # 1D params use 0.3x learning rate
            geodesic_mode=args.geodesic_mode,
            orthog_method=args.orthog_method,
            variance_reduction=args.variance_reduction,
            beta2=args.beta2,
            cautious_wd=args.cautious_wd
        )
        print(f"\nOptimizer: SimpleMuon (lr={args.lr}, momentum={args.momentum}, geodesic={args.geodesic_mode}, orthog={args.orthog_method})")
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

    # Early stopping tracking
    time_to_target = None
    reached_target = False

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

        # Periodic validation with early stopping
        if args.target_loss is not None and step > 0 and step % args.val_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):  # 20 batches for stable estimate
                    x, y = get_batch(val_data, config['block_size'], args.batch_size, device)
                    _, loss_val = model(x, y)
                    val_losses.append(loss_val.item())
            val_loss_current = np.mean(val_losses)
            model.train()

            print(f"  [Val @ step {step}]: train={losses[-1]:.4f}, val={val_loss_current:.4f}")

            if val_loss_current < args.target_loss:
                time_to_target = time.time() - t0
                reached_target = True
                print(f"\n{'='*80}")
                print(f"TARGET REACHED: {val_loss_current:.4f} < {args.target_loss} at step {step}")
                print(f"Time to target: {time_to_target:.1f}s")
                print(f"{'='*80}")
                break

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
        'optimizer': args.optimizer,
        'orthog_method': args.orthog_method,
        'geodesic_mode': args.geodesic_mode,
        'variance_reduction': args.variance_reduction,
        'cautious_wd': args.cautious_wd,
        'projection_count': proj_count,
        'target_loss': args.target_loss,
        'reached_target': reached_target,
        'time_to_target': time_to_target if reached_target else float('inf'),
        'final_step': step,
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
