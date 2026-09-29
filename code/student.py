"""MP1 student model: Rotary320-class wide GPT + gated causal neural cache.

Config flags: pos_encoding, mlp/mlp_hidden, dropout, tie_weights, qk_norm,
cache (bool), cache_dim (int; 0/=width means no projection).
With cache=True, forward() returns normalized log-probs (log_softmax is
idempotent on them, so the trainer and scorer are unaffected).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def rope_cos_sin(length, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                              dtype=torch.float32) / head_dim))
    t = torch.arange(length, device=device, dtype=torch.float32)
    f = torch.outer(t, inv_freq)
    return f.cos()[None, None], f.sin()[None, None]


def apply_rope(x, cos, sin):
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, width, heads, mlp, mlp_hidden, dropout, qk_norm=False):
        super().__init__()
        self.heads = heads
        head_dim = width // heads
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        if mlp == 'swiglu':
            self.mlp = SwiGLU(width, mlp_hidden)
            self.down = self.mlp.w2
        else:
            self.mlp = nn.Sequential(nn.Linear(width, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, width))
            self.down = self.mlp[2]
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = self.k_norm = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            b, t, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, w)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class NeuralCache(nn.Module):
    """Strictly causal, within-window, state-free neural cache (Grave et al. 2017
    style, gated). Key j stores token j+1, so query i may only read keys j<i;
    the newest retrievable token is i (already in the observed prefix)."""

    def __init__(self, width, vocab, dim, gate_bias=-2.5):
        super().__init__()
        self.vocab = vocab
        self.proj = nn.Linear(width, dim, bias=False) if (dim and dim != width) else None
        self.temp = nn.Parameter(torch.zeros(1))
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, h, ids):
        t = h.shape[1]
        x = self.proj(h) if self.proj is not None else h
        x = F.normalize(x.float(), dim=-1)
        tau = 0.05 + 0.45 * torch.sigmoid(self.temp)
        scores = (x @ x.transpose(-1, -2)) / tau
        mask = torch.ones(t, t, dtype=torch.bool, device=h.device).triu(1)
        scores = scores.masked_fill(mask, float('-inf'))
        w = torch.softmax(scores, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)[..., :max(t - 1, 0)]   # key T-1 never visible
        vals = F.one_hot(ids[:, 1:], self.vocab).float()         # value at key j = token j+1
        p_cache = w @ vals                                       # row 0: no keys -> zeros
        g = torch.sigmoid(self.gate(h.float()))
        return p_cache, g


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, depth, heads = config['width'], config['depth'], config['heads']
        dropout = float(config.get('dropout', 0.0))
        self.pos_mode = config.get('pos_encoding', 'rope')
        self.rope_theta = float(config.get('rope_theta', 10000.0))
        mlp = config.get('mlp', 'swiglu')
        default_hidden = int(8 * width / 3) if mlp == 'swiglu' else 4 * width
        mlp_hidden = int(config.get('mlp_hidden', default_hidden))

        self.token = nn.Embedding(config['vocab'], width)
        if self.pos_mode == 'learned':
            self.pos = nn.Embedding(self.context, width)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(width, heads, mlp, mlp_hidden, dropout,
                   bool(config.get('qk_norm', False))) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.cache = NeuralCache(width, config['vocab'],
                                 int(config.get('cache_dim', 0) or 0)) \
            if config.get('cache', False) else None

        self.apply(self._init)
        with torch.no_grad():
            for blk in self.blocks:
                blk.proj.weight.mul_(1 / math.sqrt(2 * depth))
                blk.down.weight.mul_(1 / math.sqrt(2 * depth))
            if self.cache is not None:      # re-init after global _init
                nn.init.zeros_(self.cache.gate.weight)
                nn.init.constant_(self.cache.gate.bias, -2.5)
        if config.get('tie_weights', False):
            self.head.weight = self.token.weight
        else:
            nn.init.zeros_(self.head.weight)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        cos = sin = None
        if self.pos_mode == 'learned':
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.emb_drop(x)
        if self.pos_mode == 'rope':
            hd = self.config['width'] // self.config['heads']
            cos, sin = rope_cos_sin(ids.shape[1], hd, self.rope_theta, ids.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """cache off: unnormalized logits; cache on: normalized log-probs
        (log_softmax in the trainer/scorer is idempotent on normalized log p)."""
        h = self.features(ids)
        logits = self.head(h)
        if self.cache is None:
            return logits
        p_lm = F.softmax(logits.float(), dim=-1)
        p_cache, g = self.cache(h, ids)
        p = (1.0 - g) * p_lm + g * p_cache
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.log(p.clamp_min(1e-12))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)