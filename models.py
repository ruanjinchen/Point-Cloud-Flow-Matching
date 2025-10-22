
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Time / condition embeddings ----------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal embedding of continuous time t in [0,1].
    Args:
        t: shape (...,) scalar per batch (we'll broadcast)
        dim: embedding dimension (even)
        max_period: controls min frequency
    Returns:
        emb: shape (..., dim)
    """
    assert dim % 2 == 0, "timestep_embedding dim must be even"
    half = dim // 2
    # Map t from [0,1] to [0, max_period] by simple scaling so frequencies are nicely spread.
    # Using log scale like diffusion.
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    # ensure t shape (..., 1)
    t = t.reshape(*t.shape, 1)
    args = t * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 4, act=nn.SiLU, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

        # Kaiming init for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, width: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.affine = nn.Linear(emb_dim, width * 2)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        h: (B*N, C)
        emb: (B, E) -> will be broadcast to (B*N, E) by indexing beforehand
        """
        h = self.norm(h)
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return h


class VelocityNet(nn.Module):
    """
    Per-point velocity field v_theta(x, t, cond) in R^{point_dim}.
    支持 point_dim=3（仅几何）或 point_dim=6（几何+颜色）。
    """
    def __init__(self, cond_dim: int, width: int = 512, depth: int = 6, emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1, point_dim: int = 3):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.emb_dim = int(emb_dim)
        self.cfg_dropout_p = float(cfg_dropout_p)
        self.point_dim = int(point_dim)

        # encoders
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        # trunk
        in_dim = self.point_dim + emb_dim
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth - 1)])
        self.films  = nn.ModuleList([FiLMBlock(width, emb_dim) for _ in range(depth - 1)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, self.point_dim))

        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu"); nn.init.zeros_(self.input.bias)
        for m in self.blocks:
            for l in m:
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
        for l in self.out:
            if isinstance(l, nn.Linear):
                nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, point_dim)
        t: (B,) in [0,1]
        cond: (B, C) or None
        """
        B, N, D = x.shape
        assert D == self.point_dim, f"VelocityNet expected point_dim={self.point_dim}, got {D}"
        if t.dim() == 1: t = t[:, None]
        t_emb = torch.nn.functional.silu(self.t_proj(timestep_embedding(t.squeeze(-1), self.emb_dim).to(x.dtype)))

        if self.cond_dim > 0 and cond is not None:
            if cond_drop_mask is not None:
                cond = cond * (1.0 - cond_drop_mask)  # mask=1 -> drop
            c_in = cond
        else:
            c_in = x.new_zeros((B, self.cond_dim if self.cond_dim > 0 else 1))
        c_emb = torch.nn.functional.silu(self.c_proj(c_in))
        emb = t_emb + c_emb

        # broadcast emb to (B*N, E)
        emb_bn = emb[:, None, :].expand(B, N, -1).reshape(B * N, -1)

        h = torch.cat([x, emb[:, None, :].expand(B, N, -1)], dim=-1).reshape(B * N, -1)
        h = self.input(h)
        for blk, fim in zip(self.blocks, self.films):
            h = fim(h, emb_bn)
            h = h + blk(h)
        v = self.out(h).reshape(B, N, self.point_dim)
        return v

    @torch.no_grad()
    def guided_velocity(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                        guidance_scale: float = 0.0) -> torch.Tensor:
        """
        Classifier-free guidance: v = v(cond) + s * (v(cond) - v(uncond))
        If s==0 -> pure conditional branch.
        """
        if guidance_scale <= 0.0 or cond is None or self.cond_dim == 0:
            return self.forward(x, t, cond, cond_drop_mask=None)
        # cond branch
        v_c = self.forward(x, t, cond, cond_drop_mask=None)
        # uncond branch (drop all cond)
        mask = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        v_u = self.forward(x, t, cond, cond_drop_mask=mask)
        return v_c + guidance_scale * (v_c - v_u)

class ShapeEncoder(nn.Module):
    """PointNet-lite encoder -> z (angle-invariant target with aux losses)
       支持 in_channels=3(xyz) 或 6(xyz+rgb)。"""
    def __init__(self, latent_dim: int = 256, width: int = 128, depth: int = 4, in_channels: int = 3):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        ch = width
        # 第一层 Linear 的输入维度从 3 改为 in_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, ch), nn.SiLU(),
            nn.Linear(ch, ch), nn.SiLU(),
            nn.Linear(ch, ch), nn.SiLU(),
        )
        heads = []
        in_d = ch
        for _ in range(max(1, depth - 3)):
            heads += [nn.Linear(in_d, ch), nn.SiLU()]
            in_d = ch
        heads += [nn.Linear(in_d, latent_dim)]
        self.head = nn.Sequential(*heads)
        for m in list(self.mlp) + list(self.head):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, pts_or_feats: torch.Tensor):
        """
        输入：
          - 当 in_channels=3：pts_or_feats = (B,N,3) 的 xyz
          - 当 in_channels=6：pts_or_feats = (B,N,6) 的 [xyz | rgb]，rgb 已归一化到 [0,1]
        返回：
          z: (B, D)
          feats: (B, N, C) 中间点特征
        """
        h = self.mlp(pts_or_feats)   # (B, N, C)
        g = h.max(dim=1).values      # (B, C)
        z = self.head(g)             # (B, D)
        return z, h

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd: float):
    return GradReverse.apply(x, float(lambd))

class CondAdversary(nn.Module):
    """predict joints from z (for GRL adversarial removal of joint info)"""
    def __init__(self, z_dim: int, cond_dim: int, width: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = z_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, cond_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    t = t.reshape(*t.shape, 1)
    args = t * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ConditionalLatentVelocityNet(nn.Module):
    """
    v_phi(y, t, cond) in latent space.
    - y: (B, Dz)
    - t: (B,) or (B,1)
    - cond: (B, C) or None
    """
    def __init__(self, latent_dim: int, cond_dim: int, width: int = 512, depth: int = 6, emb_dim: int = 256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.emb_dim = int(emb_dim)

        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        in_dim = latent_dim + emb_dim
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth-1)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, latent_dim))

        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu"); nn.init.zeros_(self.input.bias)
        for blk in self.blocks:
            for l in blk:
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
        for l in self.out:
            if isinstance(l, nn.Linear):
                nn.init.zeros_(l.bias)

    def forward(self, y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None, cond_drop_p: float = 0.0) -> torch.Tensor:
        if t.dim() == 1: t = t[:, None]
        t_emb = timestep_embedding(t.squeeze(-1), self.emb_dim).to(y.dtype)
        t_emb = torch.nn.functional.silu(self.t_proj(t_emb))
        if self.cond_dim > 0 and cond is not None:
            if cond_drop_p > 0.0:
                drop = (torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype) < cond_drop_p).to(y.dtype)
                cond = cond * (1.0 - drop)
            c_in = cond
        else:
            c_in = y.new_zeros((y.shape[0], self.cond_dim if self.cond_dim > 0 else 1))
        c_emb = torch.nn.functional.silu(self.c_proj(c_in))
        emb = t_emb + c_emb

        h = torch.cat([y, emb], dim=-1)
        h = self.input(h)
        for blk in self.blocks:
            h = h + blk(h)
        v = self.out(h)
        return v

    @torch.no_grad()
    def euler_sample(self, y0: torch.Tensor, cond: torch.Tensor | None, steps: int = 50, guidance_scale: float = 0.0) -> torch.Tensor:
        """
        Classifier-free style guidance: v = v(cond) + s * (v(cond) - v(uncond))
        """
        y = y0
        dt = 1.0 / steps
        for i in range(steps):
            t = y.new_full((y.shape[0],), (i + 0.5) * dt)
            v_c = self.forward(y, t, cond, cond_drop_p=0.0)
            if guidance_scale > 0.0 and self.cond_dim > 0 and cond is not None:
                v_u = self.forward(y, t, None, cond_drop_p=1.0)
                v = v_c + guidance_scale * (v_c - v_u)
            else:
                v = v_c
            y = y + v * dt
        return y

