from __future__ import annotations
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
here = os.path.dirname(os.path.abspath(__file__))
tp = os.path.join(here, "third_party", "pvcnn")
if os.path.isdir(tp) and (tp not in sys.path):
    sys.path.insert(0, tp)
from modules.pvconv import PVConv
from modules.shared_mlp import SharedMLP


# ======================================
# =============== 基础组件 ==============
# ======================================

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
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    t = t.reshape(*t.shape, 1)  # ensure (...,1)
    args = t * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 4,
                 act=nn.SiLU, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        # Kaiming init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    """
    用于逐点 MLP 的层间 FiLM 调制（零初始化 -> 恒等起步）。
    """
    def __init__(self, width: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.affine = nn.Linear(emb_dim, width * 2)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        h: (B*N, C)
        emb: (B, E) -> 需在外部 broadcast 到 (B*N, E)
        """
        h = self.norm(h)
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class VelocityNet(nn.Module):
    """
    逐点 MLP：v_theta(x, t, cond) in R^{point_dim}
    - point_dim=3：几何
    - point_dim=6：几何+颜色
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
        B, N, D = x.shape
        assert D == self.point_dim, f"VelocityNet expected point_dim={self.point_dim}, got {D}"
        if t.dim() == 1:
            t = t[:, None]
        t_emb = torch.nn.functional.silu(
            self.t_proj(timestep_embedding(t.squeeze(-1), self.emb_dim).to(x.dtype))
        )
        if self.cond_dim > 0 and cond is not None:
            if cond_drop_mask is not None:
                cond = cond * (1.0 - cond_drop_mask)  # mask=1 -> drop
            c_in = cond
        else:
            c_in = x.new_zeros((B, self.cond_dim if self.cond_dim > 0 else 1))
        c_emb = torch.nn.functional.silu(self.c_proj(c_in))
        emb = t_emb + c_emb                       # (B,E)
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
        if guidance_scale <= 0.0 or cond is None or self.cond_dim == 0:
            return self.forward(x, t, cond, cond_drop_mask=None)
        v_c = self.forward(x, t, cond, cond_drop_mask=None)
        mask = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        v_u = self.forward(x, t, cond, cond_drop_mask=mask)
        return v_c + guidance_scale * (v_c - v_u)


class ShapeEncoder(nn.Module):
    """
    PointNet-lite encoder -> z
    支持 in_channels=3(xyz) / 6(xyz+rgb)
    """
    def __init__(self, latent_dim: int = 256, width: int = 128, depth: int = 4, in_channels: int = 3):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        ch = width
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
        h = self.mlp(pts_or_feats)           # (B, N, C)
        g = h.max(dim=1).values              # (B, C)
        z = self.head(g)                     # (B, D)
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
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth - 1)])
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


# =====================================================
# ===============  Hybrid 上下文 + 逐点 MLP  ===========
# =====================================================

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(a, 1)


def _choose_gn_groups(channels: int, prefer: int = 32) -> int:
    prefer = min(prefer, channels)
    g = _gcd(channels, prefer)
    if g == 1 and channels >= 16:
        for cand in [32, 16, 8, 4, 2]:
            if channels % cand == 0 and cand <= channels:
                return cand
    return g


def _make_norm(norm_type: str, channels: int, gn_groups: int):
    if norm_type == "group":
        return nn.GroupNorm(_choose_gn_groups(channels, gn_groups), channels)
    elif norm_type in ("batch", "syncbn"):
        return nn.BatchNorm1d(channels)
    else:
        return nn.Identity()


class _FiLM1d(nn.Module):
    """
    (B,C,N) 上的 FiLM：Norm -> (1+γ)·x + β；γ/β 零初始化，确保恒等起步。
    """
    def __init__(self, channels: int, emb_dim: int, norm_type: str = "group",
                 gn_groups: int = 32, one_plus: bool = True):
        super().__init__()
        self.norm = _make_norm(norm_type, channels, gn_groups)
        self.affine = nn.Linear(emb_dim, channels * 2)
        self.one_plus = bool(one_plus)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,N)    emb: (B,E)
        """
        B, C, N = x.shape
        y = self.norm(x)
        emb = emb.to(y.dtype)
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1); beta = beta.view(B, C, 1)
        if self.one_plus:
            return y * (1.0 + gamma) + beta
        else:
            return y * gamma + beta


class _PVBlock(nn.Module):
    """
    单个 PVConv Block：PVConv -> SharedMLP(1x1) -> FiLM -> 残差
    - normalize=True 可稳定体素邻域（强烈建议开启）
    """
    def __init__(self, channels: int, resolution: int, emb_dim: int, with_se: bool,
                 norm_type: str = "group", gn_groups: int = 32, voxel_normalize: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.pvconv = PVConv(channels, channels, kernel_size=3,
                             resolution=int(resolution), with_se=bool(with_se),
                             normalize=bool(voxel_normalize), eps=eps)
        self.post = SharedMLP(channels, [channels])
        self.film = _FiLM1d(channels, emb_dim, norm_type=norm_type, gn_groups=gn_groups, one_plus=True)

    def forward(self, feat_coords: Tuple[torch.Tensor, torch.Tensor], emb: torch.Tensor):
        f, c = self.pvconv(feat_coords)
        f, c = self.post((f, c))
        f = f + self.film(f, emb)
        return f, c


class _PVStage(nn.Module):
    """
    Stage: 1x1 SharedMLP 通道提升 -> k × PVBlock
    """
    def __init__(self, in_c: int, out_c: int, num_blocks: int, resolution: int, emb_dim: int,
                 with_se: bool, norm_type: str = "group", gn_groups: int = 32, voxel_normalize: bool = True):
        super().__init__()
        self.proj = SharedMLP(in_c, [out_c])
        self.blocks = nn.ModuleList([
            _PVBlock(out_c, resolution, emb_dim, with_se,
                     norm_type=norm_type, gn_groups=gn_groups, voxel_normalize=voxel_normalize)
            for _ in range(int(num_blocks))
        ])

    def forward(self, feat: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor):
        f, c = self.proj((feat, coords))
        for blk in self.blocks:
            f, c = blk((f, c), emb)
        return f, c


class ContextNet(nn.Module):
    """
    多尺度 PVConv 金字塔，输出每点 ctx 特征。
    - 输入：(B,N,3 or 6)
    - 输出：(B,N,ctx_dim)
    设计要点：
      * 多 stage（分辨率递减）：提升上下文感受野，利于细长与末端细节
      * 全程 FiLM(t,cond)，时间/条件贯穿所有 stage
      * 可选全局通道（max-pool -> MLP -> repeat），增强全局一致性
    """
    def __init__(self,
                 in_point_dim: int,              # 3 / 6
                 cond_dim: int,                  # z (+ joint)
                 emb_dim: int = 256,             # t/cond 嵌入维度
                 ctx_dim: int = 64,              # 输出的上下文维度
                 stage_channels: List[int] = (128, 256, 256),
                 stage_blocks:   List[int] = (2, 2, 2),
                 stage_res:      List[int] = (32, 16, 8),
                 with_se: bool = True,
                 norm_type: str = "group",
                 gn_groups: int = 32,
                 with_global: bool = True,
                 voxel_normalize: bool = True):
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(stage_res)
        self.in_point_dim = int(in_point_dim)
        self.emb_dim = int(emb_dim)
        self.ctx_dim = int(ctx_dim)
        self.with_global = bool(with_global)

        self.use_xyz = True
        self.use_rgb = (self.in_point_dim == 6)

        # t/cond -> emb
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        # stem 输入通道：emb + xyz(+rgb)
        stem_in_c = emb_dim + (3 if self.use_xyz else 0) + (3 if self.use_rgb else 0)

        # stages
        self.stages = nn.ModuleList()
        in_c = stem_in_c
        for sc, nb, rs in zip(stage_channels, stage_blocks, stage_res):
            self.stages.append(
                _PVStage(in_c, sc, nb, rs, emb_dim, with_se,
                         norm_type=norm_type, gn_groups=gn_groups, voxel_normalize=voxel_normalize)
            )
            in_c = sc

        # 可选全局通道
        if self.with_global:
            C_last = stage_channels[-1]
            self.global_mlp = nn.Sequential(
                nn.Linear(C_last, C_last), nn.SiLU(),
                nn.Linear(C_last, C_last)
            )
            for m in self.global_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        # 输出头：把多尺度末端特征 -> ctx_dim
        self.stage_channels = list(stage_channels)  # 记录以便 forward 使用
        head_in = sum(self.stage_channels) + (self.stage_channels[-1] if self.with_global else 0)
        self.head_pre = nn.Conv1d(head_in, self.stage_channels[-1], 1, bias=True)
        self.head_norm = _make_norm(norm_type, self.stage_channels[-1], gn_groups)
        self.head_act  = nn.SiLU()
        self.head_out  = nn.Conv1d(self.stage_channels[-1], ctx_dim, 1, bias=True)

        nn.init.kaiming_normal_(self.head_pre.weight, nonlinearity="relu")
        nn.init.zeros_(self.head_pre.bias)
        nn.init.zeros_(self.head_out.weight); nn.init.zeros_(self.head_out.bias)

        self.norm_type = norm_type
        self.gn_groups = int(gn_groups)

    def _t_emb(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return F.silu(self.t_proj(timestep_embedding(t.squeeze(-1), self.emb_dim).to(dtype)))

    def _c_emb(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None or cond.numel() == 0:
            c_in = x.new_zeros((x.shape[0], 1))
        else:
            c_in = cond
        return F.silu(self.c_proj(c_in))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B,N,3/6)  ->  ctx: (B,N,ctx_dim)
        """
        B, N, D = x.shape
        coords = x[..., :3].permute(0, 2, 1).contiguous()  # (B,3,N)
        # emb
        emb = self._t_emb(t, x.dtype) + self._c_emb(x, cond)  # (B,E)

        # stem feats = [emb | xyz | (rgb)]
        feats = [emb[:, :, None].expand(B, self.emb_dim, N)]
        if self.use_xyz:
            feats.append(coords)
        if self.use_rgb and D == 6:
            feats.append(x[..., 3:].permute(0, 2, 1).contiguous())
        feats = torch.cat(feats, dim=1)  # (B,C_in,N)

        # 为了稳定 PVConv 的数值，强制在 FP32 下计算
        with torch.amp.autocast("cuda", enabled=False):
            f, c = feats.float(), coords.float()
            emb32 = emb.float()

            ms_feats = []  # <<< 新增：收集每个阶段的特征 (B,C_i,N)
            for si, stage in enumerate(self.stages):
                f, c = stage(f, c, emb32)
                ms_feats.append(f)  # (B,C_i,N)

            # 可选全局分支：用最后一层通道
            if self.with_global:
                g = f.max(dim=-1).values               # (B,C_last)
                g = self.global_mlp(g)                 # (B,C_last)
                g = g[:, :, None].expand_as(f)         # (B,C_last,N)
                ms_feats.append(g)                     # 直接当作另一“尺度”通道堆叠

            f_cat = torch.cat(ms_feats, dim=1)         # (B, sum(C_i)+C_last(if global), N)

            h = self.head_pre(f_cat)
            h = self.head_act(self.head_norm(h))
            ctx32 = self.head_out(h)                   # (B,ctx_dim,N)

        return ctx32.permute(0, 2, 1).contiguous().to(x.dtype)


class VelocityNetWithContext(nn.Module):
    """
    把 [x_i | ctx_i | emb(t,cond)] -> v_i 的逐点 MLP 头。
    初始化/CFG/FiLM 与基础 VelocityNet 保持一致。
    """
    def __init__(self, cond_dim: int, point_dim: int = 3, ctx_dim: int = 64,
                 width: int = 512, depth: int = 6, emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1):
        super().__init__()
        self.cond_dim, self.point_dim = int(cond_dim), int(point_dim)
        self.emb_dim, self.ctx_dim = int(emb_dim), int(ctx_dim)
        self.cfg_dropout_p = float(cfg_dropout_p)

        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        in_dim = self.point_dim + self.ctx_dim + emb_dim
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth - 1)])
        self.films  = nn.ModuleList([FiLMBlock(width, emb_dim) for _ in range(depth - 1)])
        self.out    = nn.Sequential(nn.SiLU(), nn.Linear(width, self.point_dim))

        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu"); nn.init.zeros_(self.input.bias)
        for m in self.blocks:
            for l in m:
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
        for l in self.out:
            if isinstance(l, nn.Linear):
                nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                ctx: torch.Tensor, cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        assert ctx.shape[:2] == (B, N), f"ctx shape mismatch: {tuple(ctx.shape)} vs {(B,N,'*')}"
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

        emb_bn = emb[:, None, :].expand(B, N, -1).reshape(B * N, -1)
        h = torch.cat([x, ctx, emb[:, None, :].expand(B, N, -1)], dim=-1).reshape(B * N, -1)
        h = self.input(h)
        for blk, fim in zip(self.blocks, self.films):
            h = fim(h, emb_bn)
            h = h + blk(h)
        v = self.out(h).reshape(B, N, self.point_dim)
        return v


class HybridMLP(nn.Module):
    """
    新的点流骨干：
      - ContextNet(PVConv 金字塔) 产生每点上下文 ctx
      - VelocityNetWithContext 逐点回归速度
      - forward(x, t, cond, cond_drop_mask=None) -> v
      - guided_velocity(x, t, cond, guidance_scale=0.0)
      - set_bn_eval(freeze=True)  用于训练中期冻结 BN
    """
    def __init__(self,
                 cond_dim: int,
                 point_dim: int = 3,
                 # 上下文分支
                 ctx_dim: int = 64,
                 ctx_emb_dim: int = 256,
                 stage_channels: List[int] = (128, 256, 256),
                 stage_blocks:   List[int] = (2, 2, 2),
                 stage_res:      List[int] = (32, 16, 8),
                 with_se: bool = True,
                 norm_type: str = "group",
                 gn_groups: int = 32,
                 with_global: bool = True,
                 voxel_normalize: bool = True,
                 # 逐点 MLP 头
                 pf_width: int = 512,
                 pf_depth: int = 6,
                 pf_emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.point_dim = int(point_dim)

        # 上下文提取器
        self.ctx_net = ContextNet(
            in_point_dim=point_dim, cond_dim=cond_dim, emb_dim=ctx_emb_dim, ctx_dim=ctx_dim,
            stage_channels=list(stage_channels), stage_blocks=list(stage_blocks), stage_res=list(stage_res),
            with_se=with_se, norm_type=norm_type, gn_groups=gn_groups,
            with_global=with_global, voxel_normalize=voxel_normalize
        )
        # 逐点头
        self.head = VelocityNetWithContext(
            cond_dim=cond_dim, point_dim=point_dim, ctx_dim=ctx_dim,
            width=pf_width, depth=pf_depth, emb_dim=pf_emb_dim, cfg_dropout_p=cfg_dropout_p
        )

    @staticmethod
    def _cond_eff(cond: Optional[torch.Tensor], mask: Optional[torch.Tensor], x: torch.Tensor):
        if cond is None:
            return None if hasattr(x, "new_zeros") is False else x.new_zeros((x.shape[0], 1))
        if mask is None:
            return cond
        return cond * (1.0 - mask.to(cond.dtype))

    def set_bn_eval(self, freeze: bool = True):
        """
        冻结/解冻 BatchNorm
        - vendor SharedMLP/PVConv 内部若含 BN/SyncBN，会被设置 eval()。
        - GroupNorm 不受影响。
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval() if freeze else m.train()
                m.track_running_stats = True
                m.momentum = 0.0 if freeze else 0.1

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,N,3/6)  t: (B,)  cond: (B,C)
        return: v (B,N,3/6)
        """
        cond_eff = self._cond_eff(cond, cond_drop_mask, x)
        ctx = self.ctx_net(x, t, cond_eff if self.cond_dim > 0 else None)  # (B,N,ctx_dim)
        v = self.head(x, t, cond, ctx, cond_drop_mask=cond_drop_mask)
        return v

    @torch.no_grad()
    def guided_velocity(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                        guidance_scale: float = 0.0) -> torch.Tensor:
        if guidance_scale <= 0.0 or self.cond_dim == 0 or cond is None:
            return self.forward(x, t, cond, cond_drop_mask=None)
        zero = torch.zeros_like(cond)
        v_c = self.forward(x, t, cond, cond_drop_mask=None)
        v_u = self.forward(x, t, zero, cond_drop_mask=None)
        return v_c + guidance_scale * (v_c - v_u)
