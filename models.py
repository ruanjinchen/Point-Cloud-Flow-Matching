
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

# ---------- PVCNN-based point-flow backbone ----------
class PVCNNVelocityNet(nn.Module):
    """
    完整版 PVCNN 点流骨干：
      - 多 stage / 多尺度（每个 stage: proj → [PVConv+SharedMLP+FiLM]×k）
      - 逐层 FiLM（time/cond 嵌入逐层调制通道）
      - 全局分支（global pooling → MLP → repeat → 与点特征级联）
      - 稳定化：GroupNorm/BatchNorm/SyncBN + 可选冻结 BN
      - 零初始化回归头（Conv1d -> SiLU -> Conv1d(零初始化)）
    接口：
      forward(x: (B,N,D), t: (B,), cond: (B,C), cond_drop_mask?: (B,1)) -> (B,N,D)
    """
    def __init__(self,
                 cond_dim: int,
                 point_dim: int = 3,
                 emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1,
                 # 兼容旧参数（当未显式提供分段配置时会自动推导 3-stage）
                 pv_width: int = 128,
                 pv_blocks: int = 3,
                 pv_resolution: int = 32,
                 pv_with_se: bool = True,
                 use_xyz_feat: bool = True,
                 # 多 stage 配置（若为 None 则用上面的旧参数自动推导）
                 pv_channels: Optional[list[int]] = None,
                 pv_blocks_per_stage: Optional[list[int]] = None,
                 pv_resolutions: Optional[list[int]] = None,
                 # 归一化/FiLM/全局分支/头部细节
                 norm_type: str = "group",        # ["group","batch","syncbn","none"]
                 gn_groups: int = 32,
                 film_one_plus: bool = False,     # False: y*γ+β（zero-init=恒等残差）；True: y*(1+γ)+β
                 with_global: bool = True,
                 head_drop: float = 0.0,
                 head_zero_init: bool = True,
                 voxel_normalize: bool = False):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.point_dim = int(point_dim)
        self.emb_dim = int(emb_dim)
        self.cfg_dropout_p = float(cfg_dropout_p)
        self.use_xyz_feat = bool(use_xyz_feat)
        self.use_rgb_feat = (self.point_dim == 6)
        self.norm_type = norm_type
        self.gn_groups = int(gn_groups)
        self.film_one_plus = bool(film_one_plus)
        self.with_global = bool(with_global)
        self.head_drop = float(head_drop)
        self.head_zero_init = bool(head_zero_init)
        self.voxel_normalize = bool(voxel_normalize)
        # --------- time/cond embedding ---------
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        # --------- 引入官方 PVCNN 模块 ---------
        try:
            import os, sys
            here = os.path.dirname(os.path.abspath(__file__))
            tp = os.path.join(here, "third_party", "pvcnn")
            if os.path.isdir(tp) and (tp not in sys.path):
                sys.path.insert(0, tp)
            from modules.pvconv import PVConv
            from modules.shared_mlp import SharedMLP
            self._PVConv = PVConv
            self._SharedMLP = SharedMLP
        except Exception as e:
            raise ImportError(f"Cannot import PVConv/SharedMLP: {e}. "
                              f"Make sure 'third_party/pvcnn/modules' is available with CUDA build.")

        # --------- 组网配置：多 stage 自动推导（兼容旧命令行） ---------
        if pv_channels is None or pv_blocks_per_stage is None or pv_resolutions is None:
            baseC = int(pv_width)
            pv_channels = [baseC, baseC * 2, baseC * 2]
            pv_blocks_per_stage = [max(1, pv_blocks // 3)] * 3 if pv_blocks >= 3 else [1, 1, 1]
            r = int(pv_resolution)
            pv_resolutions = [max(1, r), max(1, r // 2), max(1, r // 4)]
        assert len(pv_channels) == len(pv_blocks_per_stage) == len(pv_resolutions), \
            "pv_channels / pv_blocks_per_stage / pv_resolutions 长度必须一致"
        self.num_stages = len(pv_channels)
        self.stage_channels = list(map(int, pv_channels))
        self.stage_blocks = list(map(int, pv_blocks_per_stage))
        self.stage_res = list(map(int, pv_resolutions))
        C_last = self.stage_channels[-1]

        # --------- Stem 输入通道：emb + xyz(可选) + rgb(可选) ---------
        stem_in_c = self.emb_dim + (3 if self.use_xyz_feat else 0) + (3 if self.use_rgb_feat else 0)

        # --------- 构建多 stage ---------
        self.stages = nn.ModuleList()
        in_c = stem_in_c
        for sc, nb, rs in zip(self.stage_channels, self.stage_blocks, self.stage_res):
            self.stages.append(_PVStage(in_c, sc, nb, rs, self.emb_dim, pv_with_se,
                                        self._PVConv, self._SharedMLP,
                                        norm_type=self.norm_type, gn_groups=self.gn_groups,
                                        film_one_plus=self.film_one_plus,
                                        voxel_normalize=self.voxel_normalize))
            in_c = sc  # 下一 stage 的输入通道

        # --------- 全局分支（末端） ---------
        if self.with_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(C_last, C_last), nn.SiLU(),
                nn.Linear(C_last, C_last)
            )
            for m in self.global_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        # --------- 头部（线性回归，最后一层零初始化） ---------
        head_in = C_last * (2 if self.with_global else 1)
        self.head_pre = nn.Conv1d(head_in, C_last, 1, bias=True)
        self.head_norm = _make_norm(self.norm_type, C_last, self.gn_groups)
        self.head_act = nn.SiLU()
        self.head_drop_layer = nn.Dropout(self.head_drop) if self.head_drop > 0 else nn.Identity()
        self.head_out = nn.Conv1d(C_last, self.point_dim, 1, bias=True)

        nn.init.kaiming_normal_(self.head_pre.weight, nonlinearity="relu")
        nn.init.zeros_(self.head_pre.bias)
        if self.head_zero_init:
            nn.init.zeros_(self.head_out.weight); nn.init.zeros_(self.head_out.bias)
        else:
            nn.init.normal_(self.head_out.weight, std=1e-3); nn.init.zeros_(self.head_out.bias)

        # --------- 若选择 GroupNorm：把 vendor 模块里的 BN 换成 GN ---------
        if self.norm_type == "group":
            _replace_bn_with_gn_(self, groups=self.gn_groups)

    # ------- 内部：时间/条件嵌入 -------
    def _timestep_emb(self, t: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 1: t = t[:, None]
        emb = timestep_embedding(t.squeeze(-1), self.emb_dim).to(x_dtype)  # 来自本文件的函数
        return F.silu(self.t_proj(emb))

    def _cond_emb(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_dim > 0 and cond is not None:
            c_in = cond
        else:
            c_in = x.new_zeros((x.shape[0], self.cond_dim if self.cond_dim > 0 else 1))
        return F.silu(self.c_proj(c_in))

    # ------- 公开：冻结/解冻全部 BN（训练中期收敛更稳） -------
    def set_bn_eval(self, freeze: bool = True):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval() if freeze else m.train()
                m.track_running_stats = True
                m.momentum = 0.0 if freeze else 0.1

    # ------- 前向 -------
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        if self.cond_dim > 0 and cond is not None and cond_drop_mask is not None:
            # 原：cond = cond * (1.0 - cond_drop_mask)
            cond = cond * (1.0 - cond_drop_mask.to(cond.dtype))  # <<< dtype 对齐，防止 AMP 下类型混淆

        emb = self._timestep_emb(t, x.dtype) + self._cond_emb(x, cond)   # (B,E)

        coords = x[..., :3].permute(0, 2, 1).contiguous()    # (B,3,N)
        feats = [emb[:, :, None].expand(B, self.emb_dim, N)]
        if self.use_xyz_feat:
            feats.append(coords)
        if self.use_rgb_feat and D == 6:
            feats.append(x[..., 3:].permute(0, 2, 1).contiguous())
        feats = torch.cat(feats, dim=1)  # (B,C_in,N)

        # 原：with torch.cuda.amp.autocast(enabled=False):
        # 新 API（去掉 FutureWarning）
        with torch.amp.autocast("cuda", enabled=False):
            f, c = feats.float(), coords.float()
            emb32 = emb.float()  # <<< 关键：把 FiLM 的条件向量也切到 FP32，和 vendor/Linear 保持一致

            for stage in self.stages:
                f, c = stage(f, c, emb32)   # <<< 传 emb32

            if self.with_global:
                g = f.max(dim=-1).values
                g = self.global_mlp(g)
                g = g[:, :, None].expand_as(f)
                f = torch.cat([f, g], dim=1)

            h = self.head_pre(f)
            h = self.head_act(self.head_norm(h))
            h = self.head_drop_layer(h)
            out32 = self.head_out(h)

        return out32.permute(0, 2, 1).contiguous().to(x.dtype)


    @torch.no_grad()
    def guided_velocity(self, x, t, cond, guidance_scale: float = 0.0):
        if guidance_scale <= 0.0 or cond is None or self.cond_dim == 0:
            return self.forward(x, t, cond, cond_drop_mask=None)
        v_c = self.forward(x, t, cond, cond_drop_mask=None)
        mask = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        v_u = self.forward(x, t, cond, cond_drop_mask=mask)
        return v_c + guidance_scale * (v_c - v_u)

# ---------- 下面是本类用到的辅助模块 ----------

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(a, 1)

def _choose_gn_groups(channels: int, prefer: int = 32) -> int:
    prefer = min(prefer, channels)
    g = _gcd(channels, prefer)
    if g == 1 and channels >= 16:
        # 尽量避免组数为 1（退化为 LN），优先找 8/16 等
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
    """对 (B,C,N) 特征做归一化与 FiLM 调制；zero-init 以实现“恒等残差”初始化。"""
    def __init__(self, channels: int, emb_dim: int, norm_type: str = "group",
                 gn_groups: int = 32, one_plus: bool = False):
        super().__init__()
        self.norm = _make_norm(norm_type, channels, gn_groups)
        self.affine = nn.Linear(emb_dim, channels * 2)
        self.one_plus = bool(one_plus)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        y = self.norm(x)
        emb = emb.to(y.dtype)  # <<< 兜底：保证 affine 输入与权重/特征同 dtype
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1); beta = beta.view(B, C, 1)
        if self.one_plus:
            y = y * (1.0 + gamma) + beta
        else:
            y = y * gamma + beta
        return y


class _PVBlock(nn.Module):
    """ 单个 PVConvBlock：PVConv → SharedMLP → FiLM → 残差 """
    def __init__(self, channels: int, resolution: int, emb_dim: int, with_se: bool,
                 PVConv, SharedMLP, norm_type: str = "group", gn_groups: int = 32, film_one_plus: bool = False,
                 voxel_normalize=False):
        super().__init__()
        self.pvconv = PVConv(channels, channels, kernel_size=3,
                             resolution=int(resolution), with_se=bool(with_se),
                             normalize=bool(voxel_normalize),
                             eps=1e-6)
        self.post = SharedMLP(channels, [channels])
        self.film = _FiLM1d(channels, emb_dim, norm_type=norm_type, gn_groups=gn_groups, one_plus=film_one_plus)

    def forward(self, feat_coords: Tuple[torch.Tensor, torch.Tensor], emb: torch.Tensor):
        f, c = self.pvconv(feat_coords)
        f, c = self.post((f, c))
        f = f + self.film(f, emb)
        return f, c

class _PVStage(nn.Module):
    """ Stage: 1×1 SharedMLP 做通道提升 → k×(PVConvBlock) """
    def __init__(self, in_c: int, out_c: int, num_blocks: int, resolution: int, emb_dim: int, with_se: bool,
                 PVConv, SharedMLP, norm_type: str = "group", gn_groups: int = 32, film_one_plus: bool = False,
                 voxel_normalize=False):
        super().__init__()
        self.proj = SharedMLP(in_c, [out_c])
        self.blocks = nn.ModuleList([
            _PVBlock(out_c, resolution, emb_dim, with_se, PVConv, SharedMLP,
                     norm_type=norm_type, gn_groups=gn_groups, film_one_plus=film_one_plus,
                     voxel_normalize=voxel_normalize)
            for _ in range(int(num_blocks))
        ])

    def forward(self, feat: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor):
        f, c = self.proj((feat, coords))
        for blk in self.blocks:
            f, c = blk((f, c), emb)
        return f, c

def _replace_bn_with_gn_(module: nn.Module, groups: int = 32):
    """把子模块里的 BatchNorm1d/2d/SyncBN 全换成 GroupNorm（用于 vendor 的 SharedMLP/PVConv）。"""
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.SyncBatchNorm)):
            g = _choose_gn_groups(child.num_features, groups)
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        elif isinstance(child, nn.BatchNorm2d):
            g = _choose_gn_groups(child.num_features, groups)
            # GroupNorm 也适用于 (B,C,H,W)，这里保持通用性
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        else:
            _replace_bn_with_gn_(child, groups=groups)

