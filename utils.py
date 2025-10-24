# utils.py
from __future__ import annotations
import os, math, random, time, shutil
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch import distributed as dist

# ------------------- 你现有的代码（原样保留） -------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_point_cloud_xyz(xyz: torch.Tensor, path: str):
    """
    xyz: (N, 3), tensor or np
    """
    import numpy as np
    arr = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in arr:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_point_cloud_ply(xyz: torch.Tensor, path: str):
    import numpy as np
    arr = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = arr.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "end_header\n",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header))
        for p in arr:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed():
    """
    Initialize torch.distributed if launched with torchrun.
    Returns (is_distributed, rank, world_size, local_rank).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class MetricEMA:
    def __init__(self, alpha: float = 0.98):
        self.a = float(alpha)
        self.value = None

    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = self.a * self.value + (1 - self.a) * x

    def get(self) -> float:
        return float(self.value if self.value is not None else 0.0)


def shard_print(*args, rank: int = 0, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def cosine_lr(step: int, total: int, base_lr: float, min_lr: float = 1e-6, warmup: int = 0):
    if step < warmup:
        return min_lr + (base_lr - min_lr) * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------- 新增：保存带 RGB 的 PLY -------------------

def save_point_cloud_ply_rgb(xyz: torch.Tensor, rgb: torch.Tensor, path: str):
    """
    保存带颜色的 PLY：
      - xyz: (N,3) float
      - rgb: (N,3) float in [0,1] 或 uint8 in [0,255]
    """
    import numpy as np
    xyz_np = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    rgb_t = rgb.detach().cpu() if isinstance(rgb, torch.Tensor) else torch.tensor(rgb)
    if rgb_t.dtype.is_floating_point:
        rgb_np = (rgb_t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    else:
        rgb_np = rgb_t.to(torch.uint8).cpu().numpy()

    assert xyz_np.shape[0] == rgb_np.shape[0] and rgb_np.shape[1] == 3, \
        f"xyz/rgb shape mismatch: {xyz_np.shape} vs {rgb_np.shape}"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = xyz_np.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header))
        for p, c in zip(xyz_np, rgb_np):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def integrate_point_flow(net_pf, x0: torch.Tensor, cond: torch.Tensor | None,
                         steps: int, solver: str = "heun", guidance_scale: float = 0.0) -> torch.Tensor:
    """
    在 t∈[0,1] 区间用给定求解器积分：dx/dt = v_theta(x,t,cond)
    net_pf: 具有 guided_velocity(x,t,cond, guidance_scale) 接口（你已有）
    """
    x = x0
    dt = 1.0 / steps
    B = x.shape[0]
    if solver == "euler":
        for k in range(steps):
            t_mid = x.new_full((B,), (k + 0.5) * dt, dtype=x.dtype)
            v = net_pf.guided_velocity(x, t_mid, cond, guidance_scale=guidance_scale)
            x = x + v * dt
    else:  # Heun / RK2 predictor-corrector
        for k in range(steps):
            t0 = x.new_full((B,), k * dt, dtype=x.dtype)
            v1 = net_pf.guided_velocity(x, t0, cond, guidance_scale=guidance_scale)
            x_hat = x + v1 * dt
            t1 = x.new_full((B,), (k + 1) * dt, dtype=x.dtype)
            v2 = net_pf.guided_velocity(x_hat, t1, cond, guidance_scale=guidance_scale)
            x = x + 0.5 * dt * (v1 + v2)
    return x


def integrate_latent_flow(net_lf, y0: torch.Tensor, steps: int, solver: str = "heun") -> torch.Tensor:
    """
    同上，用于 latent-flow（无条件），调用 net_lf(y,t,cond=None)
    """
    y = y0
    dt = 1.0 / steps
    B = y.shape[0]
    if solver == "euler":
        for k in range(steps):
            t_mid = y.new_full((B,), (k + 0.5) * dt, dtype=y.dtype)
            v = net_lf(y, t_mid, cond=None)
            y = y + v * dt
    else:
        for k in range(steps):
            t0 = y.new_full((B,), k * dt, dtype=y.dtype)
            v1 = net_lf(y, t0, cond=None)
            y_hat = y + v1 * dt
            t1 = y.new_full((B,), (k + 1) * dt, dtype=y.dtype)
            v2 = net_lf(y_hat, t1, cond=None)
            y = y + 0.5 * dt * (v1 + v2)
    return y