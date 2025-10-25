from __future__ import annotations
import os, argparse
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch import distributed as dist

# ---- datasets / models / utils ----
from datasets import get_datasets, init_np_seed
from models import VelocityNet, HybridMLP, ConditionalLatentVelocityNet, ShapeEncoder
from util import EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr, \
                         save_point_cloud_ply, save_point_cloud_xyz, count_parameters
from util import save_point_cloud_ply_rgb


# ========== EMA eval helper ==========
from contextlib import contextmanager
import torch.nn as nn
import torch

@contextmanager
def use_ema_weights(module: nn.Module, ema_shadow: dict | None, enabled: bool = True):
    """
    临时用 EMA 权重覆盖 module 的浮点参数/缓冲区，退出时恢复原值。
    用法：
        with use_ema_weights(net_pf, ema_pf.shadow, enabled=True):
            # eval forward ...
    """
    if (not enabled) or (ema_shadow is None):
        yield module
        return

    device = next(module.parameters()).device
    saved_params, saved_bufs = {}, {}

    # 覆盖可训练参数
    for name, p in module.named_parameters(recurse=True):
        if p.dtype.is_floating_point and (name in ema_shadow):
            saved_params[name] = p.data.detach().clone()
            p.data.copy_(ema_shadow[name].to(device=device, dtype=p.dtype))

    # 覆盖浮点 buffer（如 BN 的 running_mean/var 等）
    for name, b in module.named_buffers(recurse=True):
        if torch.is_tensor(b) and b.dtype.is_floating_point and (name in ema_shadow):
            saved_bufs[name] = b.data.detach().clone()
            b.data.copy_(ema_shadow[name].to(device=device, dtype=b.dtype))

    try:
        yield module
    finally:
        # 恢复
        for name, p in module.named_parameters(recurse=True):
            if name in saved_params:
                p.data.copy_(saved_params[name])
        for name, b in module.named_buffers(recurse=True):
            if name in saved_bufs:
                b.data.copy_(saved_bufs[name])


# ---- AMP helpers ----
_USE_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
def make_autocast(enabled: bool, use_bf16: bool):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if _USE_NEW_AMP:
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    else:
        from torch.cuda.amp import autocast as _autocast
        return _autocast(enabled=enabled, dtype=dtype)
def make_scaler(enabled: bool):
    if _USE_NEW_AMP:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        from torch.cuda.amp import GradScaler as _GradScaler
        return _GradScaler(enabled=enabled)

# ---- metrics ----
@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 双向最近邻 L2^2 的和（按 batch 求均值）
    d2 = torch.cdist(pred, target, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)

def main():
    p = argparse.ArgumentParser("FM training (MLP / HybridMLP point-flow)")
    # ========== Data ==========
    p.add_argument("--dataset_type", type=str, default="partnet_h5", choices=["tdcr_h5","partnet_h5"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tr_max_sample_points", type=int, default=2048)
    p.add_argument("--te_max_sample_points", type=int, default=2048)
    p.add_argument("--tdcr_use_norm", action="store_true", default=True)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--train_subset_seed", type=int, default=0)

    # ========== Backbone & Models ==========
    # 选择点流骨干：mlp（原始）或 hybrid（SA/FP + 逐点 MLP）
    p.add_argument("--pf_backbone", type=str, default="mlp", choices=["mlp","hybrid"])

    # Encoder / PF / LF 公共超参
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--enc_width", type=int, default=128)
    p.add_argument("--enc_depth", type=int, default=4)

    p.add_argument("--pf_width", type=int, default=512)
    p.add_argument("--pf_depth", type=int, default=6)
    p.add_argument("--pf_emb_dim", type=int, default=256)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)

    p.add_argument("--lf_width", type=int, default=512)
    p.add_argument("--lf_depth", type=int, default=6)
    p.add_argument("--lf_emb_dim", type=int, default=256)

    # Hybrid 上下文分支（ContextNet）超参
    p.add_argument("--ctx_dim", type=int, default=64)
    p.add_argument("--ctx_emb_dim", type=int, default=256)
    p.add_argument("--ctx_stage_channels", type=int, nargs="+", default=[128, 256, 256])
    p.add_argument("--ctx_stage_blocks", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--ctx_stage_res", type=int, nargs="+", default=[32, 16, 8])
    p.add_argument("--ctx_with_se", action="store_true", default=True)
    p.add_argument("--ctx_norm", type=str, default="group", choices=["group","batch","syncbn","none"])
    p.add_argument("--ctx_gn_groups", type=int, default=32)
    p.add_argument("--ctx_with_global", action="store_true", default=True)
    p.add_argument("--ctx_voxel_normalize", action="store_true", default=True)  # 强烈建议 True

    # 颜色开关（不想学颜色就关掉）
    p.add_argument("--use_rgb_in_latent", action="store_true", default=True,
                   help="Encoder 输入是否拼 rgb（若数据有 rgb）")
    p.add_argument("--pointflow_rgb", action="store_true", default=True,
                   help="Point-flow 是否在 6D(xyz+rgb) 上学习/采样（若数据有 rgb）")

    # ========== Training ==========
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr_enc", type=float, default=3e-4)
    p.add_argument("--lr_pf", type=float, default=3e-4)
    p.add_argument("--lr_lf", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--use_cosine_lr", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--t_beta_a", type=float, default=2.0, help="t ~ Beta(a, 1). a>1 时靠近 1 更稠密（默认 2.0）")
    p.add_argument("--geom_warmup_epochs", type=int, default=200, help="前多少个 epoch 仅训练几何（颜色维度置零且不计入损失）")
    # ========== FM priors ==========
    p.add_argument("--point_prior_std", type=float, default=1.0, help="XYZ 高斯先验的 std")
    p.add_argument("--latent_prior_std", type=float, default=1.0)
    p.add_argument("--color_prior", type=str, choices=["gauss","uniform","zeros"], default="gauss",
                   help="PF 中 RGB 维度初始分布")
    p.add_argument("--color_prior_std", type=float, default=1.0, help="当 color_prior=gauss 时使用")

    # ========== Sampling / CFG / EMA ==========
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_eval", action="store_true", default=True)

    # ========== Loss ==========
    p.add_argument("--lambda_point", type=float, default=1.0)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_color", type=float, default=1.0)

    # ========== System / I/O ==========
    p.add_argument("--out_dir", type=str, default="./runs/hybrid")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--vis_count", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--use_bf16", action="store_true", default=True)

    args = p.parse_args()

    # ---- DDP / device / seed ----
    is_dist, rank, world_size, local_rank = init_distributed()
    args.is_distributed = is_dist; args.rank=rank; args.world_size=world_size; args.local_rank=local_rank
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if rank == 0: os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed + rank)

    # ---- datasets (内部会设置 args.cond_dim & args.has_rgb) ----
    tr_ds, te_ds = get_datasets(args)
    args.has_rgb = bool(getattr(args, "has_rgb", False))

    # ---- loaders ----
    if is_dist:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True, drop_last=True)
        te_sampler = DistributedSampler(te_ds, shuffle=False, drop_last=False)
    else:
        tr_sampler = None; te_sampler = None
    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=(tr_sampler is None),
                              sampler=tr_sampler, num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, worker_init_fn=init_np_seed)
    val_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=te_sampler, num_workers=max(1, args.num_workers//2),
                            drop_last=False, pin_memory=True, worker_init_fn=init_np_seed)

    # ---- models ----
    enc_in_ch = 6 if (args.use_rgb_in_latent and args.has_rgb) else 3
    enc = ShapeEncoder(args.latent_dim, width=args.enc_width, depth=args.enc_depth, in_channels=enc_in_ch).to(args.device)

    pf_point_dim = 6 if (args.pointflow_rgb and args.has_rgb) else 3
    pf_cond_dim  = args.latent_dim + args.cond_dim

    if args.pf_backbone == "mlp":
        pf = VelocityNet(cond_dim=pf_cond_dim, width=args.pf_width, depth=args.pf_depth,
                         emb_dim=args.pf_emb_dim, cfg_dropout_p=args.cfg_drop_p,
                         point_dim=pf_point_dim).to(args.device)
    else:
        # HybridMLP：ContextNet(SA/FP) + VelocityNetWithContext
        pf = HybridMLP(
            cond_dim=pf_cond_dim,
            point_dim=pf_point_dim,
            # ContextNet
            ctx_dim=args.ctx_dim, ctx_emb_dim=args.ctx_emb_dim,
            stage_channels=args.ctx_stage_channels, stage_blocks=args.ctx_stage_blocks, stage_res=args.ctx_stage_res,
            with_se=args.ctx_with_se, norm_type=args.ctx_norm, gn_groups=args.ctx_gn_groups,
            with_global=args.ctx_with_global, voxel_normalize=args.ctx_voxel_normalize,
            # Head (逐点 MLP)
            pf_width=args.pf_width, pf_depth=args.pf_depth, pf_emb_dim=args.pf_emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        ).to(args.device)

    lf = ConditionalLatentVelocityNet(args.latent_dim, cond_dim=0, width=args.lf_width,
                                      depth=args.lf_depth, emb_dim=args.lf_emb_dim).to(args.device)

    ema_pf = EMA(pf, decay=args.ema_decay); ema_lf = EMA(lf, decay=args.ema_decay)
    pf.ema_shadow = ema_pf.shadow; lf.ema_shadow = ema_lf.shadow

    if rank == 0:
        print(f"[Models] enc: {count_parameters(enc)/1e6:.2f}M  pf: {count_parameters(pf)/1e6:.2f}M  lf: {count_parameters(lf)/1e6:.2f}M")
        print(f"[Dims] cond_dim(joint)={args.cond_dim} latent_dim={args.latent_dim} pf_cond_dim={pf_cond_dim} enc_in={enc_in_ch} pf_point_dim={pf_point_dim}")

    model_pf = pf
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        model_pf = DDP(pf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        lf = DDP(lf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    # ---- optim / scaler ----
    # opt = torch.optim.AdamW(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
    #                         lr=args.lr_pf, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW([
        {"params": enc.parameters(), "lr": args.lr_enc},
        {"params": pf.parameters(),  "lr": args.lr_pf},
        {"params": lf.parameters(),  "lr": args.lr_lf},
    ], weight_decay=args.weight_decay)

    scaler = make_scaler(enabled=args.amp)

    args.total_steps = args.epochs * max(1, len(train_loader))
    args.global_step = 0

    # ---- 固定 val batch 便于对比 ----
    val_iter = iter(val_loader)
    try:    val_batch = next(val_iter)
    except: val_batch = next(iter(val_loader))

    # ---- prior 生成器（支持 RGB 先验） ----
    def make_pf_prior_like(data_pf: torch.Tensor) -> torch.Tensor:
        B, N, D = data_pf.shape
        if D == 3:
            return torch.randn_like(data_pf) * args.point_prior_std
        else:
            z = data_pf.new_empty(B, N, 6)
            z[..., :3] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.point_prior_std
            if args.color_prior == "gauss":
                z[..., 3:] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.color_prior_std
            elif args.color_prior == "uniform":
                z[..., 3:] = torch.rand(B, N, 3, device=data_pf.device, dtype=data_pf.dtype)  # U[0,1]
            else:
                z[..., 3:] = 0.0
            return z

    # ---- 可视化：GT 编码重建（z=encoder(x)） ----
    @torch.no_grad()
    def save_val_recon(ep: int):
        """
        使用 EMA 权重评估 + Heun(RK2) 积分的重建可视化。
        - latent 直接由 encoder 得到（不是采样）
        - point-flow 用 Heun (predictor-corrector)
        """
        # 取出 DDP 包裹内的真实模块（若无 DDP，等价于自身）
        net_pf  = pf.module  if hasattr(pf,  "module") else pf
        net_enc = enc.module if hasattr(enc, "module") else enc
        net_lf  = lf.module  if hasattr(lf,  "module") else lf   # 为了统一用 EMA 包裹

        net_pf.eval(); net_enc.eval(); net_lf.eval()

        # 取出一个固定的验证 batch
        pts = val_batch["test_points"].to(args.device).float()  # (B,N,3)
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()

        # 编码器输入（是否拼 RGB 由 enc_in_ch 决定）
        enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)

        # 是否在评估时使用 EMA 权重
        use_ema = bool(getattr(args, "ema_eval", True))
        with use_ema_weights(net_pf, ema_pf.shadow, enabled=use_ema), \
            use_ema_weights(net_lf, ema_lf.shadow, enabled=use_ema):

            # 1) 得到 z_gt
            z_gt, _ = net_enc(enc_in)   # (B, latent_dim)

            # 2) 构造 cond_full = [z_gt | cond]，维度与 pf 构造时的 cond_dim 对齐
            B = z_gt.shape[0]
            cond_j = val_batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).to(z_gt.dtype)
                cond_full = torch.cat([z_gt, cond_j], dim=1)
            else:
                # 若模型有额外 cond 维度但验证集未提供，则补零
                need = int(getattr(args, "cond_dim", 0))
                if need > 0:
                    pad = torch.zeros((B, need), device=args.device, dtype=z_gt.dtype)
                    cond_full = torch.cat([z_gt, pad], dim=1)
                else:
                    cond_full = z_gt

            # 3) Point-Flow 的初始先验 x0
            data_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
            x = make_pf_prior_like(data_pf)  # (B,N,3/6)

            # 4) Heun (RK2) 预测-校正积分：t0=k/steps → t1=(k+1)/steps
            steps = max(1, int(args.sample_steps))
            dt = 1.0 / steps
            for k in range(steps):
                t0 = torch.full((B,), k * dt,          device=x.device, dtype=x.dtype)
                v1 = net_pf.guided_velocity(x, t0, cond_full, guidance_scale=args.guidance_scale)
                x_hat = x + v1 * dt
                t1 = torch.full((B,), (k + 1) * dt,    device=x.device, dtype=x.dtype)
                v2 = net_pf.guided_velocity(x_hat, t1, cond_full, guidance_scale=args.guidance_scale)
                x = x + 0.5 * dt * (v1 + v2)

        # 5) 保存与评估
        out_dir = os.path.join(args.out_dir, f"samples_recon_ep{ep:04d}")
        if args.rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (rgb is not None) and ("save_point_cloud_ply_rgb" in globals() and save_point_cloud_ply_rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i],       rgb[i].clamp(0,1),    os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                           os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val-Recon ep{ep:04d}] CD = {cd:.4f} (EMA={use_ema}, Heun)")


    # ---- 可视化：随机 z 采样 ----
    @torch.no_grad()
    def save_val_samples(ep: int):
        """
        使用 EMA 权重评估 + Heun(RK2) 的随机采样可视化：
        - latent-flow：Heun 采样 z
        - point-flow：Heun 从 x0 积分到数据域
        """
        net_pf = pf.module if hasattr(pf, "module") else pf
        net_lf = lf.module if hasattr(lf, "module") else lf
        net_pf.eval(); net_lf.eval()

        pts = val_batch["test_points"].to(args.device).float()
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()

        use_ema = bool(getattr(args, "ema_eval", True))
        with use_ema_weights(net_pf, ema_pf.shadow, enabled=use_ema), \
            use_ema_weights(net_lf, ema_lf.shadow, enabled=use_ema):

            B = pts.shape[0]

            # 1) latent-flow（无条件）Heun 采样：y0 ~ N(0,σ^2 I) → z
            z = torch.randn((B, args.latent_dim), device=args.device, dtype=pts.dtype) * args.latent_prior_std
            steps = max(1, int(args.sample_steps))
            dt = 1.0 / steps
            for k in range(steps):
                t0 = torch.full((B,), k * dt,       device=z.device, dtype=z.dtype)
                v1 = net_lf(z, t0, cond=None)
                z_hat = z + v1 * dt
                t1 = torch.full((B,), (k + 1) * dt, device=z.device, dtype=z.dtype)
                v2 = net_lf(z_hat, t1, cond=None)
                z = z + 0.5 * dt * (v1 + v2)

            # 2) cond_full = [z | cond]，与训练对齐
            cond_j = val_batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).to(z.dtype)
                cond_full = torch.cat([z, cond_j], dim=1)
            else:
                need = int(getattr(args, "cond_dim", 0))
                if need > 0:
                    pad = torch.zeros((B, need), device=args.device, dtype=z.dtype)
                    cond_full = torch.cat([z, pad], dim=1)
                else:
                    cond_full = z

            # 3) point-flow：x0 ~ prior → Heun 积分
            target_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
            x = make_pf_prior_like(target_pf)
            for k in range(steps):
                t0 = torch.full((B,), k * dt,       device=x.device, dtype=x.dtype)
                v1 = net_pf.guided_velocity(x, t0, cond_full, guidance_scale=args.guidance_scale)
                x_hat = x + v1 * dt
                t1 = torch.full((B,), (k + 1) * dt, device=x.device, dtype=x.dtype)
                v2 = net_pf.guided_velocity(x_hat, t1, cond_full, guidance_scale=args.guidance_scale)
                x = x + 0.5 * dt * (v1 + v2)

        # 4) 保存 & 评估
        if args.rank == 0:
            out_dir = os.path.join(args.out_dir, f"samples_ep{ep:04d}")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (rgb is not None) and ("save_point_cloud_ply_rgb" in globals() and save_point_cloud_ply_rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i],       rgb[i].clamp(0,1),    os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                           os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val ep{ep:04d}] random-z CD = {cd:.4f} (EMA={use_ema}, Heun)")


    # =========================
    # [Auto-Resume] 自动恢复段 + 设备修复
    # =========================
    import re
    from contextlib import contextmanager

    def _find_latest_ckpt(ckpt_dir: str):
        """返回 (path, epoch)；找不到则 (None, 0)。"""
        if not os.path.isdir(ckpt_dir):
            return None, 0
        best_ep, best_path = 0, None
        for fn in os.listdir(ckpt_dir):
            m = re.match(r"hybrid_ep(\d+)\.pt$", fn)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    best_path = os.path.join(ckpt_dir, fn)
        return best_path, best_ep

    def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
        """把优化器状态里的 tensor 迁移到目标 device，防止恢复后因设备不一致报错。"""
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    def _safe_load_ema(ema_obj: EMA, state_dict: dict, ref_model: nn.Module, device: torch.device):
        """
        以当前 ema_obj.shadow 为“完整键集合”，用 ckpt 中重叠键覆盖，并迁移到 device。
        这样避免 KeyError，同时解决 CPU/GPU 设备不一致。
        """
        cur = ema_obj.shadow  # 已含全键
        ref_sd = ref_model.state_dict()
        for k in cur.keys():
            if k in state_dict:
                v = state_dict[k]
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    cur[k] = v.to(device=device, dtype=ref_sd[k].dtype)
        ema_obj.shadow = cur

    start_epoch = 1
    ckpt_path, ckpt_ep = _find_latest_ckpt(os.path.join(args.out_dir, "ckpts"))
    if ckpt_path is not None:
        if rank == 0:
            print(f"[Auto-Resume] Found latest ckpt: {ckpt_path} (ep={ckpt_ep})")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 恢复模型（DDP 下要拿 .module）
        enc_t = enc.module if hasattr(enc, "module") else enc
        pf_t  = pf.module  if hasattr(pf,  "module") else pf
        lf_t  = lf.module  if hasattr(lf,  "module") else lf

        if "encoder" in ckpt: enc_t.load_state_dict(ckpt["encoder"], strict=True)
        if "pf" in ckpt:      pf_t.load_state_dict(ckpt["pf"], strict=False)
        elif "model" in ckpt: pf_t.load_state_dict(ckpt["model"], strict=False)  # 兼容旧键名
        if "lf" in ckpt:      lf_t.load_state_dict(ckpt["lf"], strict=False)

        # 恢复 EMA（并迁移到正确设备 + 键对齐）
        if "ema_pf" in ckpt and isinstance(ckpt["ema_pf"], dict):
            _safe_load_ema(ema_pf, ckpt["ema_pf"], pf_t, device=torch.device(args.device))
            pf.ema_shadow = ema_pf.shadow
        if "ema_lf" in ckpt and isinstance(ckpt["ema_lf"], dict):
            _safe_load_ema(ema_lf, ckpt["ema_lf"], lf_t, device=torch.device(args.device))
            lf.ema_shadow = ema_lf.shadow

        # 恢复优化器 / AMP scaler（若存在），并把优化器状态迁移到当前 device
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
                _move_opt_state_to_device(opt, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt state load failed: {e}")
        elif "opt_main" in ckpt:  # 兼容老 ckpt 键名
            try:
                opt.load_state_dict(ckpt["opt_main"])
                _move_opt_state_to_device(opt, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt_main->opt load failed: {e}")

        if args.amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] scaler state load failed: {e}")


        # 恢复 epoch 与全局步数（下一轮继续）
        last_epoch = int(ckpt.get("epoch", ckpt_ep))
        approx_gs  = last_epoch * max(1, len(train_loader))
        args.global_step = int(ckpt.get("global_step", approx_gs))
        start_epoch = last_epoch + 1

        if rank == 0:
            remain = max(0, args.epochs - last_epoch)
            print(f"[Auto-Resume] Resume from epoch {last_epoch}. "
                f"Target total epochs = {args.epochs}. Will run {remain} more epoch(s).")

        # 若已训练到/超过目标总轮数，直接结束
        if start_epoch > args.epochs:
            if rank == 0:
                print("[Auto-Resume] Training already completed for the requested total epochs. Nothing to do.")
            cleanup_distributed()
            return
    else:
        if rank == 0:
            print("[Auto-Resume] No checkpoint found. Start training from scratch.")


        

    # ================= 训练 =================
    for ep in range(start_epoch, args.epochs + 1):
        # --- 本轮是否启用颜色（两阶段训练）：预热阶段仅几何 ---
        use_rgb_this_epoch = (ep > args.geom_warmup_epochs) and (args.pointflow_rgb and args.has_rgb)

        if is_dist and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)
        enc.train(); pf.train(); lf.train()
        pbar = tqdm(total=len(train_loader), desc=f"Ep{ep}") if rank == 0 else None

        for batch in train_loader:
            pts = batch["train_points"].to(args.device).float()    # (B,N,3)
            rgb = batch.get("train_rgb", None)
            if rgb is not None:
                rgb = rgb.to(args.device).float()
            cond_j = batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).float()

            # ---- 编码器 ----
            # 预热阶段建议编码器也只看几何（避免颜色“帮忙”过多）
            # 若 enc_in_ch == 6，则无论是否预热，都保证编码器输入为 6 维；
            # 预热阶段把 rgb 三维置零，避免颜色干扰但不改通道数。
            if enc_in_ch == 6:
                if rgb is not None:
                    if use_rgb_this_epoch:
                        enc_in = torch.cat([pts, rgb], dim=-1)          # (B,N,6)
                    else:
                        zeros_rgb = torch.zeros_like(pts)                # (B,N,3)
                        enc_in = torch.cat([pts, zeros_rgb], dim=-1)     # (B,N,6) — 预热: rgb=0
                else:
                    # 理论上 has_rgb=True 时不会走到这；兜底仍然补零
                    zeros_rgb = torch.zeros_like(pts)
                    enc_in = torch.cat([pts, zeros_rgb], dim=-1)
            else:
                enc_in = pts                                             # (B,N,3)

            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                z, _ = enc(enc_in)   # (B, Dz)

            # ---- Point-flow FM（3D 或 6D）----
            # 6D 情况下：预热阶段把颜色维度置零（输入与目标都为 0，用于“几何先行”）
            if pf_point_dim == 6:
                if (rgb is not None) and use_rgb_this_epoch:
                    data_pf = torch.cat([pts, rgb], dim=-1)                     # (B,N,6)
                    # 颜色先验按你的设置（gauss/uniform/zeros）
                    z_pts = make_pf_prior_like(data_pf)                          # (B,N,6)
                else:
                    # 预热阶段：颜色维度恒为 0；先验也置 0（避免噪声干扰几何）
                    zeros_rgb = torch.zeros_like(pts)
                    data_pf = torch.cat([pts, zeros_rgb], dim=-1)                # (B,N,6)
                    z_pts = torch.empty_like(data_pf)
                    z_pts[..., :3] = torch.randn_like(pts) * args.point_prior_std
                    z_pts[..., 3:] = 0.0
            else:
                data_pf = pts
                z_pts  = torch.randn_like(data_pf) * args.point_prior_std

            B, N, D = data_pf.shape

            # ---- t 采样向 1 倾斜（Beta）----
            beta = torch.distributions.Beta(concentration1=args.t_beta_a, concentration0=1.0)
            t_pts = beta.sample((B,)).to(device=args.device, dtype=data_pf.dtype)  # (B,)
            x_t   = (1.0 - t_pts)[:, None, None] * z_pts + t_pts[:, None, None] * data_pf
            target_v = (data_pf - z_pts)

            # ---- 条件拼接（与训练保持一致）----
            cond_full = z if cond_j is None else torch.cat([z, cond_j], dim=1)
            cond_drop_mask = None
            if args.cfg_drop_p > 0.0 and cond_full is not None:
                drop = (torch.rand(B, device=args.device) < args.cfg_drop_p).to(data_pf.dtype)
                cond_drop_mask = drop[:, None]

            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                pred_v = model_pf(x_t, t_pts, cond_full, cond_drop_mask=cond_drop_mask)
                if D == 6:
                    if (rgb is not None) and use_rgb_this_epoch:
                        # 正常 6D：几何 + 颜色
                        loss_pos = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                        loss_col = F.mse_loss(pred_v[..., 3:], target_v[..., 3:])
                        loss_point = loss_pos + args.lambda_color * loss_col
                    else:
                        # 预热：仅几何监督
                        loss_point = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                else:
                    loss_point = F.mse_loss(pred_v, target_v)


            # ---- Latent-flow FM（无条件）----
            with torch.no_grad(): z_det = z.detach()
            eps_z = torch.randn_like(z_det) * args.latent_prior_std
            # t_z 旧：torch.rand(...)
            beta_latent = torch.distributions.Beta(concentration1=args.t_beta_a, concentration0=1.0)
            t_z = beta_latent.sample((B,)).to(device=args.device, dtype=z_det.dtype)
            y_t = (1.0 - t_z)[:, None] * eps_z + t_z[:, None] * z_det
            target_v_z = (z_det - eps_z)
            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                pred_v_z = lf(y_t, t_z, cond=None)
                loss_latent = F.mse_loss(pred_v_z, target_v_z)


            # ---- 总损失 ----
            loss = args.lambda_point * loss_point + args.lambda_latent * loss_latent

            # ---- 反传/优化 ----
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
                                               args.grad_clip_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            # ---- EMA ----
            ema_pf.update(pf.module if hasattr(pf, "module") else pf)
            ema_lf.update(lf.module if hasattr(lf, "module") else lf)

            # ---- LR schedule ----
            if args.use_cosine_lr:
                lr_now = cosine_lr(args.global_step, args.total_steps, args.lr_pf, args.min_lr, args.warmup_steps)
                for pg in opt.param_groups: pg['lr'] = lr_now
            args.global_step += 1

            if pbar is not None:
                pbar.set_postfix(lp=float(loss_point.detach().cpu()), lz=float(loss_latent.detach().cpu()))
                pbar.update(1)
        
        if pbar is not None: pbar.close()

        # ---- Save & Eval ----
        if (ep % args.save_every) == 0 or ep == args.epochs:
            if rank == 0:
                ckpt = {
                    "epoch": ep,
                    # --- 三个子模型 ---
                    "encoder": (enc.module if hasattr(enc, "module") else enc).state_dict(),
                    "pf":      (pf.module  if hasattr(pf,  "module") else pf).state_dict(),
                    "lf":      (lf.module  if hasattr(lf,  "module") else lf).state_dict(),
                    # --- EMA shadow（评估时可切 EMA） ---
                    "ema_pf": ema_pf.shadow,
                    "ema_lf": ema_lf.shadow,
                    # --- 关键信息（便于检查 3D/6D 配置等） ---
                    "args": {
                        **vars(args),
                        "enc_in_channels": enc_in_ch,   # 3 or 6
                        "pf_point_dim": pf_point_dim,   # 3 or 6
                    },
                    "cond_dim": args.cond_dim,
                    # --- 断点续训关键：优化器 / AMP / 全局步数 ---
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "global_step": args.global_step,
                }
                os.makedirs(os.path.join(args.out_dir, "ckpts"), exist_ok=True)
                torch.save(ckpt, os.path.join(args.out_dir, "ckpts", f"hybrid_ep{ep:04d}.pt"))
            if is_dist and dist.is_initialized():
                dist.barrier()

            # 你已有的可视化
            save_val_recon(ep)
            save_val_samples(ep)


    cleanup_distributed()

if __name__ == "__main__":
    main()



'''
[眼镜]
export CUDA_VISIBLE_DEVICES=0
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/Eyeglasses \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/eyeglasses_rgb/_train_report.json \
  --out_dir runs/eyeglasses_rgb


[椅子]
export CUDA_VISIBLE_DEVICES=1
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/FoldingChair \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/foldingchair_rgb/_train_report.json \
  --out_dir runs/foldingchair_rgb

[钳子]
export CUDA_VISIBLE_DEVICES=2
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/Pliers \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/pliers_rgb/_train_report.json \
  --out_dir runs/pliers_rgb

[笔记本电脑]]
export CUDA_VISIBLE_DEVICES=4
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/Laptop \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/laptop_rgb/_train_report.json \
  --out_dir runs/laptop_rgb

[盒子]
export CUDA_VISIBLE_DEVICES=5
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/Box \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/box_rgb/_train_report.json \
  --out_dir runs/box_rgb


export CUDA_VISIBLE_DEVICES=4,5
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset_type partnet_h5 \
  --data_dir ../Dataset/partnet/Pliers \
  --batch_size 8 --epochs 3000 --save_every 10 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --pf_backbone hybrid \
  --lr_pf 1e-4 \
  --ctx_dim 64 --ctx_stage_channels 128 256 256 --ctx_stage_blocks 2 2 2 --ctx_stage_res 64 32 16 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --lambda_color 0.1\
  --use_rgb_in_latent --pointflow_rgb \
  --sample_steps 100 --guidance_scale 0.0 \
  --color_prior uniform \
  --out_dir runs/pliers_step_rgb

'''
