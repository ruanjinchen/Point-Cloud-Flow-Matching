from __future__ import annotations
import os, argparse
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# ---- datasets / models / utils ----
from datasets import get_datasets, init_np_seed
from models import VelocityNet, HybridMLP, ConditionalLatentVelocityNet, ShapeEncoder
from util import EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr, \
                         save_point_cloud_ply, save_point_cloud_xyz, count_parameters
from util import save_point_cloud_ply_rgb


# ---- AMP helpers（与我们先前脚本一致） ----
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
    # 与我们之前的实现一致：双向最近邻 L2^2 的和（按 batch 求均值）
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
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
                            lr=args.lr_pf, weight_decay=args.weight_decay)
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
        net_pf  = pf.module  if hasattr(pf, "module")  else pf
        net_enc = enc.module if hasattr(enc, "module") else enc
        net_pf.eval(); net_enc.eval()

        pts = val_batch["test_points"].to(args.device).float()
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()

        # 编码 z
        enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)
        z_gt, _ = net_enc(enc_in)

        # PF 目标维度
        data_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts

        # flow 积分（Heun/Euler 二选一都行，这里用简单中点 Euler）
        x = make_pf_prior_like(data_pf)
        dt = 1.0 / max(1, args.sample_steps)
        for i in range(args.sample_steps):
            t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
            # 构造与训练一致的 cond_full（z [+ joint]）
            B = z_gt.shape[0]
            cond_j = val_batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).to(z_gt.dtype)
                cond_full = torch.cat([z_gt, cond_j], dim=1)
            else:
                # 若模型的 cond_dim>0，而验证 batch 没有 cond，就用 0 填充
                if getattr(args, "cond_dim", 0) > 0:
                    pad = torch.zeros((B, args.cond_dim), device=args.device, dtype=z_gt.dtype)
                    cond_full = torch.cat([z_gt, pad], dim=1)
                else:
                    cond_full = z_gt

            v = net_pf.guided_velocity(x, t, cond_full, guidance_scale=args.guidance_scale)

            x = x + v * dt

        # 保存 PLY & 打印 CD
        out_dir = os.path.join(args.out_dir, f"samples_recon_ep{ep:04d}")
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (save_point_cloud_ply_rgb is not None) and (rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i], rgb[i].clamp(0,1),          os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                        os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val-Recon ep{ep:04d}] CD = {cd:.4f}")

    # ---- 可视化：随机 z 采样 ----
    @torch.no_grad()
    def save_val_samples(ep: int):
        net_pf = pf.module if hasattr(pf, "module") else pf
        net_lf = lf.module if hasattr(lf, "module") else lf
        net_pf.eval(); net_lf.eval()

        pts = val_batch["test_points"].to(args.device).float()
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()

        # latent 采样：y0 ~ N(0,σ^2 I) → z（无条件）
        B = pts.shape[0]
        eps_z = torch.randn((B, args.latent_dim), device=args.device, dtype=pts.dtype) * args.latent_prior_std
        z = eps_z
        dt = 1.0 / max(1, args.sample_steps)
        for i in range(args.sample_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
            v = net_lf(z, t, cond=None)
            z = z + v * dt

        # point-flow：x0 ~ prior → data
        target_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
        x = make_pf_prior_like(target_pf)
        for i in range(args.sample_steps):
            t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
            cond_j = val_batch.get("cond", None)
            B = z.shape[0]
            if cond_j is not None:
                cond_j = cond_j.to(args.device).to(z.dtype)
                cond_full = torch.cat([z, cond_j], dim=1)
            else:
                if getattr(args, "cond_dim", 0) > 0:
                    pad = torch.zeros((B, args.cond_dim), device=args.device, dtype=z.dtype)
                    cond_full = torch.cat([z, pad], dim=1)
                else:
                    cond_full = z

            v = net_pf.guided_velocity(x, t, cond_full, guidance_scale=args.guidance_scale)

            x = x + v * dt

        # 保存 & CD
        if rank == 0:
            out_dir = os.path.join(args.out_dir, f"samples_ep{ep:04d}")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (save_point_cloud_ply_rgb is not None) and (rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i], rgb[i].clamp(0,1),          os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                        os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val ep{ep:04d}] random-z CD = {cd:.4f}")

    # ================= 训练 =================
    for ep in range(1, args.epochs + 1):
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
            enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)
            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                z, _ = enc(enc_in)   # (B, Dz)

            # ---- Point-flow FM（3D 或 6D）----
            data_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
            B, N, D = data_pf.shape
            z_pts  = make_pf_prior_like(data_pf)                         # same shape
            t_pts  = torch.rand(B, device=args.device, dtype=data_pf.dtype)
            x_t    = (1.0 - t_pts)[:, None, None] * z_pts + t_pts[:, None, None] * data_pf
            target_v = (data_pf - z_pts)

            cond_full = z if cond_j is None else torch.cat([z, cond_j], dim=1)
            cond_drop_mask = None
            if args.cfg_drop_p > 0.0 and cond_full is not None:
                drop = (torch.rand(B, device=args.device) < args.cfg_drop_p).to(data_pf.dtype)
                cond_drop_mask = drop[:, None]

            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                pred_v = model_pf(x_t, t_pts, cond_full, cond_drop_mask=cond_drop_mask)
                if D == 6:
                    loss_pos = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                    loss_col = F.mse_loss(pred_v[..., 3:], target_v[..., 3:])
                    loss_point = loss_pos + args.lambda_color * loss_col
                else:
                    loss_point = F.mse_loss(pred_v, target_v)

            # ---- Latent-flow FM（无条件）----
            with torch.no_grad(): z_det = z.detach()
            eps_z = torch.randn_like(z_det) * args.latent_prior_std
            t_z   = torch.rand(B, device=args.device, dtype=z_det.dtype)
            y_t   = (1.0 - t_z)[:, None] * eps_z + t_z[:, None] * z_det
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
                    "encoder": (enc.module if hasattr(enc, "module") else enc).state_dict(),
                    "pf":      (pf.module  if hasattr(pf,  "module") else pf).state_dict(),
                    "lf":      (lf.module  if hasattr(lf,  "module") else lf).state_dict(),
                    "ema_pf": ema_pf.shadow,
                    "ema_lf": ema_lf.shadow,
                    "args": {
                        **vars(args),
                        "enc_in_channels": enc_in_ch,
                        "pf_point_dim": pf_point_dim,
                    },
                    "cond_dim": args.cond_dim,
                    "global_step": args.global_step,
                }
                os.makedirs(os.path.join(args.out_dir, "ckpts"), exist_ok=True)
                torch.save(ckpt, os.path.join(args.out_dir, "ckpts", f"hybrid_ep{ep:04d}.pt"))
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



python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../Dataset/partnet/Pliers \
  --batch_size 8 --epochs 3000 --save_every 10 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --pf_backbone hybrid \
  --ctx_dim 64 --ctx_stage_channels 128 256 256 --ctx_stage_blocks 2 2 2 --ctx_stage_res 64 32 16 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --sample_steps 200 --guidance_scale 0.0 \
  --color_prior uniform \
  --out_dir runs/pliers_rgb_hybrid

'''
