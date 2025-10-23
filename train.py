# train.py
from __future__ import annotations
import os, argparse, random, re  # [Auto-Resume] 引入 re 用于解析 ckpt 文件名
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from tqdm import tqdm
from models import VelocityNet, PVCNNVelocityNet
from datasets import get_datasets, init_np_seed
from models import VelocityNet, ConditionalLatentVelocityNet
from models import ShapeEncoder, CondAdversary, grad_reverse
from utils import EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr, \
                  save_point_cloud_ply, save_point_cloud_xyz, count_parameters, \
                  save_point_cloud_ply_rgb   # 新增

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

def sample_noise_like(x: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    return torch.randn_like(x) * std

@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(pred, target, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)

def group_pairs_by_anno(annos: List[str]) -> List[Tuple[int,int]]:
    from collections import defaultdict
    g = defaultdict(list)
    for i, a in enumerate(annos):
        g[str(a)].append(i)
    pairs = []
    for _, idxs in g.items():
        if len(idxs) >= 2:
            random.shuffle(idxs)
            m = min(8, len(idxs) - 1)
            for t in range(m):
                pairs.append((idxs[t], idxs[t+1]))
    return pairs

def main():
    p = argparse.ArgumentParser("Route-C Joint FM: latent + conditional point-flow (xyz+rgb)")
    # Data
    p.add_argument("--dataset_type", type=str, default="partnet_h5", choices=["tdcr_h5","partnet_h5"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tr_max_sample_points", type=int, default=2048)
    p.add_argument("--te_max_sample_points", type=int, default=2048)
    p.add_argument("--tdcr_use_norm", action="store_true", default=True)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--train_subset_seed", type=int, default=0)

    # Models
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
    p.add_argument("--pf_backbone", type=str, default="mlp", choices=["mlp","pvcnn"])

    p.add_argument("--pv_width", type=int, default=128)
    p.add_argument("--pv_blocks", type=int, default=3)
    p.add_argument("--pv_res", type=int, default=32)
    p.add_argument("--pv_with_se", action="store_true", default=True)

    p.add_argument("--pv_norm", type=str, default="group", choices=["group","batch","syncbn","none"])
    p.add_argument("--pv_gn_groups", type=int, default=32)

    p.add_argument("--pv_num_stages", type=int, default=3,
                help="仅用于检查/日志；实际由下三项的长度决定")
    p.add_argument("--pv_stage_channels", type=int, nargs="+", default=None,
                help="例如: 128 256 256；若为空将基于 pv_width 自动推导")
    p.add_argument("--pv_stage_blocks", type=int, nargs="+", default=None,
                help="例如: 2 2 2；若为空将基于 pv_blocks 自动推导")
    p.add_argument("--pv_stage_res", type=int, nargs="+", default=None,
                help="例如: 32 16 8；若为空将基于 pv_res 自动推导")

    p.add_argument("--pv_freeze_bn_after", type=int, default=50,
                help="第几轮开始将 BN 固定为 eval()（仅当 norm=batch/syncbn 时有意义）")
    p.add_argument("--pv_head_drop", type=float, default=0.0)


    p.add_argument("--use_rgb_in_latent", action="store_true", default=True,
                   help="Encoder 的输入是否拼 rgb（若数据有 rgb）")
    p.add_argument("--pointflow_rgb", action="store_true", default=False,
                   help="Point flow 是否在 6D(xyz+rgb) 上学习/采样（若数据有 rgb）")

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr_enc", type=float, default=3e-4)
    p.add_argument("--lr_pf", type=float, default=3e-4)
    p.add_argument("--lr_lf", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--use_cosine_lr", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)

    # FM priors
    p.add_argument("--point_prior_std", type=float, default=1.0, help="XYZ 高斯先验的 std")
    p.add_argument("--latent_prior_std", type=float, default=1.0)
    p.add_argument("--color_prior", type=str, choices=["gauss","uniform","zeros"], default="gauss",
                   help="PF 中 RGB 维度的初始分布：高斯/均匀[0,1]/全 0")
    p.add_argument("--color_prior_std", type=float, default=1.0, help="当 color_prior=gauss 时使用")
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=0.0)

    # Loss weights
    p.add_argument("--lambda_point", type=float, default=1.0)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_pair", type=float, default=0.1)
    p.add_argument("--lambda_adv", type=float, default=0.0)
    p.add_argument("--lambda_var", type=float, default=1.0)
    p.add_argument("--lambda_cov", type=float, default=0.01)
    p.add_argument("--lambda_zreg", type=float, default=1e-4)
    p.add_argument("--lambda_color", type=float, default=1.0, help="PF 中颜色分量的损失权重")

    # System
    p.add_argument("--out_dir", type=str, default="./runs/routeC_joint_rgb")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--vis_count", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--use_bf16", action="store_true", default=True)

    # PartNet 特有
    p.add_argument("--partnet_cond_policy", type=str, default="mode", choices=["mode","max"])
    p.add_argument("--partnet_exclude_outliers", action="store_true", default=False)
    p.add_argument("--partnet_report_file_train", type=str, default="")
    p.add_argument("--partnet_report_file_eval", type=str, default="")

    args = p.parse_args()

    # ---- PVCNN stage 默认推导 ----

    if args.pv_stage_channels is None:
        baseC = int(args.pv_width)
        args.pv_stage_channels = [baseC, baseC * 2, baseC * 2]
    if args.pv_stage_blocks is None:
        if args.pv_blocks >= 3:
            args.pv_stage_blocks = [max(1, args.pv_blocks // 3)] * 3
        else:
            args.pv_stage_blocks = [1, 1, 1]
    if args.pv_stage_res is None:
        r = int(args.pv_res)
        args.pv_stage_res = [max(1, r), max(1, r // 2), max(1, r // 4)]
    args.pv_num_stages = len(args.pv_stage_channels)


    is_dist, rank, world_size, local_rank = init_distributed()
    args.is_distributed = is_dist; args.rank=rank; args.world_size=world_size; args.local_rank=local_rank
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if rank == 0: os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed + rank)

    # datasets（内部会设置 args.cond_dim & args.has_rgb）
    tr_ds, te_ds = get_datasets(args)
    args.has_rgb = bool(getattr(args, "has_rgb", False))

    # loaders
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

    # Models
    enc_in_ch = 6 if (args.use_rgb_in_latent and args.has_rgb) else 3
    enc = ShapeEncoder(args.latent_dim, width=args.enc_width, depth=args.enc_depth, in_channels=enc_in_ch).to(args.device)
    # PF point_dim 决定是否在 6D 上学习
    pf_point_dim = 6 if (args.pointflow_rgb and args.has_rgb) else 3
    pf_cond_dim = args.latent_dim + args.cond_dim
    
    if args.pf_backbone == "pvcnn":
        pf = PVCNNVelocityNet(
            cond_dim=pf_cond_dim, point_dim=pf_point_dim,
            emb_dim=args.pf_emb_dim, cfg_dropout_p=args.cfg_drop_p,
            # 兼容旧参数：若下方 lists 未给，类里也会再次推导
            pv_width=args.pv_width, pv_blocks=args.pv_blocks,
            pv_resolution=args.pv_res, pv_with_se=args.pv_with_se,
            # 新：多 stage 配置
            pv_channels=args.pv_stage_channels,
            pv_blocks_per_stage=args.pv_stage_blocks,
            pv_resolutions=args.pv_stage_res,
            # 新：归一化/FiLM/全局/头部
            norm_type=args.pv_norm, gn_groups=args.pv_gn_groups,
            film_one_plus=False,      # Zero-init 残差更稳；如需 1+γ 方式可改 True
            with_global=True,
            head_drop=args.pv_head_drop,
            head_zero_init=True
        ).to(args.device)
    else:
        pf = VelocityNet(cond_dim=pf_cond_dim, width=args.pf_width, depth=args.pf_depth,
                        emb_dim=args.pf_emb_dim, cfg_dropout_p=args.cfg_drop_p,
                        point_dim=pf_point_dim).to(args.device)

    
    lf = ConditionalLatentVelocityNet(args.latent_dim, cond_dim=0, width=args.lf_width,
                                      depth=args.lf_depth, emb_dim=args.lf_emb_dim).to(args.device)
    ema_pf = EMA(pf, decay=0.999); ema_lf = EMA(lf, decay=0.999)
    pf.ema_shadow = ema_pf.shadow; lf.ema_shadow = ema_lf.shadow
    adv = CondAdversary(args.latent_dim, cond_dim=args.cond_dim, width=256, depth=3).to(args.device)

    if rank == 0:
        print(f"[Route-C] enc: {count_parameters(enc)/1e6:.2f}M  pf: {count_parameters(pf)/1e6:.2f}M  lf: {count_parameters(lf)/1e6:.2f}M")
        print(f"cond_dim(joint)={args.cond_dim}  latent_dim={args.latent_dim}  pf_cond_dim={pf_cond_dim}  enc_in_channels={enc_in_ch}  pf_point_dim={pf_point_dim}")

    model_pf = pf

    # 若选择 syncbn：将全模型 BN 转换为 SyncBatchNorm
    if args.pf_backbone == "pvcnn" and args.pv_norm == "syncbn" and torch.cuda.device_count() > 1:
        pf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(pf)

    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        model_pf = DDP(pf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        lf = DDP(lf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        adv = DDP(adv, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    opt_main = torch.optim.AdamW(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
                                 lr=args.lr_pf, weight_decay=args.weight_decay)
    opt_adv  = torch.optim.AdamW(adv.parameters(), lr=max(1e-4, args.lr_pf*0.5), weight_decay=0.0)
    scaler = make_scaler(enabled=args.amp)

    args.total_steps = args.epochs * max(1, len(train_loader))
    args.global_step = 0

    # 固定 val batch 便于可视化
    val_iter = iter(val_loader)
    try:
        val_batch = next(val_iter)
    except StopIteration:
        val_batch = next(iter(val_loader))

    # ------- 工具：构造 PF 初始噪声（支持颜色不同先验） -------
    def make_pf_prior_like(data_pf: torch.Tensor) -> torch.Tensor:
        """
        data_pf: (B,N,point_dim)  -> 返回同形状的初始噪声
        xyz 用高斯 std=args.point_prior_std；RGB 由 args.color_prior 决定
        """
        B, N, D = data_pf.shape
        if D == 3:
            return torch.randn_like(data_pf) * args.point_prior_std
        else:
            # D==6 : [xyz | rgb]
            z = data_pf.new_empty(B, N, 6)
            # xyz 高斯
            z[..., :3] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.point_prior_std
            # rgb
            if args.color_prior == "gauss":
                z[..., 3:] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.color_prior_std
            elif args.color_prior == "uniform":
                z[..., 3:] = torch.rand(B, N, 3, device=data_pf.device, dtype=data_pf.dtype)  # U[0,1]
            else:
                z[..., 3:] = 0.0
            return z

    # --------- 可视化：GT 编码重建（直接用模型生成颜色） ---------
    def save_val_recon(ep: int):
        net_pf = pf.module if hasattr(pf, "module") else pf
        net_enc = enc.module if hasattr(enc, "module") else enc
        net_pf.eval(); net_enc.eval()

        pts = val_batch["test_points"].to(args.device).float()          # (B,N,3)
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None: rgb = rgb.to(args.device).float()           # (B,N,3) in [0,1]
        cond_j = val_batch.get("cond", None)
        if cond_j is not None: cond_j = cond_j.to(args.device).float()

        # Encoder：输入 xyz 或 xyz+rgb
        enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)
        with torch.no_grad():
            z_gt, _ = net_enc(enc_in)                                    # (B, D)

        # 构造 PF 目标维度（3 或 6）
        if pf_point_dim == 6 and rgb is not None:
            data_pf = torch.cat([pts, rgb], dim=-1)
        else:
            data_pf = pts

        # Point flow：从 prior -> data
        x = make_pf_prior_like(data_pf)
        dt = 1.0 / args.sample_steps
        for i in range(args.sample_steps):
            t = torch.full((x.shape[0],), (i + 0.5)*dt, device=x.device, dtype=x.dtype)
            cond_full = torch.cat([z_gt, cond_j], dim=1) if cond_j is not None else z_gt
            v = net_pf.guided_velocity(x, t, cond_full, guidance_scale=args.guidance_scale)
            x = x + v * dt

        # 拆分输出 & 保存
        out_dir = os.path.join(args.out_dir, f"samples_recon_ep{ep:04d}")
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if pf_point_dim == 6 and rgb is not None:
                    pred_xyz = x[i, :, :3]
                    pred_rgb = x[i, :, 3:].clamp(0.0, 1.0)
                    save_point_cloud_ply_rgb(pred_xyz, pred_rgb, os.path.join(out_dir, f"pred_{i}.ply"))
                    # GT 也带色保存（若有）
                    save_point_cloud_ply_rgb(pts[i], rgb[i].clamp(0,1), os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i], os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val-Recon ep{ep:04d}] GT-encode(z) -> PF(+joint)  CD = {cd:.4f}（Pred 颜色为模型直接生成）")

    # --------- 可视化：随机 z（也直接输出预测颜色） ---------
    def save_val_samples(ep: int):
        net_pf = pf.module if hasattr(pf, "module") else pf
        net_lf = lf.module if hasattr(lf, "module") else lf
        net_pf.eval(); net_lf.eval()

        pts = val_batch["test_points"].to(args.device).float()
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None: rgb = rgb.to(args.device).float()
        cond_j = val_batch.get("cond", None)
        if cond_j is not None: cond_j = cond_j.to(args.device).float()

        with torch.no_grad():
            eps_z = torch.randn((pts.shape[0], args.latent_dim), device=args.device) * args.latent_prior_std
            z = net_lf.euler_sample(eps_z, cond=None, steps=args.sample_steps)  # sample z
            # prior（3D 或 6D）
            if pf_point_dim == 6 and rgb is not None:
                target_pf = torch.cat([pts, rgb], dim=-1)  # 只为 shape
            else:
                target_pf = pts
            x = make_pf_prior_like(target_pf)
            dt = 1.0 / args.sample_steps
            for i in range(args.sample_steps):
                t = torch.full((x.shape[0],), (i + 0.5)*dt, device=x.device, dtype=x.dtype)
                cond_full = torch.cat([z, cond_j], dim=1) if cond_j is not None else z
                v = net_pf.guided_velocity(x, t, cond_full, guidance_scale=args.guidance_scale)
                x = x + v * dt
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()

        if rank == 0:
            out_dir = os.path.join(args.out_dir, f"samples_ep{ep:04d}")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6:
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    if rgb is not None:
                        save_point_cloud_ply_rgb(pts[i], rgb[i].clamp(0,1), os.path.join(out_dir, f"gt_{i}.ply"))
                    else:
                        save_point_cloud_ply(pts[i], os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i], os.path.join(out_dir, f"gt_{i}.ply"))
            print(f"[Val ep{ep:04d}] random-sample vs single-GT CD = {cd:.4f}（Pred 颜色为模型直接生成）")

    # =========================
    # [Auto-Resume] 自动恢复段 + 设备修复
    # =========================
    def _find_latest_ckpt(ckpt_dir: str):
        """返回 (path, epoch)；找不到则 (None, 0)。"""
        if not os.path.isdir(ckpt_dir):
            return None, 0
        best_ep, best_path = 0, None
        for fn in os.listdir(ckpt_dir):
            m = re.match(r"routec_ep(\d+)\.pt$", fn)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    best_path = os.path.join(ckpt_dir, fn)
        return best_path, best_ep

    def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    def _safe_load_ema(ema_obj: EMA, state_dict: dict, ref_model: nn.Module, device: torch.device):
        """
        以当前 ema_obj.shadow 为“完整键集合”，用 ckpt 中重叠键覆盖，并迁移到 device。
        这样既避免 KeyError，又修复 CPU/GPU 设备不一致。
        """
        cur = ema_obj.shadow  # 已含全键的字典
        # 参考：只迁移/覆盖浮点项（与 EMA.update 的使用一致）
        ref_sd = ref_model.state_dict()
        for k in cur.keys():
            if k in state_dict:
                v = state_dict[k]
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    cur[k] = v.to(device=device, dtype=ref_sd[k].dtype)
        # 非浮点项保留原来的（一般不会在 update 中使用）
        ema_obj.shadow = cur

    start_epoch = 1
    ckpt_path, ckpt_ep = _find_latest_ckpt(os.path.join(args.out_dir, "ckpts"))
    if ckpt_path is not None:
        if rank == 0:
            print(f"[Auto-Resume] Found latest ckpt: {ckpt_path} (ep={ckpt_ep})")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 恢复模型
        enc_t = enc.module if hasattr(enc, "module") else enc
        pf_t  = pf.module if hasattr(pf, "module") else pf
        lf_t  = lf.module if hasattr(lf, "module") else lf

        if "encoder" in ckpt: enc_t.load_state_dict(ckpt["encoder"], strict=True)
        if "pf" in ckpt:      pf_t.load_state_dict(ckpt["pf"], strict=False)
        elif "model" in ckpt: pf_t.load_state_dict(ckpt["model"], strict=False)  # 兼容老键名
        if "lf" in ckpt:      lf_t.load_state_dict(ckpt["lf"], strict=False)

        # 恢复 EMA（并迁移到正确设备 + 键对齐）
        if "ema_pf" in ckpt and isinstance(ckpt["ema_pf"], dict):
            _safe_load_ema(ema_pf, ckpt["ema_pf"], pf_t, device=torch.device(args.device))
            pf.ema_shadow = ema_pf.shadow
        if "ema_lf" in ckpt and isinstance(ckpt["ema_lf"], dict):
            _safe_load_ema(ema_lf, ckpt["ema_lf"], lf_t, device=torch.device(args.device))
            lf.ema_shadow = ema_lf.shadow

        # 恢复优化器 / AMP scaler（若存在），并迁移优化器状态到设备
        if "opt_main" in ckpt:
            try:
                opt_main.load_state_dict(ckpt["opt_main"])
                _move_opt_state_to_device(opt_main, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt_main state load failed: {e}")
        if "opt_adv" in ckpt:
            try:
                opt_adv.load_state_dict(ckpt["opt_adv"])
                _move_opt_state_to_device(opt_adv, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt_adv state load failed: {e}")
        if args.amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try: scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] scaler state load failed: {e}")

        # 恢复 epoch 与全局步数
        last_epoch = int(ckpt.get("epoch", ckpt_ep))
        approx_gs = last_epoch * max(1, len(train_loader))
        args.global_step = int(ckpt.get("global_step", approx_gs))
        start_epoch = last_epoch + 1

        if rank == 0:
            remain = max(0, args.epochs - last_epoch)
            print(f"[Auto-Resume] Resume from epoch {last_epoch}. "
                  f"Target total epochs = {args.epochs}. Will run {remain} more epoch(s).")

        # 若已经完成或超过总轮数，直接结束
        if start_epoch > args.epochs:
            if rank == 0:
                print("[Auto-Resume] Training already completed for the requested total epochs. Nothing to do.")
            cleanup_distributed()
            return
    else:
        if rank == 0:
            print("[Auto-Resume] No checkpoint found. Start training from scratch.")

    # --------------------------- 训练循环 ---------------------------
    for ep in range(start_epoch, args.epochs + 1):  # [Auto-Resume] 从 start_epoch 继续
        # 冻结 BN（提升小 batch/长训稳定性）
        if args.pf_backbone == "pvcnn" and args.pv_norm in ["batch","syncbn"] and ep == args.pv_freeze_bn_after:
            net_pf = pf.module if hasattr(pf, "module") else pf
            if hasattr(net_pf, "set_bn_eval"):
                net_pf.set_bn_eval(True)
                if rank == 0: print(f"[Info] Freeze BatchNorm at epoch {ep}.")

        if is_dist and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)
        enc.train(); pf.train(); lf.train(); adv.train()
        pbar = tqdm(total=len(train_loader), desc=f"RouteC Ep{ep}") if rank == 0 else None

        for batch in train_loader:
            pts = batch["train_points"].to(args.device).float()    # (B,N,3)
            rgb = batch.get("train_rgb", None)
            if rgb is not None: rgb = rgb.to(args.device).float()  # (B,N,3) in [0,1]
            cond_j = batch.get("cond", None)
            if cond_j is not None: cond_j = cond_j.to(args.device).float()
            annos = batch.get("anno_id", [""] * pts.shape[0])

            # ----- Encoder（xyz 或 xyz+rgb） -----
            enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)
            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                z, _ = enc(enc_in)   # (B,D)

            # ----- Point-flow FM（3D/6D） -----
            if pf_point_dim == 6 and (rgb is not None):
                data_pf = torch.cat([pts, rgb], dim=-1)      # (B,N,6)
            else:
                data_pf = pts                                 # (B,N,3)

            B, N, D = data_pf.shape
            z_pts = make_pf_prior_like(data_pf)               # same shape
            t_pts = torch.rand(B, device=args.device, dtype=data_pf.dtype)
            x_t = (1.0 - t_pts)[:, None, None] * z_pts + t_pts[:, None, None] * data_pf
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

            # ----- Latent-flow FM（无条件 p(z)） -----
            with torch.no_grad(): z_det = z.detach()
            eps_z = torch.randn_like(z_det) * args.latent_prior_std
            t_z = torch.rand(z_det.shape[0], device=args.device, dtype=z_det.dtype)
            y_t = (1.0 - t_z)[:, None] * eps_z + t_z[:, None] * z_det
            target_v_z = (z_det - eps_z)
            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                pred_v_z = lf(y_t, t_z, cond=None)
                loss_latent = F.mse_loss(pred_v_z, target_v_z)

            # ----- Pair invariance / VICReg / Adversary / z 正则 -----
            pairs = group_pairs_by_anno(annos)
            loss_pair = data_pf.new_zeros([])
            if pairs:
                za = torch.stack([z[i] for i, j in pairs], dim=0)
                zb = torch.stack([z[j] for i, j in pairs], dim=0)
                loss_pair = F.mse_loss(za, zb)

            def var_loss(z: torch.Tensor, eps: float = 1e-4):
                std = torch.sqrt(z.var(dim=0) + eps); return torch.relu(1.0 - std).mean()
            def cov_loss(z: torch.Tensor):
                zc = z - z.mean(dim=0, keepdim=True)
                C = (zc.T @ zc) / max(1, z.shape[0] - 1)
                off = C - torch.diag(torch.diag(C))
                return (off ** 2).mean()
            loss_var = var_loss(z.float()); loss_cov = cov_loss(z.float())

            if args.lambda_adv > 0.0 and cond_j is not None and args.cond_dim > 0:
                for p in adv.parameters(): p.requires_grad_(False)
                z_rev = grad_reverse(z, args.lambda_adv)
                cond_pred = adv(z_rev)
                loss_adv = F.mse_loss(cond_pred, cond_j)
                for p in adv.parameters(): p.requires_grad_(True)
            else:
                loss_adv = data_pf.new_zeros([])

            loss_zreg = args.lambda_zreg * z.pow(2).mean()

            loss = args.lambda_point * loss_point + \
                   args.lambda_latent * loss_latent + \
                   args.lambda_pair * loss_pair + \
                   args.lambda_var * loss_var + \
                   args.lambda_cov * loss_cov + \
                   loss_adv + loss_zreg

            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(opt_main)
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
                                               args.grad_clip_norm)
            scaler.step(opt_main); scaler.update(); opt_main.zero_grad(set_to_none=True)

            if args.lambda_adv > 0.0 and cond_j is not None and args.cond_dim > 0:
                with torch.no_grad(): z_det2, _ = enc(enc_in)
                pred_c = adv(z_det2.detach())
                loss_c = F.mse_loss(pred_c, cond_j)
                loss_c.backward(); opt_adv.step(); opt_adv.zero_grad(set_to_none=True)

            ema_pf.update(pf.module if hasattr(pf, "module") else pf)
            ema_lf.update(lf.module if hasattr(lf, "module") else lf)

            if args.use_cosine_lr:
                lr_now = cosine_lr(args.global_step, args.total_steps, args.lr_pf, args.min_lr, args.warmup_steps)
                for pg in opt_main.param_groups: pg['lr'] = lr_now
                for pg in opt_adv.param_groups:  pg['lr'] = lr_now * 0.5
            args.global_step += 1

            if pbar is not None:
                pbar.set_postfix(lp=float(loss_point.detach().cpu()),
                                 lz=float(loss_latent.detach().cpu()))
                pbar.update(1)
        if pbar is not None: pbar.close()

        # save & viz
        if (ep % args.save_every) == 0 or ep == args.epochs:
            if rank == 0:
                ckpt = {
                    "epoch": ep,
                    "encoder": (enc.module if hasattr(enc, "module") else enc).state_dict(),
                    "pf": (pf.module if hasattr(pf, "module") else pf).state_dict(),
                    "lf": (lf.module if hasattr(lf, "module") else lf).state_dict(),
                    "ema_pf": ema_pf.shadow,
                    "ema_lf": ema_lf.shadow,
                    "args": {
                        **vars(args),
                        "enc_in_channels": enc_in_ch,
                        "pf_point_dim": pf_point_dim,   # 关键：记录 PF 的点维度（3/6）
                    },
                    "cond_dim": args.cond_dim,
                    # [Auto-Resume] 额外保存优化器/AMP/全局步数，方便下次精确恢复
                    "opt_main": opt_main.state_dict(),
                    "opt_adv": opt_adv.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "global_step": args.global_step,
                }
                os.makedirs(os.path.join(args.out_dir, "ckpts"), exist_ok=True)
                torch.save(ckpt, os.path.join(args.out_dir, "ckpts", f"routec_ep{ep:04d}.pt"))
            if is_dist and dist.is_initialized(): dist.barrier()
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


[剪刀]
export CUDA_VISIBLE_DEVICES=5
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/partnet/Scissors \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/scissors_rgb/_train_report.json \
  --out_dir runs/scissors_rgb


[shapenet]
export CUDA_VISIBLE_DEVICES=4
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../dataset/shapenet/airplane \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 15000 --te_max_sample_points 15000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --lambda_pair 0.0 \
  --out_dir runs/partnet_airplane



[钳子 pvcnn]
python train.py \
  --dataset_type partnet_h5 \
  --data_dir ../Dataset/partnet/Pliers \
  --batch_size 8 --epochs 3000 --save_every 10 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 \
  --pf_backbone pvcnn \
  --pv_width 128 --pv_blocks 6 --pv_res 32 --pv_with_se \
  --pv_norm group --pv_gn_groups 32 \
  --pv_stage_channels 128 256 256 \
  --pv_stage_blocks 2 2 2 \
  --pv_stage_res 32 16 8 \
  --pv_freeze_bn_after 50 \
  --partnet_report_file_train runs/pliers_pvcnn/_train_report.json \
  --out_dir runs/pliers_pvcnn


'''
