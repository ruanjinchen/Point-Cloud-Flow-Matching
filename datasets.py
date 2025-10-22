import os
import glob
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, Subset
from torch.utils import data as torch_data

# ----------------------------- Utilities -----------------------------

def init_np_seed(worker_id: int):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class SubsetWithAttrs(Subset):
    """Forward attribute access safely to the base dataset (handles nesting, avoids recursion)."""
    def __getattr__(self, name):
        base = object.__getattribute__(self, 'dataset')
        while isinstance(base, Subset):
            base = object.__getattribute__(base, 'dataset')
        return getattr(base, name)

    def __dir__(self):
        base = object.__getattribute__(self, 'dataset')
        attrs = set(super().__dir__())
        while isinstance(base, Subset):
            base = object.__getattribute__(base, 'dataset')
        attrs.update(dir(base))
        return sorted(attrs)


def _attach_shuffle_idx(ds, sel_idx):
    sel_idx = np.asarray(sel_idx, dtype=np.int64)
    try:
        setattr(ds, "shuffle_idx", sel_idx)
    except Exception:
        pass
    base = getattr(ds, "dataset", None)
    if base is not None:
        try:
            setattr(base, "shuffle_idx", sel_idx)
        except Exception:
            pass


def _pick_subset_indices(args, N: int):
    train_count = getattr(args, "train_count", None)
    train_fraction = getattr(args, "train_fraction", 1.0)
    if train_count is None and (train_fraction is None or not (0.0 < float(train_fraction) < 1.0)):
        return None
    if N <= 1:
        return None
    if train_count is not None:
        n_keep = max(1, min(int(train_count), N))
    else:
        n_keep = max(1, min(int(np.ceil(N * float(train_fraction))), N))
    seed = getattr(args, "train_subset_seed", None)
    if seed is None:
        seed = getattr(args, "seed", 0)
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(N, generator=g)[:n_keep].tolist()
    idx.sort()
    print(f"[datasets] Use subset of training data: {n_keep}/{N} ({n_keep/N:.2%}) with seed={seed}")
    return np.asarray(idx, dtype=np.int64)


# ----------------------------- Conditioning helpers（TDCR） -----------------------------
# Prefer project-provided encoders if available
try:
    from condition import encode_motors as _enc_motors  # type: ignore
    from condition import get_cond_dim as _get_cond_dim  # type: ignore
except Exception:
    # 安全兜底：仅在项目里没有 condition.py 时使用
    def _module_resultant(m123, angles_deg, offset_deg=0.0):
        th = np.deg2rad(np.asarray(angles_deg, dtype=np.float32) + offset_deg)
        c = np.stack([np.cos(th), np.sin(th)], axis=0)
        vec = c @ m123.astype(np.float32)
        T = float(np.sum(m123))
        mean = T / 3.0 if T > 0 else 0.0
        A = float(np.sqrt(np.mean((m123 - mean) ** 2)))
        return vec.astype(np.float32), T, A

    def _enc_motors(m: np.ndarray, enc_mode="raw6+geom",
                    mod2_offset_deg: float = 0.0, max_pos: float = 0.4,
                    mod3_offset_deg: float = 0.0):
        m = np.asarray(m, dtype=np.float32).reshape(-1)
        assert m.shape[0] in (6, 9), f"motors dim must be 6 or 9, got {m.shape[0]}"
        nseg = 2 if m.shape[0] == 6 else 3
        mn = (m / max_pos).clip(0.0, 1.0)
        base = [180, 300, 60]
        v1, T1, A1 = _module_resultant(mn[0:3], base, 0.0)
        if nseg == 2:
            v2, T2, A2 = _module_resultant(mn[3:6], base, mod2_offset_deg)
            geom = np.concatenate([v1, [T1, A1], v2, [T2, A2], [T1 - T2, T1 + T2]], 0).astype(np.float32)
            if enc_mode == "raw6": return mn
            if enc_mode == "geom": return geom
            if enc_mode == "raw6+geom": return np.concatenate([mn, geom], 0)
            raise ValueError(f"unknown enc_mode={enc_mode}")
        else:
            v2, T2, A2 = _module_resultant(mn[3:6], base, mod2_offset_deg)
            v3, T3, A3 = _module_resultant(mn[6:9], base, mod3_offset_deg)
            geom3 = np.concatenate([v1, [T1, A1],
                                    v2, [T2, A2],
                                    v3, [T3, A3],
                                    [T1 - T2, T2 - T3, T1 - T3, T1 + T2 + T3]], 0).astype(np.float32)
            if enc_mode == "raw9": return mn
            if enc_mode == "geom3": return geom3
            if enc_mode == "raw9+geom3": return np.concatenate([mn, geom3], 0)
            raise ValueError(f"unknown enc_mode={enc_mode}")

    def _get_cond_dim(enc: str) -> int:
        table = {"raw6": 6, "geom": 10, "raw6+geom": 16,
                 "raw9": 9, "geom3": 16, "raw9+geom3": 25}
        if enc in table: return table[enc]
        return 16


# ----------------------------- 解析 anno_id 过滤 -----------------------------
def _parse_keep_annos(args) -> Tuple[Set[str], Set[str]]:
    """return (keep_ids, keep_splits)"""
    keep = set()
    if getattr(args, "keep_anno", None):
        for s in str(args.keep_anno).split(","):
            s = s.strip()
            if s:
                keep.add(s)
    if getattr(args, "keep_anno_file", None):
        p = Path(args.keep_anno_file)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        keep.add(s)
        else:
            print(f"[WARN] --keep-anno-file not found: {p}")
    splits = set()
    raw = getattr(args, "keep_anno_splits", "train")
    if raw.lower() == "all":
        splits = {"train", "val", "test"}
    else:
        for s in raw.split(","):
            s = s.strip().lower()
            if s in {"train","val","test"}:
                splits.add(s)
    if not splits:
        splits = {"train"}
    return keep, splits


# ----------------------------- TDCR-H5 Dataset（原有） -----------------------------
class TDCRH5PointClouds(Dataset):
    """Dataset for TDCR H5 shards with keys: data, data_norm, motors, center, scale."""
    def __init__(self,
                 data_dir: str = None, root_dir: str = None,
                 split: str = "train",
                 use_norm: bool = True, expand_stats: bool = False,
                 tr_sample_size: int = 2048, te_sample_size: int = 2048,
                 cond_mode: str = "motors", motor_enc: str = "raw6+geom",
                 motor_mod2_offset_deg: float = 0.0, motor_max_pos: float = 0.4,
                 motor_mod3_offset_deg: float = 0.0,
                 files=None, **kwargs) -> None:
        super().__init__()
        # Early init
        self._handles = {}
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.split = str(split)
        self.use_norm = bool(use_norm)
        self.expand_stats = bool(expand_stats)
        self.tr_n = int(tr_sample_size)
        self.te_n = int(te_sample_size)
        self.cond_mode = str(cond_mode)
        self.motor_enc = str(motor_enc)
        self.motor_mod2_offset_deg = float(motor_mod2_offset_deg)
        self.motor_mod3_offset_deg = float(motor_mod3_offset_deg)
        self.motor_max_pos = float(motor_max_pos)

        # Resolve directory
        if data_dir is None and root_dir is not None:
            data_dir = root_dir
        if data_dir is None:
            raise ValueError("TDCRH5PointClouds: data_dir/root_dir must be provided.")
        self.data_dir = os.path.abspath(data_dir)

        # Discover shards
        if files is not None:
            if isinstance(files, (list, tuple)):
                self.files = sorted(set([str(x) for x in files]))
            elif isinstance(files, str):
                self.files = sorted(set(glob.glob(files)))
            else:
                raise TypeError("files must be None, list/tuple of paths, or a glob pattern string")
        else:
            patterns = [
                os.path.join(self.data_dir, self.split, "*.h5"),
                os.path.join(self.data_dir, self.split, "*.hdf5"),
                os.path.join(self.data_dir, f"{self.split}*.h5"),
                os.path.join(self.data_dir, "*.h5"),
                os.path.join(self.data_dir, "*.hdf5"),
            ]
            flist = []
            for p in patterns:
                flist.extend(glob.glob(p))
            self.files = sorted(set(flist))
        if not self.files:
            raise FileNotFoundError(
                f"[TDCR-H5] No shard files found under '{self.data_dir}/{self.split}'. Expect shard-*.h5"
            )

        # Build index
        self._index = []
        self._key_points_map = {}
        self._has_motors = False
        eff_map = {}      # (fi, ri) -> 有效维度
        anno_map = {}     # (fi, ri) -> anno_id（可选）
        # …… 遍历每个 shard 时，算出 eff_dim 并放进 eff_map ……
        if "motors" in f:
            self._has_motors = True
            M = f["motors"][()]                                 # (B, Dmax)
            if np.issubdtype(M.dtype, np.floating):
                eff_all = (~np.isnan(M)).sum(axis=1).astype(int)
            else:
                eff_all = np.array([int(M.shape[1])] * B)
            # 记录每个 row 的有效维度
            for i in range(B):
                eff_map[(fi, i)] = int(eff_all[i])
                
        for fi, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                key = "data_norm" if (self.use_norm and "data_norm" in f) else "data"
                if key not in f:
                    raise KeyError(f"[TDCR-H5] Missing key '{key}' in file: {fp}")
                B = int(f[key].shape[0])
                self._index.extend([(fi, i) for i in range(B)])
                self._key_points_map[fi] = key
                if "motors" in f:
                    self._has_motors = True


        # —— 扫描结束后，按 _index 对齐有效维度 ——
        if self._has_motors and len(self._index) > 0:
            eff_dims_indexed = [eff_map.get((fi, ri), 0) for (fi, ri) in self._index]
            # 直方图
            dim_hist = {}
            for d in eff_dims_indexed:
                dim_hist[d] = dim_hist.get(d, 0) + 1
            # 规范维度
            if self.cond_dim_policy == "mode":
                self.cond_dim = max(dim_hist.items(), key=lambda kv: kv[1])[0]
            else:
                self.cond_dim = max(eff_dims_indexed)

            # 构建异常样本清单（仅针对 _index 内的样本）
            self.outliers = []
            if "anno_id" in f:
                # 如果各 shard 均可能有 anno_id，则建议在扫描时放 anno_map[(fi,ri)] = anno_id
                pass
            for (fi, ri), ei in zip(self._index, eff_dims_indexed):
                if ei != self.cond_dim:
                    aid = anno_map.get((fi,ri), "")
                    self.outliers.append({"file": self.files[fi], "row": int(ri),
                                        "anno_id": str(aid), "eff_dim": int(ei)})

            if self.exclude_outliers:
                oldN = len(self._index)
                self._index = [(pfi, pri) for (pfi, pri), ei in zip(self._index, eff_dims_indexed) if ei == self.cond_dim]
                print(f"[PartNet-H5:{self.split}] exclude_outliers=True -> kept {len(self._index)}/{oldN}; "
                    f"outliers={len(self.outliers)} (canon_dim={self.cond_dim}, policy={self.cond_dim_policy})")
            else:
                print(f"[PartNet-H5:{self.split}] canon_dim={self.cond_dim} (policy={self.cond_dim_policy}); "
                    f"dim_hist={dict(sorted(dim_hist.items()))}; outliers={len(self.outliers)}")

        # Condition dimension（直接由 enc_mode 决定）
        self.cond_dim = _get_cond_dim(self.motor_enc) if (self.cond_mode == "motors" and self._has_motors) else 0

        # Dataset-level stats
        self.all_points_mean = np.zeros(3, dtype=np.float32)
        self.all_points_std = np.ones(3, dtype=np.float32)
        if not self.use_norm:
            try:
                with h5py.File(self.files[0], "r") as f0:
                    if ("center" in f0) and ("scale" in f0):
                        c0 = np.asarray(f0["center"][0], dtype=np.float32)
                        s0 = float(np.asarray(f0["scale"][0], dtype=np.float32))
                        self.all_points_mean = c0
                        self.all_points_std = np.array([s0, s0, s0], dtype=np.float32)
            except Exception:
                pass

        # default shuffle_idx before subsetting
        self.shuffle_idx = np.arange(len(self._index), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_open(self, fi: int):
        h = self._handles.get(fi, None)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    @staticmethod
    def _sample_idx(N: int, K: int) -> np.ndarray:
        if K <= 0: return np.empty((0,), dtype=np.int64)
        if K <= N: return np.random.choice(N, K, replace=False)
        base = np.arange(N, dtype=np.int64)
        extra = np.random.choice(N, K - N, replace=True)
        return np.concatenate([base, extra], axis=0)

    def __getitem__(self, idx: int):
        fi, ri = self._index[idx]
        f = self._ensure_open(fi)
        key = self._key_points_map[fi]
        pts = f[key][ri].astype(np.float32)  # (N,3)
        N = pts.shape[0]

        tr_idx = self._sample_idx(N, self.tr_n)
        te_idx = self._sample_idx(N, self.te_n)
        tr_pts = pts[tr_idx]
        te_pts = pts[te_idx]

        item = {
            "idx": idx,
            "train_points": torch.from_numpy(tr_pts),
            "test_points": torch.from_numpy(te_pts),
            "mean": torch.from_numpy(self.all_points_mean.reshape(1, 3)),
            "std": torch.from_numpy(self.all_points_std.reshape(1, 3)),
        }

        if self.expand_stats and ("center" in f) and ("scale" in f):
            center = f["center"][ri].astype(np.float32)
            scale = np.asarray([f["scale"][ri]], dtype=np.float32)
            item["center"] = torch.from_numpy(center)
            item["scale"] = torch.from_numpy(scale)

        if self.cond_mode == "motors" and self._has_motors and ("motors" in f):
            m = f["motors"][ri].astype(np.float32)            # (6,) or (9,)
            cond = _enc_motors(
                m, self.motor_enc,
                mod2_offset_deg=self.motor_mod2_offset_deg,
                max_pos=self.motor_max_pos,
                mod3_offset_deg=self.motor_mod3_offset_deg,
            )
            item["cond"] = torch.from_numpy(cond.astype(np.float32))

        return item

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if handles:
            for h in list(handles.values()):
                try:
                    h.close()
                except Exception:
                    pass
            handles.clear()


# --- 替换 datasets.py 里原有的 PartNetH5PointClouds 定义，保持文件其余部分不变 ---

def _rgb_to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mx = float(np.max(arr)) if arr.size > 0 else 1.0
    if mx > 1.0:  # 多数数据是 0..255
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)

class PartNetH5PointClouds(Dataset):
    """
    Dataset for PartNet category H5:
      - keys: data / data_norm / motors / (optional anno_id, center, scale, rgb)
      - condition: 基于 motors 的“原始/归一化”向量（将 NaN 视作缺失；统一到规范维度）
      - 新增：
          * 支持读取每点 rgb（若存在），并与点抽样对齐，输出 train_rgb/test_rgb
          * 自动统计每个样本“有效关节数”，选择规范维度（policy: 'mode' 或 'max'）
          * 可选上报异常样本（维度 != 规范维度）
    """
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 use_norm: bool = True, expand_stats: bool = False,
                 tr_sample_size: int = 2048, te_sample_size: int = 2048,
                 keep_annos: Optional[Set[str]] = None,
                 cond_dim_policy: str = "mode",
                 exclude_outliers: bool = False,
                 report_file: str = "",
                 report_topk: int = 200,
                 files=None, **kwargs) -> None:
        super().__init__()
        self._handles = {}
        self.split = str(split)
        self.use_norm = bool(use_norm)
        self.expand_stats = bool(expand_stats)
        self.tr_n = int(tr_sample_size)
        self.te_n = int(te_sample_size)
        self.data_dir = os.path.abspath(data_dir)
        self.keep_annos = set(keep_annos or [])
        self.cond_dim_policy = str(cond_dim_policy).lower()
        assert self.cond_dim_policy in {"mode", "max"}
        self.exclude_outliers = bool(exclude_outliers)
        self.report_file = str(report_file)
        self.report_topk = int(report_topk)

        # 发现 shard
        if files is not None:
            if isinstance(files, (list, tuple)):
                self.files = sorted(set([str(x) for x in files]))
            elif isinstance(files, str):
                self.files = sorted(set(glob.glob(files)))
            else:
                raise TypeError("files must be None, list/tuple of paths, or a glob pattern string")
        else:
            patterns = [
                os.path.join(self.data_dir, self.split, "shard-*.h5"),
                os.path.join(self.data_dir, self.split, "*.h5"),
                os.path.join(self.data_dir, self.split, "*.hdf5"),
            ]
            flist = []
            for p in patterns: flist.extend(glob.glob(p))
            self.files = sorted(set(flist))
        if not self.files:
            raise FileNotFoundError(f"[PartNet-H5] No shards under '{self.data_dir}/{self.split}'")

        # 扫描 & 构建索引；统计 motors 有效维度；探测是否包含 rgb
        self._index = []
        self._key_points_map = {}
        self._has_motors = False
        self._has_rgb = False

        eff_dims = []
        eff_meta = []
        dim_hist = {}

        n_before = 0; n_after = 0
        for fi, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                key = "data_norm" if (self.use_norm and "data_norm" in f) else "data"
                if key not in f:
                    raise KeyError(f"[PartNet-H5] Missing key '{key}' in {fp}")
                B = int(f[key].shape[0])
                self._key_points_map[fi] = key

                if "rgb" in f:
                    self._has_rgb = True  # 至少这个 shard 有 rgb

                annos = None
                if "anno_id" in f:
                    annos = list(f["anno_id"][:])
                    annos = [a.decode("utf-8", "ignore") if isinstance(a, (bytes, np.bytes_)) else str(a) for a in annos]

                if "motors" in f:
                    self._has_motors = True
                    M = f["motors"][()]
                    if np.issubdtype(M.dtype, np.floating):
                        isn = np.isnan(M)
                        eff = (~isn).sum(axis=1).astype(int) if isn.ndim == 2 else np.array([int((~isn).sum())]*B)
                    else:
                        eff = np.array([M.shape[1]]*B, dtype=int)
                    for i in range(B):
                        ei = int(eff[i]); eff_dims.append(ei)
                        aid = (annos[i] if annos is not None else "")
                        eff_meta.append((fi, i, aid))
                        dim_hist[ei] = dim_hist.get(ei, 0) + 1

                # 过滤 anno
                if self.keep_annos and annos is not None:
                    for i in range(B):
                        n_before += 1
                        if annos[i] in self.keep_annos:
                            self._index.append((fi, i)); n_after += 1
                    continue
                self._index.extend([(fi, i) for i in range(B)])
                n_before += B; n_after += B

        # 规范 joints 维度
        if self._has_motors and eff_dims:
            if self.cond_dim_policy == "mode":
                canon_dim = max(dim_hist.items(), key=lambda kv: kv[1])[0]
            else:
                canon_dim = max(eff_dims)
        else:
            canon_dim = 0
        self.cond_dim = int(canon_dim)

        # 异常样本
        self.outliers = []
        if self._has_motors and eff_dims:
            for (fi, ri, aid), ei in zip(eff_meta, eff_dims):
                if ei != self.cond_dim:
                    self.outliers.append({
                        "file": self.files[fi], "row": int(ri),
                        "anno_id": str(aid), "eff_dim": int(ei)
                    })
            if self.exclude_outliers:
                oldN = len(self._index)
                self._index = [(fi, ri) for (fi, ri), ei in zip(self._index, eff_dims) if ei == self.cond_dim]
                print(f"[PartNet-H5:{self.split}] exclude_outliers=True -> kept {len(self._index)}/{oldN}; "
                      f"outliers={len(self.outliers)} (canon_dim={self.cond_dim}, policy={self.cond_dim_policy})")
            else:
                print(f"[PartNet-H5:{self.split}] canon_dim={self.cond_dim} (policy={self.cond_dim_policy}); "
                      f"dim_hist={dict(sorted(dim_hist.items()))}; outliers={len(self.outliers)}")

        # 数据集级别的平移/尺度提示
        self.all_points_mean = np.zeros(3, dtype=np.float32)
        self.all_points_std = np.ones(3, dtype=np.float32)
        if not self.use_norm and self.files:
            try:
                with h5py.File(self.files[0], "r") as f0:
                    if ("center" in f0) and ("scale" in f0):
                        c0 = np.asarray(f0["center"][0], dtype=np.float32)
                        s0 = float(np.asarray(f0["scale"][0], dtype=np.float32))
                        self.all_points_mean = c0
                        self.all_points_std = np.array([s0, s0, s0], dtype=np.float32)
            except Exception:
                pass

        self.shuffle_idx = np.arange(len(self._index), dtype=np.int64)

        # 可选报告
        if self.report_file:
            try:
                os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
                import json
                rep = {
                    "split": self.split,
                    "canon_dim": self.cond_dim,
                    "policy": self.cond_dim_policy,
                    "dim_hist": dim_hist,
                    "outliers_count": len(self.outliers),
                    "outliers_preview": self.outliers[:min(self.report_topk, len(self.outliers))],
                }
                with open(self.report_file, "w", encoding="utf-8") as f:
                    json.dump(rep, f, ensure_ascii=False, indent=2)
                print(f"[PartNet-H5:{self.split}] wrote report -> {self.report_file}")
            except Exception as e:
                print(f"[WARN] failed to write report: {e}")

        # 标记 rgb 能力（供上层选择是否用 rgb 训练）
        self.has_rgb = bool(self._has_rgb)

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_open(self, fi: int):
        h = self._handles.get(fi, None)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    @staticmethod
    def _sample_idx(N: int, K: int) -> np.ndarray:
        if K <= 0: return np.empty((0,), dtype=np.int64)
        if K <= N: return np.random.choice(N, K, replace=False)
        base = np.arange(N, dtype=np.int64)
        extra = np.random.choice(N, K - N, replace=True)
        return np.concatenate([base, extra], axis=0)

    def __getitem__(self, idx: int):
        fi, ri = self._index[idx]
        f = self._ensure_open(fi)
        key = self._key_points_map[fi]
        pts = f[key][ri].astype(np.float32)   # (N,3)
        N = pts.shape[0]

        tr_idx = self._sample_idx(N, self.tr_n)
        te_idx = self._sample_idx(N, self.te_n)
        tr_pts = pts[tr_idx]
        te_pts = pts[te_idx]

        item = {
            "idx": idx,
            "train_points": torch.from_numpy(tr_pts),
            "test_points": torch.from_numpy(te_pts),
            "mean": torch.from_numpy(self.all_points_mean.reshape(1, 3)),
            "std": torch.from_numpy(self.all_points_std.reshape(1, 3)),
        }

        if self.expand_stats and ("center" in f) and ("scale" in f):
            center = f["center"][ri].astype(np.float32)
            scale = np.asarray([f["scale"][ri]], dtype=np.float32)
            item["center"] = torch.from_numpy(center)
            item["scale"] = torch.from_numpy(scale)

        # 条件：motors -> 规范维度
        if self._has_motors and ("motors" in f) and self.cond_dim > 0:
            m = f["motors"][ri].astype(np.float32).reshape(-1)
            if np.isnan(m).any():
                m = np.nan_to_num(m, nan=0.0)
            d = m.shape[0]
            if d < self.cond_dim:
                pad = np.zeros(self.cond_dim, dtype=np.float32); pad[:d] = m; m = pad
            elif d > self.cond_dim:
                m = m[:self.cond_dim]
            item["cond"] = torch.from_numpy(m.astype(np.float32))

        # 新增：rgb（若存在）
        if self.has_rgb and ("rgb" in f):
            rgb_all = f["rgb"][ri]             # (N,3), uint8 或 float
            tr_rgb = _rgb_to_float01(rgb_all[tr_idx])
            te_rgb = _rgb_to_float01(rgb_all[te_idx])
            item["train_rgb"] = torch.from_numpy(tr_rgb.astype(np.float32))
            item["test_rgb"]  = torch.from_numpy(te_rgb.astype(np.float32))

        if "anno_id" in f:
            try:
                aid = f["anno_id"][ri]
                if isinstance(aid, (bytes, np.bytes_)):
                    aid = aid.decode("utf-8", "ignore")
                else:
                    aid = str(aid)
                item["anno_id"] = aid
            except Exception:
                pass
        return item

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if handles:
            for h in list(handles.values()):
                try: h.close()
                except Exception: pass
            handles.clear()


# ----------------------------- Factory & loaders -----------------------------

def get_datasets(args):
    ds_type = getattr(args, "dataset_type", "tdcr_h5").lower()
    keep_ids, keep_splits = _parse_keep_annos(args)

    if ds_type == "tdcr_h5":
        if keep_ids:
            print("[WARN] --keep-anno 仅对 PartNet H5 生效；TDCR H5 未包含 anno_id，已忽略。")
        tr_dataset = TDCRH5PointClouds(
            data_dir=args.data_dir, split="train",
            use_norm=getattr(args, "tdcr_use_norm", True),
            expand_stats=getattr(args, "tdcr_expand_stats", False),
            tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
            te_sample_size=getattr(args, "te_max_sample_points", 2048),
            cond_mode=getattr(args, "cond_mode", "motors"),
            motor_enc=getattr(args, "motor_enc", "raw6+geom"),
            motor_mod2_offset_deg=getattr(args, "motor_mod2_offset_deg", 0.0),
            motor_mod3_offset_deg=getattr(args, "motor_mod3_offset_deg", 0.0),
            motor_max_pos=getattr(args, "motor_max_pos", 0.4),
        )
        # Prefer val/ if exists, else test/
        val_dir = Path(args.data_dir, "val")
        split = "val" if val_dir.exists() and any(val_dir.glob("shard-*.h5")) else "test"
        te_dataset = TDCRH5PointClouds(
            data_dir=args.data_dir, split=split,
            use_norm=getattr(args, "tdcr_use_norm", True),
            expand_stats=getattr(args, "tdcr_expand_stats", False),
            tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
            te_sample_size=getattr(args, "te_max_sample_points", 2048),
            cond_mode=getattr(args, "cond_mode", "motors"),
            motor_enc=getattr(args, "motor_enc", "raw6+geom"),
            motor_mod2_offset_deg=getattr(args, "motor_mod2_offset_deg", 0.0),
            motor_mod3_offset_deg=getattr(args, "motor_mod3_offset_deg", 0.0),
            motor_max_pos=getattr(args, "motor_max_pos", 0.4),
        )
    elif ds_type == "partnet_h5":
        tr_dataset = PartNetH5PointClouds(
            data_dir=args.data_dir, split="train",
            use_norm=getattr(args, "tdcr_use_norm", True),
            expand_stats=getattr(args, "tdcr_expand_stats", False),
            tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
            te_sample_size=getattr(args, "te_max_sample_points", 2048),
            keep_annos=(keep_ids if "train" in keep_splits else None),
            # 新增（都带缺省值；不想改训练脚本 CLI 的话，也可以先不从命令行透传）
            cond_dim_policy=getattr(args, "partnet_cond_policy", "mode"),
            exclude_outliers=getattr(args, "partnet_exclude_outliers", False),
            report_file=getattr(args, "partnet_report_file_train", ""),
        )
        val_dir = Path(args.data_dir, "val")
        split = "val" if val_dir.exists() and any(val_dir.glob("shard-*.h5")) else "test"
        te_dataset = PartNetH5PointClouds(
            data_dir=args.data_dir, split=split,
            use_norm=getattr(args, "tdcr_use_norm", True),
            expand_stats=getattr(args, "tdcr_expand_stats", False),
            tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
            te_sample_size=getattr(args, "te_max_sample_points", 2048),
            keep_annos=(keep_ids if (split in keep_splits) else None),
            cond_dim_policy=getattr(args, "partnet_cond_policy", "mode"),
            exclude_outliers=False,  # eval 默认不断，只报告
            report_file=getattr(args, "partnet_report_file_eval", ""),
        )
        base = getattr(tr_dataset, "dataset", tr_dataset)
        args.has_rgb = bool(getattr(base, "has_rgb", False))
        args.cond_dim = getattr(base, "cond_dim", 0)
    else:
        raise ValueError(f"Unknown --dataset_type: {ds_type}")

    # 训练集子集（可选）
    sel_idx = _pick_subset_indices(args, len(tr_dataset))
    if sel_idx is not None:
        tr_dataset = SubsetWithAttrs(tr_dataset, sel_idx.tolist())
        _attach_shuffle_idx(tr_dataset, sel_idx)
    else:
        _attach_shuffle_idx(tr_dataset, np.arange(len(tr_dataset), dtype=np.int64))

    # 确保测试集有 shuffle_idx
    if not hasattr(te_dataset, "shuffle_idx"):
        setattr(te_dataset, "shuffle_idx", np.arange(len(te_dataset), dtype=np.int64))

    # 回传 cond_dim 给训练脚本
    base = getattr(tr_dataset, "dataset", tr_dataset)
    args.cond_dim = getattr(base, "cond_dim", 0)

    return tr_dataset, te_dataset


def get_data_loaders(args):
    tr_dataset, te_dataset = get_datasets(args)

    train_loader = torch_data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed,
    )
    train_unshuffle_loader = torch_data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed,
    )
    test_loader = torch_data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
        worker_init_fn=init_np_seed,
    )

    return {
        "test_loader": test_loader,
        "train_loader": train_loader,
        "train_unshuffle_loader": train_unshuffle_loader,
    }
