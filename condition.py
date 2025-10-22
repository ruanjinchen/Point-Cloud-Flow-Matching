
import numpy as np

def _module_resultant(m123, angles_deg, offset_deg=0.0):
    """
    与你现有 2 节实现一致的单段三拉索合成（保持语义不变）：
      vec = [cos, sin] @ m123      # 注意这里不做去均值，沿用你当前版本
      T   = sum(m)                 # 总量（你当前就是 sum/mean 的一类量）
      A   = std-like(m)            # 幅度（与现有 A 的定义一致）
    """
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float32) + float(offset_deg))
    c = np.stack([np.cos(th), np.sin(th)], axis=0)       # (2,3)
    vec = c @ m123.astype(np.float32)                   # (2,)
    T   = float(np.sum(m123))
    mean = T / 3.0 if T > 0 else 0.0
    A   = float(np.sqrt(np.mean((m123 - mean) ** 2)))
    return vec.astype(np.float32), T, A

def encode_motors(motors: np.ndarray,
                  enc_mode: str = "raw6+geom",
                  mod2_offset_deg: float = 0.0,
                  max_pos: float = 0.04,
                  mod3_offset_deg: float = 0.0) -> np.ndarray:
    """
    统一的电机编码（与训练一致）：
      - 输入 motors 支持 6（两段×3）或 9（3 段×3）
      - enc_mode: "raw6", "geom", "raw6+geom"（两段）
                  "raw9", "geom3", "raw9+geom3"（三段）
      - 偏航角：第二段用 mod2_offset_deg；第三段用 mod3_offset_deg
      - 归一化：按 max_pos 到 [0,1] 并 clip（与采集/训练一致）
    """
    m = np.asarray(motors, dtype=np.float32).reshape(-1)
    assert m.shape[0] in (6, 9), f"motors dim must be 6 or 9, got {m.shape[0]}"
    nseg = 2 if m.shape[0] == 6 else 3

    # 归一化
    mn = (m / float(max_pos)).clip(0.0, 1.0).astype(np.float32)

    # 与你当前版本保持相同的相位基准（180,300,60）
    base_angles = [180.0, 300.0, 60.0]

    # 第一段
    v1, T1, A1 = _module_resultant(mn[0:3], base_angles, 0.0)
    v2 = np.zeros(2, np.float32); T2 = 0.0; A2 = 0.0
    v3 = np.zeros(2, np.float32); T3 = 0.0; A3 = 0.0

    # 第二段
    if nseg >= 2:
        v2, T2, A2 = _module_resultant(mn[3:6], base_angles, mod2_offset_deg)

    # 第三段
    if nseg == 3:
        v3, T3, A3 = _module_resultant(mn[6:9], base_angles, mod3_offset_deg)

    # 组装 geom
    if nseg == 2:
        geom = np.concatenate([v1, [T1, A1],
                               v2, [T2, A2],
                               [T1 - T2, T1 + T2]], axis=0).astype(np.float32)   # 10 维
        if enc_mode == "raw6":         return mn
        if enc_mode == "geom":         return geom
        if enc_mode == "raw6+geom":    return np.concatenate([mn, geom], axis=0)
        raise ValueError(f"unknown enc_mode={enc_mode} for 2-seg")
    else:
        geom3 = np.concatenate([v1, [T1, A1],
                                v2, [T2, A2],
                                v3, [T3, A3],
                                [T1 - T2, T2 - T3, T1 - T3, T1 + T2 + T3]],
                               axis=0).astype(np.float32)                        # 16 维
        if enc_mode == "raw9":         return mn
        if enc_mode == "geom3":        return geom3
        if enc_mode == "raw9+geom3":   return np.concatenate([mn, geom3], axis=0)  # 25 维
        raise ValueError(f"unknown enc_mode={enc_mode} for 3-seg")

def get_cond_dim(enc_mode: str) -> int:
    """
    返回 cond 维度（与 encode_motors 一致）
    """
    table = {
        "raw6": 6, "geom": 10, "raw6+geom": 16,
        "raw9": 9, "geom3": 16, "raw9+geom3": 25,
    }
    if enc_mode in table:
        return table[enc_mode]
    # 兜底：用 0 填充跑一遍
    n = 9 if (("raw9" in enc_mode) or ("geom3" in enc_mode)) else 6
    return int(encode_motors(np.zeros(n, np.float32), enc_mode).shape[0])
