"""Preprocess variable-length keypoint sequences for the dynamic recognizer.

- normalize_sequence: translation + scale invariant, but preserves the motion
  trajectory across the sequence (subtracts one mean / divides one std over all
  valid points, not per-frame). Missing landmarks (all-zero) stay zero.
- resample_sequence: linear-interpolate to a fixed number of frames so signs
  of slightly different duration align.
"""
import numpy as np

SIGN_DIM = 126
T_FIXED = 32


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32).reshape(-1, SIGN_DIM)
    T = seq.shape[0]
    pts = seq.reshape(T, 42, 3).copy()
    valid = ~np.all(pts == 0, axis=2)          # (T, 42)
    coords = pts[valid]
    if coords.shape[0] == 0:
        return seq
    mean = coords.mean(axis=0)
    std = coords.std() + 1e-6
    pts[valid] = (pts[valid] - mean) / std
    return pts.reshape(T, SIGN_DIM).astype(np.float32)


def resample_sequence(seq: np.ndarray, T: int = T_FIXED) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n == 0:
        return np.zeros((T, SIGN_DIM), dtype=np.float32)
    if n == T:
        return seq
    idx = np.linspace(0, n - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = (idx - lo)[:, None]
    return ((1 - frac) * seq[lo] + frac * seq[hi]).astype(np.float32)


def preprocess(seq, T: int = T_FIXED) -> np.ndarray:
    """Full pipeline: (variable, 126) -> normalized (T, 126)."""
    return resample_sequence(normalize_sequence(seq), T)


def augment_clip(clip, rng) -> np.ndarray:
    """Random variation that survives normalization: temporal crop (timing /
    start-end shift), coordinate jitter, and a small in-plane rotation.
    Global translation/scale are NOT added — normalize_sequence removes them."""
    clip = np.asarray(clip, dtype=np.float32).reshape(-1, SIGN_DIM)
    T = clip.shape[0]
    if T >= 12:                                   # temporal crop
        lo = int(rng.integers(0, T // 6 + 1))
        hi = T - int(rng.integers(0, T // 6 + 1))
        if hi - lo >= 8:
            clip = clip[lo:hi]
    pts = clip.reshape(-1, 42, 3).copy()
    valid = ~np.all(pts == 0, axis=2)             # (t, 42)
    if valid.any():
        pts[valid] += rng.normal(0, 0.012, pts.shape).astype(np.float32)[valid]
        ang = float(rng.uniform(-0.2, 0.2))       # ~±11 deg, x-y plane
        c, s = np.cos(ang), np.sin(ang)
        xy = pts[..., :2]
        m = xy[valid].mean(0)
        xyc = xy - m
        rot = np.stack([xyc[..., 0] * c - xyc[..., 1] * s,
                        xyc[..., 0] * s + xyc[..., 1] * c], -1) + m
        pts[..., :2] = np.where(valid[..., None], rot, 0.0)
    return pts.reshape(-1, SIGN_DIM)
