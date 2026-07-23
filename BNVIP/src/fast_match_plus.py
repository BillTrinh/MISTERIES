"""Stronger fast similarity matching using dance-style scoring (no DTW).

Compared with fast_match.py (left unchanged):
  - no whole-sequence cosine
  - dance-like pose: weighted landmark distance + finger/palm angles
  - dance-like motion: per-landmark delta direction + magnitude
  - small lag search (±frames) instead of full DTW
  - motion gate, peak picking, same-word merge
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sequence_utils import (
    SIGN_DIM,
    T_FIXED,
    normalize_sequence,
    preprocess,
    resample_sequence,
)

# Same blend as dance_scorer alignment when motion is informative.
POSE_WEIGHT = 0.75
MOTION_WEIGHT = 0.25
COORD_SIGMA = 0.65
ANGLE_SIGMA = np.deg2rad(35.0)
MOTION_FLOOR = 0.025  # same idea as dance_scorer template magnitude gate
N_LANDMARKS = 42  # 21 per hand × 2
SCORE_FRAMES = 8  # subsample frames for angle/motion (keeps dance formula, stays fast)


def _hand_angle_triplets() -> tuple[tuple[int, int, int], ...]:
    """MediaPipe hand joints for both hands (offsets 0 and 21)."""
    one_hand = (
        (1, 2, 3),
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20),
        (0, 5, 9),
        (0, 9, 13),
        (0, 13, 17),
    )
    triplets = []
    for offset in (0, 21):
        for a, b, c in one_hand:
            triplets.append((a + offset, b + offset, c + offset))
    return tuple(triplets)


ANGLE_TRIPLETS = _hand_angle_triplets()
# Prefer fingertips / distal joints, like dance prefers wrists/ankles.
_HAND_W = np.asarray(
    [
        0.8,  # wrist
        0.7,
        1.0,
        1.1,
        1.3,  # thumb
        0.9,
        1.0,
        1.1,
        1.3,  # index
        0.9,
        1.0,
        1.1,
        1.3,  # middle
        0.9,
        1.0,
        1.1,
        1.3,  # ring
        0.9,
        1.0,
        1.1,
        1.3,  # pinky
    ],
    dtype=np.float32,
)
KEYPOINT_WEIGHTS = np.concatenate([_HAND_W, _HAND_W])
ANGLE_WEIGHTS = np.tile(
    np.asarray([1.1, 1.3, 1.1, 1.3, 1.1, 1.3, 1.1, 1.3, 1.1, 1.3, 1.0, 1.0, 1.0], dtype=np.float32),
    2,
)


def _to_points(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 126) -> points (T, 42, 3), visible (T, 42)."""
    pts = np.asarray(seq, dtype=np.float32).reshape(-1, N_LANDMARKS, 3)
    visible = ~np.all(pts == 0, axis=2)
    return pts, visible


def _subsample_indices(n: int, k: int = SCORE_FRAMES) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.linspace(0, n - 1, k).astype(int)


def _vectorized_coordinate_score(
    cand_pts: np.ndarray,
    cand_vis: np.ndarray,
    tmpl_pts: np.ndarray,
    tmpl_vis: np.ndarray,
) -> tuple[float, float]:
    """Average dance-style coordinate score over all frames."""
    common = cand_vis & tmpl_vis  # (T, 42)
    weights = KEYPOINT_WEIGHTS[None, :] * common
    weight_sum = weights.sum(axis=1)
    valid_frames = weight_sum >= (8.0 * float(KEYPOINT_WEIGHTS.mean()))
    if not np.any(valid_frames):
        return 0.0, 0.0

    diff = cand_pts - tmpl_pts
    dist = np.linalg.norm(diff, axis=2)  # (T, 42)
    dist = np.where(common, dist, 0.0)
    coordinate_error = np.zeros(len(cand_pts), dtype=np.float32)
    nonzero = weight_sum > 1e-6
    coordinate_error[nonzero] = (dist * weights).sum(axis=1)[nonzero] / weight_sum[nonzero]
    scores = np.exp(-((coordinate_error / COORD_SIGMA) ** 2))
    scores = scores[valid_frames]
    coverage = float(
        np.mean(weight_sum[valid_frames] / np.maximum(KEYPOINT_WEIGHTS.sum() * 0.5, 1e-6))
    )
    coverage = float(np.clip(coverage, 0.0, 1.0))
    return float(np.mean(scores)), coverage


def _batch_joint_angles(points: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """points (T, 42, 3), visible (T, 42) -> angles (T, A), valid (T, A)."""
    t = points.shape[0]
    a_count = len(ANGLE_TRIPLETS)
    angles = np.zeros((t, a_count), dtype=np.float32)
    valid = np.zeros((t, a_count), dtype=bool)
    for i, (pa, pv, pc) in enumerate(ANGLE_TRIPLETS):
        ok = visible[:, pa] & visible[:, pv] & visible[:, pc]
        if not np.any(ok):
            continue
        va = points[:, pa] - points[:, pv]
        vc = points[:, pc] - points[:, pv]
        denom = np.linalg.norm(va, axis=1) * np.linalg.norm(vc, axis=1)
        good = ok & (denom > 1e-6)
        if not np.any(good):
            continue
        cos = np.sum(va[good] * vc[good], axis=1) / denom[good]
        cos = np.clip(cos, -1.0, 1.0)
        angles[good, i] = np.arccos(cos)
        valid[good, i] = True
    return angles, valid


def _angle_score_over_frames(
    cand_pts: np.ndarray,
    cand_vis: np.ndarray,
    tmpl_pts: np.ndarray,
    tmpl_vis: np.ndarray,
) -> float:
    cand_ang, cand_ok = _batch_joint_angles(cand_pts, cand_vis)
    tmpl_ang, tmpl_ok = _batch_joint_angles(tmpl_pts, tmpl_vis)
    common = cand_ok & tmpl_ok
    if not np.any(common):
        return -1.0
    diffs = np.abs(cand_ang - tmpl_ang)
    per = np.exp(-((diffs / ANGLE_SIGMA) ** 2))
    # Weighted average over all valid angle observations.
    w = ANGLE_WEIGHTS[None, :] * common
    denom = float(np.sum(w))
    if denom < 1e-6:
        return -1.0
    return float(np.sum(per * w) / denom)


def _motion_score_sequence(
    cand_pts: np.ndarray,
    cand_vis: np.ndarray,
    tmpl_pts: np.ndarray,
    tmpl_vis: np.ndarray,
) -> float | None:
    """Dance-like motion over consecutive subsampled frames."""
    if len(cand_pts) < 2:
        return None
    scores = []
    for i in range(len(cand_pts) - 1):
        common = cand_vis[i] & cand_vis[i + 1] & tmpl_vis[i] & tmpl_vis[i + 1]
        if int(np.sum(common)) < 4:
            continue
        c_delta = cand_pts[i + 1, common] - cand_pts[i, common]
        t_delta = tmpl_pts[i + 1, common] - tmpl_pts[i, common]
        c_mag = np.linalg.norm(c_delta, axis=1)
        t_mag = np.linalg.norm(t_delta, axis=1)
        moving = t_mag >= MOTION_FLOOR
        if int(np.sum(moving)) < 2:
            continue
        c_delta = c_delta[moving]
        t_delta = t_delta[moving]
        c_mag = c_mag[moving]
        t_mag = t_mag[moving]
        denom = np.maximum(c_mag * t_mag, 1e-6)
        direction = (np.clip(np.sum(c_delta * t_delta, axis=1) / denom, -1.0, 1.0) + 1.0) * 0.5
        magnitude = np.minimum(c_mag, t_mag) / np.maximum(c_mag, t_mag)
        scores.append(float(np.mean(0.70 * direction + 0.30 * magnitude)))
    if not scores:
        return None
    return float(np.mean(scores))


def combined_similarity(window_fixed: np.ndarray, template_fixed: np.ndarray) -> dict:
    """Dance-style pose + motion over a fixed-length window vs template."""
    cand_pts, cand_vis = _to_points(window_fixed)
    tmpl_pts, tmpl_vis = _to_points(template_fixed)
    n = min(len(cand_pts), len(tmpl_pts))
    if n < 2:
        return {"score": 0.0, "pose": 0.0, "motion": 0.0, "angle": 0.0, "coordinate": 0.0}

    cand_pts = cand_pts[:n]
    cand_vis = cand_vis[:n]
    tmpl_pts = tmpl_pts[:n]
    tmpl_vis = tmpl_vis[:n]

    coordinate_score, _coverage = _vectorized_coordinate_score(
        cand_pts, cand_vis, tmpl_pts, tmpl_vis
    )
    if coordinate_score <= 0:
        return {"score": 0.0, "pose": 0.0, "motion": 0.0, "angle": 0.0, "coordinate": 0.0}

    idx = _subsample_indices(n, SCORE_FRAMES)
    angle_score = _angle_score_over_frames(
        cand_pts[idx], cand_vis[idx], tmpl_pts[idx], tmpl_vis[idx]
    )
    motion_score = _motion_score_sequence(
        cand_pts[idx], cand_vis[idx], tmpl_pts[idx], tmpl_vis[idx]
    )

    # Require dance-like evidence: valid finger angles AND motion pairs.
    if angle_score < 0 or motion_score is None:
        return {
            "score": 0.0,
            "pose": 0.0,
            "motion": 0.0,
            "angle": 0.0,
            "coordinate": float(coordinate_score),
        }

    pose_score = 0.60 * angle_score + 0.40 * coordinate_score
    score = POSE_WEIGHT * pose_score + MOTION_WEIGHT * motion_score

    return {
        "score": float(score),
        "pose": float(pose_score),
        "motion": float(motion_score),
        "angle": float(angle_score),
        "coordinate": float(coordinate_score),
    }


def coordinate_similarity(window_fixed: np.ndarray, template_fixed: np.ndarray) -> float:
    """Fast dance-style coordinate-only score (coarse filter)."""
    cand_pts, cand_vis = _to_points(window_fixed)
    tmpl_pts, tmpl_vis = _to_points(template_fixed)
    n = min(len(cand_pts), len(tmpl_pts))
    if n < 2:
        return 0.0
    score, _ = _vectorized_coordinate_score(
        cand_pts[:n], cand_vis[:n], tmpl_pts[:n], tmpl_vis[:n]
    )
    return float(score)


# --- template loading / scanning ------------------------------------------------


def load_word_templates(
    signs_dir: Path,
    max_templates_per_word: int = 4,
) -> dict[str, list[np.ndarray]]:
    """Load and cache preprocessed fixed-length templates from *.pkl."""
    signs_dir = Path(signs_dir)
    templates: dict[str, list[np.ndarray]] = {}
    for path in sorted(signs_dir.glob("*.pkl")):
        with open(path, "rb") as f:
            clips = pickle.load(f)
        if not clips:
            continue
        if len(clips) <= max_templates_per_word:
            chosen = clips
        else:
            idxs = np.linspace(0, len(clips) - 1, max_templates_per_word).astype(int)
            chosen = [clips[int(i)] for i in idxs]
        templates[path.stem.upper()] = [
            preprocess(np.asarray(clip, dtype=np.float32), T_FIXED) for clip in chosen
        ]
    if not templates:
        raise FileNotFoundError(f"No sign templates found in {signs_dir}")
    return templates


def _window_motion(window: np.ndarray) -> float:
    w = np.asarray(window, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(w) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(w[1:] - w[:-1], axis=1)))


def _prepare_window(
    window_or_fixed: np.ndarray,
    *,
    already_fixed: bool = False,
    stream_already_normalized: bool = False,
) -> np.ndarray:
    if already_fixed:
        return window_or_fixed
    if stream_already_normalized:
        return resample_sequence(window_or_fixed, T_FIXED)
    return preprocess(window_or_fixed, T_FIXED)


def _best_template_score(
    window_or_fixed: np.ndarray,
    templates_fixed: list[np.ndarray],
    *,
    already_fixed: bool = False,
    stream_already_normalized: bool = False,
    coarse_gate: float = 0.58,
) -> dict:
    """Coarse coordinate filter, then full dance score on the best template(s)."""
    window_fixed = _prepare_window(
        window_or_fixed,
        already_fixed=already_fixed,
        stream_already_normalized=stream_already_normalized,
    )
    if not templates_fixed:
        return {"score": 0.0, "pose": 0.0, "motion": 0.0, "angle": 0.0, "coordinate": 0.0}

    ranked = []
    for tmpl in templates_fixed:
        coord = coordinate_similarity(window_fixed, tmpl)
        ranked.append((coord, tmpl))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked[0][0] < coarse_gate:
        # Reject without full scoring; do not promote coarse coord to final score.
        return {
            "score": 0.0,
            "pose": 0.0,
            "motion": 0.0,
            "angle": 0.0,
            "coordinate": float(ranked[0][0]),
        }

    # Full dance score only on the best coarse template.
    comps = combined_similarity(window_fixed, ranked[0][1])
    comps["coordinate"] = max(comps.get("coordinate", 0.0), float(ranked[0][0]))
    return comps


def _hit_dict(s: int, e: int, fps: float, comps: dict, scale: float, lag: int) -> dict:
    return {
        "start_frame": s,
        "end_frame": e,
        "center_frame": (s + e) // 2,
        "start_seconds": s / fps,
        "end_seconds": e / fps,
        "score": comps["score"],
        "pose": comps["pose"],
        "motion": comps["motion"],
        "angle": comps.get("angle", 0.0),
        "coordinate": comps.get("coordinate", 0.0),
        "scale": float(scale),
        "lag": int(lag),
    }


def scan_stream_for_word(
    stream: np.ndarray,
    templates_fixed: list[np.ndarray],
    fps: float,
    base_len: int = 40,
    scales: tuple[float, ...] = (0.85, 1.0, 1.2),
    hop_seconds: float = 0.20,
    min_similarity: float = 0.88,
    lag_frames: int = 1,
    min_motion: float = 0.010,
) -> list[dict]:
    """Slide multi-scale windows; dance-score with small lag search.

    Two-stage for speed: lag=0 first, then refine only promising windows.
    """
    stream = np.asarray(stream, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(stream) < 8 or not templates_fixed:
        return []

    hop = max(1, int(round(hop_seconds * fps)))
    refine_gate = min_similarity - 0.08
    keep_gate = min_similarity - 0.05
    best_by_center: dict[int, dict] = {}

    for scale in scales:
        win = int(round(base_len * scale))
        win = max(14, min(win, len(stream)))
        for start in range(0, len(stream) - win + 1, hop):
            window0 = stream[start : start + win]
            if _window_motion(window0) < min_motion:
                continue
            comps0 = _best_template_score(
                window0, templates_fixed, stream_already_normalized=True
            )
            best_local = _hit_dict(start, start + win, fps, comps0, scale, 0)

            if comps0["score"] >= refine_gate and lag_frames > 0:
                for lag in range(-lag_frames, lag_frames + 1):
                    if lag == 0:
                        continue
                    s = start + lag
                    e = s + win
                    if s < 0 or e > len(stream):
                        continue
                    window = stream[s:e]
                    if _window_motion(window) < min_motion:
                        continue
                    comps = _best_template_score(
                        window, templates_fixed, stream_already_normalized=True
                    )
                    if comps["score"] > best_local["score"]:
                        best_local = _hit_dict(s, e, fps, comps, scale, lag)

            if best_local["score"] < keep_gate:
                continue
            key = int(round(best_local["center_frame"] / max(hop, 1)))
            prev = best_by_center.get(key)
            if prev is None or best_local["score"] > prev["score"]:
                best_by_center[key] = best_local
    return list(best_by_center.values())

def _pick_peaks_per_word(
    timeline: list[dict],
    min_similarity: float,
    peak_margin: float = 0.02,
) -> list[dict]:
    by_word: dict[str, list[dict]] = {}
    for hit in timeline:
        by_word.setdefault(hit["word"], []).append(hit)

    peaks: list[dict] = []
    for word, items in by_word.items():
        items = sorted(items, key=lambda item: item["center_frame"])
        for idx, hit in enumerate(items):
            if hit["score"] < min_similarity:
                continue
            left = items[idx - 1]["score"] if idx > 0 else -1.0
            right = items[idx + 1]["score"] if idx + 1 < len(items) else -1.0
            if hit["score"] + 1e-6 >= left - peak_margin and hit["score"] + 1e-6 >= right - peak_margin:
                if hit["score"] >= left and hit["score"] >= right:
                    peaks.append(hit)
                elif hit["score"] >= min_similarity + 0.08:
                    peaks.append(hit)
        if not any(p["word"] == word for p in peaks):
            best = max(items, key=lambda item: item["score"])
            if best["score"] >= min_similarity + 0.04:
                peaks.append(best)
    return peaks


def pick_non_overlapping(
    candidates: list[dict],
    overlap_ratio: float = 0.30,
) -> list[dict]:
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for cand in ranked:
        ok = True
        for prev in kept:
            inter = min(cand["end_frame"], prev["end_frame"]) - max(
                cand["start_frame"], prev["start_frame"]
            )
            if inter <= 0:
                continue
            union = max(cand["end_frame"], prev["end_frame"]) - min(
                cand["start_frame"], prev["start_frame"]
            )
            if union > 0 and inter / union >= overlap_ratio:
                ok = False
                break
        if ok:
            kept.append(cand)
    kept.sort(key=lambda item: item["start_frame"])
    return kept


def _merge_same_word(detections: list[dict]) -> list[dict]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item["start_frame"])
    merged = [dict(ordered[0])]
    for hit in ordered[1:]:
        prev = merged[-1]
        same = hit["word"] == prev["word"]
        inter = min(hit["end_frame"], prev["end_frame"]) - max(
            hit["start_frame"], prev["start_frame"]
        )
        gap = hit["start_frame"] - prev["end_frame"]
        close = gap <= max(6, int(0.25 * (hit["end_frame"] - hit["start_frame"])))
        if same and (inter > 0 or close):
            if hit["score"] > prev["score"]:
                for key in (
                    "score",
                    "pose",
                    "motion",
                    "angle",
                    "coordinate",
                    "scale",
                    "lag",
                    "start_frame",
                    "end_frame",
                    "start_seconds",
                    "end_seconds",
                    "center_frame",
                ):
                    if key in hit:
                        prev[key] = hit[key]
            else:
                prev["end_frame"] = max(prev["end_frame"], hit["end_frame"])
                prev["end_seconds"] = max(prev["end_seconds"], hit["end_seconds"])
                prev["start_frame"] = min(prev["start_frame"], hit["start_frame"])
                prev["start_seconds"] = min(prev["start_seconds"], hit["start_seconds"])
        else:
            merged.append(dict(hit))
    return merged


def recognize_stream(
    stream: np.ndarray,
    word_templates: dict[str, list[np.ndarray]],
    fps: float = 30.0,
    min_similarity: float = 0.88,
    hop_seconds: float = 0.20,
    base_len: int = 40,
    lag_frames: int = 1,
) -> dict:
    """Detect multiple words with dance-style similarity and lag search."""
    stream = normalize_sequence(np.asarray(stream, dtype=np.float32))
    timeline: list[dict] = []
    for label, templates in word_templates.items():
        for hit in scan_stream_for_word(
            stream,
            templates,
            fps=fps,
            base_len=base_len,
            hop_seconds=hop_seconds,
            min_similarity=min_similarity,
            lag_frames=lag_frames,
        ):
            hit = dict(hit)
            hit["word"] = label
            timeline.append(hit)

    peaks = _pick_peaks_per_word(timeline, min_similarity=min_similarity)
    selected = pick_non_overlapping(peaks, overlap_ratio=0.30)
    selected = _merge_same_word(selected)
    selected = [hit for hit in selected if hit["score"] >= min_similarity]

    words_in_time = [item["word"] for item in selected]
    unique_words = list(dict.fromkeys(words_in_time))
    return {
        "detections": selected,
        "words_in_time": words_in_time,
        "unique_words": unique_words,
        "raw_candidate_count": len(timeline),
    }
