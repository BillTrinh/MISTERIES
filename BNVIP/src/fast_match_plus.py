"""Stronger fast similarity matching (no DTW).

Compared with fast_match.py (left unchanged):
  - pose cosine after normalize + resample
  - motion direction/magnitude similarity (frame deltas)
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

POSE_WEIGHT = 0.65
MOTION_WEIGHT = 0.35
MOTION_FLOOR = 0.018  # template delta magnitude to count as "moving"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def motion_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Motion score on fixed-length keypoint sequences."""
    a = np.asarray(seq_a, dtype=np.float32).reshape(-1, SIGN_DIM)
    b = np.asarray(seq_b, dtype=np.float32).reshape(-1, SIGN_DIM)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    da = a[1:] - a[:-1]
    db = b[1:] - b[:-1]
    mag_a = np.linalg.norm(da, axis=1)
    mag_b = np.linalg.norm(db, axis=1)
    moving = mag_b >= MOTION_FLOOR
    if int(np.sum(moving)) < 2:
        return _cosine(da, db)

    da = da[moving]
    db = db[moving]
    mag_a = mag_a[moving]
    mag_b = mag_b[moving]
    denom = np.maximum(mag_a * mag_b, 1e-6)
    direction = (np.clip(np.sum(da * db, axis=1) / denom, -1.0, 1.0) + 1.0) * 0.5
    magnitude = np.minimum(mag_a, mag_b) / np.maximum(mag_a, mag_b)
    return float(np.mean(0.70 * direction + 0.30 * magnitude))


def combined_similarity(window_fixed: np.ndarray, template_fixed: np.ndarray) -> dict:
    """Pose cosine + motion blend, both sequences already (T, 126)."""
    pose = _cosine(window_fixed, template_fixed)
    motion = motion_similarity(window_fixed, template_fixed)
    score = POSE_WEIGHT * pose + MOTION_WEIGHT * motion
    return {
        "score": float(score),
        "pose": float(pose),
        "motion": float(motion),
    }


def load_word_templates(
    signs_dir: Path,
    max_templates_per_word: int = 6,
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


def _best_template_score(
    window_or_fixed: np.ndarray,
    templates_fixed: list[np.ndarray],
    *,
    already_fixed: bool = False,
    stream_already_normalized: bool = False,
) -> dict:
    if already_fixed:
        window_fixed = window_or_fixed
    elif stream_already_normalized:
        window_fixed = resample_sequence(window_or_fixed, T_FIXED)
    else:
        window_fixed = preprocess(window_or_fixed, T_FIXED)
    best = {"score": -1.0, "pose": 0.0, "motion": 0.0}
    for tmpl in templates_fixed:
        comps = combined_similarity(window_fixed, tmpl)
        if comps["score"] > best["score"]:
            best = comps
    return best


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
        "scale": float(scale),
        "lag": int(lag),
    }


def scan_stream_for_word(
    stream: np.ndarray,
    templates_fixed: list[np.ndarray],
    fps: float,
    base_len: int = 40,
    scales: tuple[float, ...] = (0.80, 1.0, 1.25),
    hop_seconds: float = 0.15,
    min_similarity: float = 0.68,
    lag_frames: int = 2,
    min_motion: float = 0.010,
) -> list[dict]:
    """Slide multi-scale windows; score with pose+motion and small lag search.

    Two-stage for speed: lag=0 first, then refine only promising windows.
    """
    stream = np.asarray(stream, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(stream) < 8 or not templates_fixed:
        return []

    hop = max(1, int(round(hop_seconds * fps)))
    refine_gate = min_similarity - 0.10
    keep_gate = min_similarity - 0.06
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
    min_similarity: float = 0.68,
    hop_seconds: float = 0.15,
    base_len: int = 40,
    lag_frames: int = 2,
) -> dict:
    """Detect multiple words with pose+motion similarity and lag search."""
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
