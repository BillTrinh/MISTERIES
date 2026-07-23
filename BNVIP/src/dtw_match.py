"""DTW helpers for continuous multi-sign recognition from keypoint streams."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sequence_utils import SIGN_DIM, normalize_sequence


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized L2 distance matrix. Shape (n, m)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2) + 1e-12)


def _pairwise_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Frame-wise cosine distance matrix: 1 - cos(a_i, b_j). Shape (n, m)."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.maximum(a_norm, 1e-8)
    b_unit = b / np.maximum(b_norm, 1e-8)
    a_ok = a_norm[:, 0] > 1e-6
    b_ok = b_norm[:, 0] > 1e-6
    cos = np.clip(a_unit @ b_unit.T, -1.0, 1.0)
    dist = 1.0 - cos
    # Missing-hand frames: both missing -> 0 cost; one missing -> far.
    both_missing = (~a_ok)[:, None] & (~b_ok)[None, :]
    one_missing = ((~a_ok)[:, None] & b_ok[None, :]) | (a_ok[:, None] & (~b_ok)[None, :])
    dist = np.where(both_missing, 0.0, dist)
    dist = np.where(one_missing, 1.0, dist)
    return dist.astype(np.float64)


def _pairwise_distance(a: np.ndarray, b: np.ndarray, metric: str = "l2") -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, SIGN_DIM)
    b = np.asarray(b, dtype=np.float32).reshape(-1, SIGN_DIM)
    if metric == "cosine":
        return _pairwise_cosine_distance(a, b)
    if metric == "l2":
        return _pairwise_l2(a, b)
    raise ValueError(f"Unknown DTW metric: {metric!r} (use 'l2' or 'cosine')")


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    band_ratio: float = 0.25,
    metric: str = "l2",
) -> float:
    """Band-limited DTW distance. Identical sequences -> ~0."""
    a = np.asarray(seq_a, dtype=np.float32).reshape(-1, SIGN_DIM)
    b = np.asarray(seq_b, dtype=np.float32).reshape(-1, SIGN_DIM)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    dist = _pairwise_distance(a, b, metric=metric)
    band = max(1, int(band_ratio * max(n, m)))
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost[i, j] = dist[i - 1, j - 1] + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
    if not np.isfinite(cost[n, m]):
        return float(np.min(dist) * max(n, m) / (n + m))
    return float(cost[n, m]) / (n + m)


def dtw_similarity(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    temperature: float | None = None,
    metric: str = "l2",
) -> float:
    """Map DTW distance to similarity in [0, 1]. Self-match ~= 1."""
    dist = dtw_distance(seq_a, seq_b, metric=metric)
    if not np.isfinite(dist):
        return 0.0
    if temperature is None:
        # Cosine distance is typically ~[0, 2]; L2 after normalize is smaller.
        temperature = 0.35 if metric == "cosine" else 0.20
    return float(np.exp(-dist / max(temperature, 1e-6)))


def load_word_templates(
    signs_dir: Path,
    max_templates_per_word: int = 4,
) -> dict[str, list[np.ndarray]]:
    """Load normalized word clips from data/signs/*.pkl."""
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
            normalize_sequence(np.asarray(clip, dtype=np.float32)) for clip in chosen
        ]
    if not templates:
        raise FileNotFoundError(f"No sign templates found in {signs_dir}")
    return templates


def scan_stream_for_word(
    stream: np.ndarray,
    templates: list[np.ndarray],
    fps: float,
    scales: tuple[float, ...] = (0.8, 1.0, 1.25),
    hop_seconds: float = 0.20,
    min_similarity: float = 0.60,
    metric: str = "l2",
) -> list[dict]:
    """Slide over a continuous stream and score one word's templates."""
    stream = np.asarray(stream, dtype=np.float32)
    if stream.ndim != 2:
        stream = stream.reshape(-1, SIGN_DIM)
    if len(stream) < 8 or not templates:
        return []

    hop = max(1, int(round(hop_seconds * fps)))
    best_by_window: dict[tuple[int, int], dict] = {}

    for template in templates:
        base_len = len(template)
        for scale in scales:
            win = int(round(base_len * scale))
            win = max(8, min(win, len(stream)))
            for start in range(0, len(stream) - win + 1, hop):
                window = stream[start : start + win]
                score = dtw_similarity(window, template, metric=metric)
                if score < min_similarity:
                    continue
                key = (start, start + win)
                prev = best_by_window.get(key)
                if prev is None or score > prev["score"]:
                    best_by_window[key] = {
                        "start_frame": start,
                        "end_frame": start + win,
                        "start_seconds": start / fps,
                        "end_seconds": (start + win) / fps,
                        "score": float(score),
                        "scale": float(scale),
                        "metric": metric,
                    }
    return list(best_by_window.values())


def pick_non_overlapping(
    candidates: list[dict],
    overlap_ratio: float = 0.35,
) -> list[dict]:
    """Greedy keep high-score detections with limited temporal overlap."""
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


def recognize_stream(
    stream: np.ndarray,
    word_templates: dict[str, list[np.ndarray]],
    fps: float = 30.0,
    min_similarity: float = 0.60,
    hop_seconds: float = 0.20,
    metric: str = "l2",
) -> dict:
    """Detect multiple words in one continuous keypoint stream."""
    stream = normalize_sequence(np.asarray(stream, dtype=np.float32))
    all_candidates: list[dict] = []
    for label, templates in word_templates.items():
        for hit in scan_stream_for_word(
            stream,
            templates,
            fps=fps,
            hop_seconds=hop_seconds,
            min_similarity=min_similarity,
            metric=metric,
        ):
            hit = dict(hit)
            hit["word"] = label
            all_candidates.append(hit)

    selected = pick_non_overlapping(all_candidates)
    words_in_time = [item["word"] for item in selected]
    unique_words = list(dict.fromkeys(words_in_time))
    return {
        "detections": selected,
        "words_in_time": words_in_time,
        "unique_words": unique_words,
        "raw_candidate_count": len(all_candidates),
        "metric": metric,
    }
