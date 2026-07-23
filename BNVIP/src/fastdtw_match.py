"""FastDTW multi-sign matching (approximate DTW, separate from dtw_match.py).

Classic multiresolution FastDTW (Salvador & Chan): coarsen sequences,
recursively align, project the path, refine only near that corridor.
Local frame cost defaults to L2 (same family as app_dtw_continuous.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dtw_match import load_word_templates, pick_non_overlapping
from sequence_utils import SIGN_DIM, normalize_sequence

# Re-export for apps.
__all__ = [
    "load_word_templates",
    "recognize_stream",
    "fastdtw_distance",
    "fastdtw_similarity",
]


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :].astype(np.float64) - b[None, :, :].astype(np.float64)
    return np.sqrt(np.mean(diff * diff, axis=2) + 1e-12)


def _dtw_windowed(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    window: set[tuple[int, int]] | None,
) -> tuple[float, list[tuple[int, int]]]:
    """Exact DTW restricted to an optional set of (i, j) cells (0-based)."""
    a = np.asarray(seq_a, dtype=np.float32).reshape(-1, SIGN_DIM)
    b = np.asarray(seq_b, dtype=np.float32).reshape(-1, SIGN_DIM)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []

    dist = _pairwise_l2(a, b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    pred = np.full((n + 1, m + 1, 2), -1, dtype=np.int32)

    if window is None:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best = cost[i - 1, j]
                src = (i - 1, j)
                if cost[i, j - 1] < best:
                    best = cost[i, j - 1]
                    src = (i, j - 1)
                if cost[i - 1, j - 1] < best:
                    best = cost[i - 1, j - 1]
                    src = (i - 1, j - 1)
                cost[i, j] = best + dist[i - 1, j - 1]
                pred[i, j] = src
    else:
        ordered = sorted(window, key=lambda ij: ij[0] + ij[1])
        for i, j in ordered:
            ii, jj = i + 1, j + 1
            best = cost[ii - 1, jj]
            src = (ii - 1, jj)
            if cost[ii, jj - 1] < best:
                best = cost[ii, jj - 1]
                src = (ii, jj - 1)
            if cost[ii - 1, jj - 1] < best:
                best = cost[ii - 1, jj - 1]
                src = (ii - 1, jj - 1)
            if np.isfinite(best):
                cost[ii, jj] = best + dist[i, j]
                pred[ii, jj] = src

    if not np.isfinite(cost[n, m]):
        band = max(1, int(0.25 * max(n, m)))
        cost[:, :] = np.inf
        cost[0, 0] = 0.0
        pred[:, :, :] = -1
        for i in range(1, n + 1):
            j0 = max(1, i - band)
            j1 = min(m, i + band)
            for j in range(j0, j1 + 1):
                best = cost[i - 1, j]
                src = (i - 1, j)
                if cost[i, j - 1] < best:
                    best = cost[i, j - 1]
                    src = (i, j - 1)
                if cost[i - 1, j - 1] < best:
                    best = cost[i - 1, j - 1]
                    src = (i - 1, j - 1)
                cost[i, j] = best + dist[i - 1, j - 1]
                pred[i, j] = src

    path: list[tuple[int, int]] = []
    i, j = n, m
    if not np.isfinite(cost[n, m]):
        return float("inf"), path
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        pi, pj = int(pred[i, j, 0]), int(pred[i, j, 1])
        if pi < 0:
            break
        i, j = pi, pj
    path.reverse()
    return float(cost[n, m]) / (n + m), path


def _reduce(seq: np.ndarray) -> np.ndarray:
    """Average adjacent frames (length ~ ceil(n/2))."""
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n <= 1:
        return seq.copy()
    even = seq[0 : n - (n % 2) : 2]
    odd = seq[1::2]
    out = 0.5 * (even + odd)
    if n % 2 == 1:
        out = np.concatenate([out, seq[-1:]], axis=0)
    return out


def _expand_window(
    path: list[tuple[int, int]],
    n: int,
    m: int,
    radius: int,
) -> set[tuple[int, int]]:
    """Project a coarse path onto the finer grid and thicken by radius."""
    window: set[tuple[int, int]] = set()
    for i, j in path:
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for ii in (2 * i + di, 2 * i + 1 + di):
                    for jj in (2 * j + dj, 2 * j + 1 + dj):
                        if 0 <= ii < n and 0 <= jj < m:
                            window.add((ii, jj))
    for i in range(n):
        window.add((i, 0))
        window.add((i, m - 1))
    for j in range(m):
        window.add((0, j))
        window.add((n - 1, j))
    return window


def fastdtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    radius: int = 1,
) -> float:
    """Approximate DTW distance (normalized by n+m). Self-match ~= 0."""
    dist, _ = _fastdtw(seq_a, seq_b, radius=radius)
    return dist


def _fastdtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    radius: int = 1,
) -> tuple[float, list[tuple[int, int]]]:
    a = np.asarray(seq_a, dtype=np.float32).reshape(-1, SIGN_DIM)
    b = np.asarray(seq_b, dtype=np.float32).reshape(-1, SIGN_DIM)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []

    min_size = radius + 2
    if n <= min_size or m <= min_size:
        return _dtw_windowed(a, b, window=None)

    a_c = _reduce(a)
    b_c = _reduce(b)
    _, path = _fastdtw(a_c, b_c, radius=radius)
    window = _expand_window(path, n, m, radius)
    return _dtw_windowed(a, b, window)


def fastdtw_similarity(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    temperature: float = 0.20,
    radius: int = 1,
) -> float:
    dist = fastdtw_distance(seq_a, seq_b, radius=radius)
    if not np.isfinite(dist):
        return 0.0
    return float(np.exp(-dist / max(temperature, 1e-6)))


def scan_stream_for_word(
    stream: np.ndarray,
    templates: list[np.ndarray],
    fps: float,
    scales: tuple[float, ...] = (0.8, 1.0, 1.25),
    hop_seconds: float = 0.20,
    min_similarity: float = 0.60,
    radius: int = 1,
) -> list[dict]:
    stream = np.asarray(stream, dtype=np.float32).reshape(-1, SIGN_DIM)
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
                score = fastdtw_similarity(window, template, radius=radius)
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
                    }
    return list(best_by_window.values())


def recognize_stream(
    stream: np.ndarray,
    word_templates: dict[str, list[np.ndarray]],
    fps: float = 30.0,
    min_similarity: float = 0.60,
    hop_seconds: float = 0.20,
    radius: int = 1,
) -> dict:
    """Detect multiple words with FastDTW template matching."""
    stream = normalize_sequence(np.asarray(stream, dtype=np.float32))
    all_candidates: list[dict] = []
    for label, templates in word_templates.items():
        for hit in scan_stream_for_word(
            stream,
            templates,
            fps=fps,
            hop_seconds=hop_seconds,
            min_similarity=min_similarity,
            radius=radius,
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
        "method": "fastdtw",
        "radius": radius,
    }
