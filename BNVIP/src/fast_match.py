"""Fast multi-sign matching without DTW.

Method:
  normalize + resample each window/template to a fixed length, then compare
  with cosine similarity. Multiple window scales handle signing speed.

This is intentionally separate from dtw_match.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sequence_utils import SIGN_DIM, T_FIXED, normalize_sequence, preprocess


def sequence_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Cosine similarity after normalize+resample to T_FIXED."""
    a = preprocess(seq_a, T_FIXED).reshape(-1)
    b = preprocess(seq_b, T_FIXED).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def load_word_templates(
    signs_dir: Path,
    max_templates_per_word: int = 6,
) -> dict[str, list[np.ndarray]]:
    """Load raw clips; preprocess happens at compare time / cache below."""
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
        # Cache already-preprocessed fixed-length templates for speed.
        templates[path.stem.upper()] = [
            preprocess(np.asarray(clip, dtype=np.float32), T_FIXED) for clip in chosen
        ]
    if not templates:
        raise FileNotFoundError(f"No sign templates found in {signs_dir}")
    return templates


def _window_similarity(window: np.ndarray, template_fixed: np.ndarray) -> float:
    a = preprocess(window, T_FIXED).reshape(-1)
    b = np.asarray(template_fixed, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def scan_stream_for_word(
    stream: np.ndarray,
    templates_fixed: list[np.ndarray],
    fps: float,
    base_len: int = 40,
    scales: tuple[float, ...] = (0.75, 1.0, 1.3),
    hop_seconds: float = 0.15,
    min_similarity: float = 0.72,
) -> list[dict]:
    """Slide windows and score against one word's fixed-length templates."""
    stream = np.asarray(stream, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(stream) < 8 or not templates_fixed:
        return []

    hop = max(1, int(round(hop_seconds * fps)))
    best_by_window: dict[tuple[int, int], dict] = {}

    for scale in scales:
        win = int(round(base_len * scale))
        win = max(12, min(win, len(stream)))
        for start in range(0, len(stream) - win + 1, hop):
            window = stream[start : start + win]
            score = max(_window_similarity(window, tmpl) for tmpl in templates_fixed)
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


def pick_non_overlapping(
    candidates: list[dict],
    overlap_ratio: float = 0.35,
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


def recognize_stream(
    stream: np.ndarray,
    word_templates: dict[str, list[np.ndarray]],
    fps: float = 30.0,
    min_similarity: float = 0.72,
    hop_seconds: float = 0.15,
    base_len: int = 40,
) -> dict:
    """Detect multiple words with resample + cosine similarity."""
    stream = normalize_sequence(np.asarray(stream, dtype=np.float32))
    all_candidates: list[dict] = []
    for label, templates in word_templates.items():
        for hit in scan_stream_for_word(
            stream,
            templates,
            fps=fps,
            base_len=base_len,
            hop_seconds=hop_seconds,
            min_similarity=min_similarity,
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
    }
