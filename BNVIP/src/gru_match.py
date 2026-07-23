"""GRU sliding-window multi-sign recognition.

Uses the existing SignRecognizer / sign_gru.pt. Does not modify DTW or fast-match
files. Windows of several durations are classified; high-confidence hits are
kept with non-overlapping selection.
"""
from __future__ import annotations

import numpy as np
import torch

from sequence_utils import SIGN_DIM, preprocess
from sign_model import SignRecognizer


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


def _predict_batch(recognizer: SignRecognizer, windows: list[np.ndarray]) -> list[tuple[str, float]]:
    """Classify many windows in one forward pass when possible."""
    if not windows:
        return []
    xs = [preprocess(w, recognizer.T) for w in windows]
    batch = torch.tensor(np.stack(xs, axis=0), device=recognizer.device)
    with torch.no_grad():
        probs = torch.softmax(recognizer.model(batch), dim=1).cpu().numpy()
    out = []
    for row in probs:
        i = int(np.argmax(row))
        out.append((recognizer.labels[i], float(row[i])))
    return out


def recognize_stream(
    stream: np.ndarray,
    recognizer: SignRecognizer | None = None,
    fps: float = 30.0,
    base_len: int = 40,
    scales: tuple[float, ...] = (0.8, 1.0, 1.25),
    hop_seconds: float = 0.15,
    min_confidence: float = 0.55,
    batch_size: int = 64,
) -> dict:
    """Slide GRU windows over a continuous keypoint stream."""
    if recognizer is None:
        recognizer = SignRecognizer()

    stream = np.asarray(stream, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(stream) < 12:
        return {
            "detections": [],
            "words_in_time": [],
            "unique_words": [],
            "raw_candidate_count": 0,
        }

    hop = max(1, int(round(hop_seconds * fps)))
    jobs: list[tuple[int, int, float, np.ndarray]] = []
    for scale in scales:
        win = int(round(base_len * scale))
        win = max(16, min(win, len(stream)))
        for start in range(0, len(stream) - win + 1, hop):
            jobs.append((start, start + win, float(scale), stream[start : start + win]))

    candidates: list[dict] = []
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        preds = _predict_batch(recognizer, [item[3] for item in chunk])
        for (start, end, scale, _window), (label, conf) in zip(chunk, preds):
            if conf < min_confidence:
                continue
            candidates.append(
                {
                    "word": str(label).upper(),
                    "start_frame": start,
                    "end_frame": end,
                    "start_seconds": start / fps,
                    "end_seconds": end / fps,
                    "score": float(conf),
                    "scale": scale,
                }
            )

    # Keep best score per exact window, then suppress overlaps.
    best_by_window: dict[tuple[int, int], dict] = {}
    for hit in candidates:
        key = (hit["start_frame"], hit["end_frame"])
        prev = best_by_window.get(key)
        if prev is None or hit["score"] > prev["score"]:
            best_by_window[key] = hit

    selected = pick_non_overlapping(list(best_by_window.values()))
    words_in_time = [item["word"] for item in selected]
    unique_words = list(dict.fromkeys(words_in_time))
    return {
        "detections": selected,
        "words_in_time": words_in_time,
        "unique_words": unique_words,
        "raw_candidate_count": len(candidates),
    }
