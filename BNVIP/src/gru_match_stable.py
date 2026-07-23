"""More stable GRU sliding-window multi-sign recognition.

Improvements over gru_match.py (left unchanged):
  - higher default confidence threshold
  - fewer window scales by default
  - confidence peak picking (rise then fall)
  - merge consecutive same-word hits
  - mild motion gate to skip near-static transition frames
"""
from __future__ import annotations

import numpy as np
import torch

from sequence_utils import SIGN_DIM, preprocess
from sign_model import SignRecognizer


def _predict_batch(
    recognizer: SignRecognizer, windows: list[np.ndarray]
) -> list[tuple[str, float, np.ndarray]]:
    if not windows:
        return []
    xs = [preprocess(w, recognizer.T) for w in windows]
    batch = torch.tensor(np.stack(xs, axis=0), device=recognizer.device)
    with torch.no_grad():
        probs = torch.softmax(recognizer.model(batch), dim=1).cpu().numpy()
    out = []
    for row in probs:
        i = int(np.argmax(row))
        out.append((str(recognizer.labels[i]).upper(), float(row[i]), row))
    return out


def _motion_score(window: np.ndarray) -> float:
    """Mean frame-to-frame keypoint motion; low means nearly static."""
    w = np.asarray(window, dtype=np.float32).reshape(-1, SIGN_DIM)
    if len(w) < 2:
        return 0.0
    deltas = np.linalg.norm(w[1:] - w[:-1], axis=1)
    return float(np.mean(deltas))


def _merge_same_word(detections: list[dict]) -> list[dict]:
    """Collapse consecutive / heavily overlapping same-word detections."""
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
            prev["end_frame"] = max(prev["end_frame"], hit["end_frame"])
            prev["end_seconds"] = max(prev["end_seconds"], hit["end_seconds"])
            prev["start_frame"] = min(prev["start_frame"], hit["start_frame"])
            prev["start_seconds"] = min(prev["start_seconds"], hit["start_seconds"])
            if hit["score"] > prev["score"]:
                prev["score"] = hit["score"]
                prev["scale"] = hit["scale"]
        else:
            merged.append(dict(hit))
    return merged


def _pick_peaks_per_word(
    timeline: list[dict],
    min_confidence: float,
    peak_margin: float = 0.03,
) -> list[dict]:
    """Keep local confidence peaks for each word label along time."""
    by_word: dict[str, list[dict]] = {}
    for hit in timeline:
        by_word.setdefault(hit["word"], []).append(hit)

    peaks: list[dict] = []
    for word, items in by_word.items():
        items = sorted(items, key=lambda item: item["center_frame"])
        if not items:
            continue
        for idx, hit in enumerate(items):
            if hit["score"] < min_confidence:
                continue
            left = items[idx - 1]["score"] if idx > 0 else -1.0
            right = items[idx + 1]["score"] if idx + 1 < len(items) else -1.0
            # Peak: not lower than neighbors (with small margin).
            if hit["score"] + 1e-6 >= left - peak_margin and hit["score"] + 1e-6 >= right - peak_margin:
                # Prefer strict local maxima when possible.
                if hit["score"] >= left and hit["score"] >= right:
                    peaks.append(hit)
                elif hit["score"] >= min_confidence + 0.10:
                    peaks.append(hit)

        # Fallback: if this word never formed a peak but has a strong max, keep it.
        if not any(p["word"] == word for p in peaks):
            best = max(items, key=lambda item: item["score"])
            if best["score"] >= min_confidence + 0.05:
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


def recognize_stream(
    stream: np.ndarray,
    recognizer: SignRecognizer | None = None,
    fps: float = 30.0,
    base_len: int = 40,
    scales: tuple[float, ...] = (1.0, 1.15),
    hop_seconds: float = 0.20,
    min_confidence: float = 0.72,
    min_motion: float = 0.012,
    batch_size: int = 64,
) -> dict:
    """Stable multi-sign decode from a continuous keypoint stream."""
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
            window = stream[start : start + win]
            if _motion_score(window) < min_motion:
                continue
            jobs.append((start, start + win, float(scale), window))

    timeline: list[dict] = []
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        preds = _predict_batch(recognizer, [item[3] for item in chunk])
        for (start, end, scale, _window), (label, conf, _probs) in zip(chunk, preds):
            if conf < min_confidence - 0.08:
                # Keep slightly weaker points only for peak context.
                continue
            timeline.append(
                {
                    "word": label,
                    "start_frame": start,
                    "end_frame": end,
                    "center_frame": (start + end) // 2,
                    "start_seconds": start / fps,
                    "end_seconds": end / fps,
                    "score": float(conf),
                    "scale": scale,
                }
            )

    peaks = _pick_peaks_per_word(timeline, min_confidence=min_confidence)
    selected = pick_non_overlapping(peaks, overlap_ratio=0.30)
    selected = _merge_same_word(selected)

    # Drop residual weak detections after merge.
    selected = [hit for hit in selected if hit["score"] >= min_confidence]

    words_in_time = [item["word"] for item in selected]
    unique_words = list(dict.fromkeys(words_in_time))
    return {
        "detections": selected,
        "words_in_time": words_in_time,
        "unique_words": unique_words,
        "raw_candidate_count": len(timeline),
    }
