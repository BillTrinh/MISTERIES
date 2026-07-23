"""Recognize multiple word-signs from one continuous video with DTW.

Usage:
    python src/recognize_video_dtw.py /path/to/video.mp4
    python src/recognize_video_dtw.py /path/to/video.mp4 --llm
    python src/recognize_video_dtw.py /path/to/video.mp4 --min-score 0.58

This scans the whole video once, finds multiple signs by template DTW
matching (handles different signing speeds), and outputs a word list.
Word order is not critical: pass --llm to let Ollama rewrite the bag of
words into an English sentence.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dtw_match import load_word_templates, recognize_stream
from landmarks import SignTracker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIGNS = ROOT / "data" / "signs"


def extract_stream(video_path: Path, sample_fps: float | None = None):
    """Extract per-frame 126-d hand features from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    use_fps = source_fps if sample_fps is None else min(sample_fps, source_fps)
    frame_stride = max(1, int(round(source_fps / use_fps)))

    tracker = SignTracker()
    feats = []
    kept_indices = []
    frame_index = 0
    next_keep = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index >= next_keep:
                frame = cv2.flip(frame, 1)
                feat, _, has_hand = tracker.process(frame)
                # Keep zero vectors too so timing stays aligned; DTW can ignore
                # missing-hand dimensions via frame_distance masking.
                feats.append(feat.astype(np.float32))
                kept_indices.append(frame_index)
                next_keep += frame_stride
            frame_index += 1
            if frame_count > 0 and frame_index % max(1, frame_count // 10) == 0:
                pct = int(frame_index * 100 / frame_count)
                print(f"Extracting landmarks: {pct}%", flush=True)
    finally:
        cap.release()
        tracker.close()

    if not feats:
        raise RuntimeError("No frames were read from the video")

    stream = np.stack(feats, axis=0)
    effective_fps = use_fps
    return stream, effective_fps, source_fps, len(feats)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DTW multi-sign recognition from one video"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--signs-dir",
        type=Path,
        default=DEFAULT_SIGNS,
        help="Directory of word templates (*.pkl)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.60,
        help="Minimum DTW similarity to keep a detection",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.20,
        help="Sliding-window hop size in seconds",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=20.0,
        help="Landmark sampling rate from the video (default 20)",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=4,
        help="Max template clips used per word",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Rewrite unique words into an English sentence with Ollama",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading templates from {args.signs_dir} ...")
    templates = load_word_templates(
        args.signs_dir, max_templates_per_word=args.max_templates
    )
    print(f"Words: {', '.join(sorted(templates))}")

    print(f"Reading video: {video_path.name}")
    t0 = time.time()
    stream, fps, source_fps, n_frames = extract_stream(
        video_path, sample_fps=args.sample_fps
    )
    print(
        f"Stream ready: {n_frames} frames @ ~{fps:.1f} FPS "
        f"(source {source_fps:.1f} FPS), extract {time.time() - t0:.1f}s"
    )

    # Drop long all-zero gaps at the ends, but keep internal timing.
    active = ~np.all(stream == 0, axis=1)
    if np.any(active):
        first = int(np.argmax(active))
        last = int(len(active) - 1 - np.argmax(active[::-1]))
        stream = stream[first : last + 1]
        time_offset = first / fps
    else:
        time_offset = 0.0
        print("Warning: no hands detected in the video", file=sys.stderr)

    print("Running DTW multi-sign scan ...")
    t1 = time.time()
    result = recognize_stream(
        stream,
        templates,
        fps=fps,
        min_similarity=args.min_score,
        hop_seconds=args.hop_seconds,
    )
    print(f"DTW done in {time.time() - t1:.1f}s")
    print(f"Raw candidates: {result['raw_candidate_count']}")
    print(f"Kept detections: {len(result['detections'])}")

    if not result["detections"]:
        print("No signs detected. Try lowering --min-score.")
        raise SystemExit(2)

    print("\nDetections (by time):")
    for hit in result["detections"]:
        start = hit["start_seconds"] + time_offset
        end = hit["end_seconds"] + time_offset
        print(
            f"  {hit['word']:<8}  {start:6.2f}s -> {end:6.2f}s  "
            f"score={hit['score']:.3f}  speed={hit['scale']:.2f}x"
        )

    unique = result["unique_words"]
    print("\nUnique words:", " ".join(unique))
    print("Words bag for LLM:", ", ".join(unique))

    if args.llm:
        from refine_prompt import refine, ollama_available

        if not ollama_available():
            print("Ollama unavailable; skipped sentence rewrite.")
        else:
            raw = " ".join(unique)
            sentence = refine(raw)
            print("LLM sentence:", sentence)


if __name__ == "__main__":
    main()
