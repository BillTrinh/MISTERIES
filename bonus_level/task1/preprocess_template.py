"""Preprocess a dance video into a timestamped YOLO pose cache.

Example:
    python preprocess_template.py dance.mp4 --output-dir template_data
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Keep the same macOS/OpenMP setup as danceapp.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

torch.set_num_threads(1)

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a timestamped pose cache from a template dance video."
    )
    parser.add_argument("video", type=Path, help="Path to the template video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "template_data",
        help="Parent output directory (default: template_data)",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-pose.pt",
        help="Ultralytics pose weights (default: yolov8s-pose.pt)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=15.0,
        help="Pose sampling rate (default: 15 FPS)",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Use one tracked person for the whole video (legacy mode)",
    )
    parser.add_argument(
        "--single-track",
        action="store_true",
        help="Pick one global dancer for the entire video (legacy mode)",
    )
    return parser.parse_args()


def _track_quality(records, sample_count):
    """Rank a template track by duration, visibility, size, and centrality."""
    if not records or sample_count <= 0:
        return 0.0, {}

    coverage = len(records) / sample_count
    visibility = float(
        np.mean([np.mean(record["confidence"] >= 0.3) for record in records])
    )
    areas = []
    center_scores = []
    for record in records:
        x1, y1, x2, y2 = record["bbox"]
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        distance = np.hypot(center_x - 0.5, center_y - 0.5)
        center_scores.append(max(0.0, 1.0 - distance / 0.71))

    median_area = float(np.median(areas))
    area_score = min(1.0, median_area / 0.25)
    center_score = float(np.mean(center_scores))
    quality = (
        0.50 * coverage
        + 0.20 * visibility
        + 0.20 * area_score
        + 0.10 * center_score
    )
    summary = {
        "quality": round(quality, 4),
        "coverage": round(coverage, 4),
        "visibility": round(visibility, 4),
        "median_frame_area": round(median_area, 4),
        "center_score": round(center_score, 4),
        "samples": len(records),
    }
    return quality, summary


def _instant_dominance(record):
    """Score one detection for dominance at a single timestamp."""
    visibility = float(np.mean(record["confidence"] >= 0.3))
    x1, y1, x2, y2 = record["bbox"]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_score = min(1.0, area / 0.25)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    distance = np.hypot(center_x - 0.5, center_y - 0.5)
    center_score = max(0.0, 1.0 - distance / 0.71)
    return 0.45 * visibility + 0.40 * area_score + 0.15 * center_score


def _build_per_sample_index(tracks):
    """Map each sample index to all track observations at that time."""
    per_sample = defaultdict(list)
    for track_id, records in tracks.items():
        for record in records:
            per_sample[record["sample_index"]].append((track_id, record))
    return per_sample


def _select_dominant_per_sample(per_sample, sample_count, hysteresis=0.08):
    """Pick the best dancer at each timestamp with short temporal hysteresis."""
    selected = []
    active_track = None

    for sample_index in range(sample_count):
        candidates = per_sample.get(sample_index, [])
        if not candidates:
            selected.append(None)
            active_track = None
            continue

        ranked = sorted(
            candidates,
            key=lambda item: _instant_dominance(item[1]),
            reverse=True,
        )
        best_track, best_record = ranked[0]
        best_score = _instant_dominance(best_record)

        if active_track is not None:
            current = next(
                (record for track_id, record in candidates if track_id == active_track),
                None,
            )
            if current is not None:
                current_score = _instant_dominance(current)
                if current_score + hysteresis >= best_score:
                    best_track = active_track
                    best_record = current

        selected.append((best_track, best_record))
        active_track = best_track

    return selected


def _segments_from_selection(timestamps, selected):
    """Summarise contiguous time ranges that use the same template track."""
    segments = []
    current_track = None
    start_time = None

    for timestamp, item in zip(timestamps, selected):
        if item is None:
            if current_track is not None:
                segments.append(
                    {
                        "track_id": int(current_track),
                        "start_seconds": round(float(start_time), 3),
                        "end_seconds": round(float(timestamp), 3),
                    }
                )
                current_track = None
                start_time = None
            continue

        track_id, _ = item
        if track_id != current_track:
            if current_track is not None:
                segments.append(
                    {
                        "track_id": int(current_track),
                        "start_seconds": round(float(start_time), 3),
                        "end_seconds": round(float(timestamp), 3),
                    }
                )
            current_track = track_id
            start_time = timestamp

    if current_track is not None and start_time is not None:
        segments.append(
            {
                "track_id": int(current_track),
                "start_seconds": round(float(start_time), 3),
                "end_seconds": round(float(timestamps[-1]), 3),
            }
        )
    return segments


def preprocess(args):
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        cap.release()
        raise RuntimeError("The video does not report a valid FPS")

    sample_fps = min(float(args.sample_fps), source_fps)
    model = YOLO(args.model)
    tracks = defaultdict(list)
    sample_timestamps = []
    sample_frame_indices = []
    next_sample_time = 0.0
    frame_index = 0
    last_progress = -1

    print(
        f"Processing {video_path.name}: {source_fps:.2f} FPS, "
        f"{frame_count} frames, sampling at {sample_fps:.2f} FPS"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / source_fps
            if timestamp + 1e-9 >= next_sample_time:
                sample_index = len(sample_timestamps)
                sample_timestamps.append(timestamp)
                sample_frame_indices.append(frame_index)

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                result = results[0]
                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                    and result.keypoints is not None
                ):
                    ids = result.boxes.id.int().cpu().numpy()
                    boxes = result.boxes.xyxyn.cpu().numpy()
                    keypoints = result.keypoints.xyn.cpu().numpy()
                    keypoint_conf = result.keypoints.conf
                    if keypoint_conf is None:
                        confidences = np.ones(keypoints.shape[:2], dtype=np.float32)
                    else:
                        confidences = keypoint_conf.cpu().numpy()

                    for person_index, track_id in enumerate(ids):
                        tracks[int(track_id)].append(
                            {
                                "sample_index": sample_index,
                                "keypoints": keypoints[person_index].astype(np.float32),
                                "confidence": confidences[person_index].astype(np.float32),
                                "bbox": boxes[person_index].astype(np.float32),
                            }
                        )

                next_sample_time += 1.0 / sample_fps

            frame_index += 1
            if frame_count > 0:
                progress = min(100, int(frame_index * 100 / frame_count))
                progress_bucket = progress // 10
                if progress_bucket != last_progress:
                    print(f"Progress: {progress_bucket * 10}%")
                    last_progress = progress_bucket
    finally:
        cap.release()

    if not sample_timestamps:
        raise RuntimeError("No frames were sampled from the video")
    if not tracks:
        raise RuntimeError("No consistently tracked people were found in the video")

    track_summaries = {}
    ranked_tracks = []
    for track_id, records in tracks.items():
        quality, summary = _track_quality(records, len(sample_timestamps))
        track_summaries[str(track_id)] = summary
        ranked_tracks.append((quality, track_id))
    ranked_tracks.sort(reverse=True)

    sample_count = len(sample_timestamps)
    keypoints = np.full((sample_count, 17, 2), np.nan, dtype=np.float32)
    confidences = np.zeros((sample_count, 17), dtype=np.float32)
    bboxes = np.full((sample_count, 4), np.nan, dtype=np.float32)
    valid = np.zeros(sample_count, dtype=bool)
    selected_track_ids = np.full(sample_count, -1, dtype=np.int32)

    if args.track_id is not None:
        if args.track_id not in tracks:
            available = ", ".join(str(track_id) for _, track_id in ranked_tracks)
            raise ValueError(
                f"Track ID {args.track_id} was not found. Available IDs: {available}"
            )
        selection_method = "manual_single_track"
        for record in tracks[args.track_id]:
            index = record["sample_index"]
            keypoints[index] = record["keypoints"]
            confidences[index] = record["confidence"]
            bboxes[index] = record["bbox"]
            valid[index] = True
            selected_track_ids[index] = args.track_id
        segments = _segments_from_selection(
            sample_timestamps,
            [(args.track_id, None) if valid[i] else None for i in range(sample_count)],
        )
    elif args.single_track:
        selected_track_id = ranked_tracks[0][1]
        selection_method = "automatic_single_track"
        for record in tracks[selected_track_id]:
            index = record["sample_index"]
            keypoints[index] = record["keypoints"]
            confidences[index] = record["confidence"]
            bboxes[index] = record["bbox"]
            valid[index] = True
            selected_track_ids[index] = selected_track_id
        segments = _segments_from_selection(
            sample_timestamps,
            [(selected_track_id, None) if valid[i] else None for i in range(sample_count)],
        )
    else:
        selection_method = "per_sample_dominant"
        per_sample = _build_per_sample_index(tracks)
        dominant = _select_dominant_per_sample(per_sample, sample_count)
        segments = _segments_from_selection(sample_timestamps, dominant)
        for sample_index, item in enumerate(dominant):
            if item is None:
                continue
            track_id, record = item
            keypoints[sample_index] = record["keypoints"]
            confidences[sample_index] = record["confidence"]
            bboxes[sample_index] = record["bbox"]
            valid[sample_index] = True
            selected_track_ids[sample_index] = int(track_id)

    output_dir = args.output_dir.expanduser().resolve() / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    poses_path = output_dir / "poses.npz"
    metadata_path = output_dir / "metadata.json"

    np.savez_compressed(
        poses_path,
        timestamps=np.asarray(sample_timestamps, dtype=np.float32),
        frame_indices=np.asarray(sample_frame_indices, dtype=np.int32),
        keypoints=keypoints,
        confidences=confidences,
        bboxes=bboxes,
        valid=valid,
        selected_track_ids=selected_track_ids,
    )

    metadata = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(video_path),
        "source_file_size": video_path.stat().st_size,
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "duration_seconds": frame_count / source_fps if frame_count > 0 else None,
        "width": width,
        "height": height,
        "model": args.model,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "sample_fps": sample_fps,
        "sample_count": sample_count,
        "selection_method": selection_method,
        "selected_valid_samples": int(np.sum(valid)),
        "segment_count": len(segments),
        "segments": segments,
        "tracks": track_summaries,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Selection mode: {selection_method}")
    print(f"Valid poses: {int(np.sum(valid))}/{sample_count}")
    print(f"Template segments: {len(segments)}")
    for index, segment in enumerate(segments[:8], start=1):
        print(
            f"  Segment {index}: track {segment['track_id']} "
            f"{segment['start_seconds']:.2f}s -> {segment['end_seconds']:.2f}s"
        )
    if len(segments) > 8:
        print(f"  ... and {len(segments) - 8} more")
    print(f"Pose cache: {poses_path}")
    print(f"Metadata:   {metadata_path}")


def main():
    try:
        preprocess(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
