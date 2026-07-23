"""Webcam continuous multi-sign recognition with DTW + cosine local cost.

Usage:
    python src/app_dtw_cosine.py

Same as app_dtw_continuous.py, but DTW frame cost is cosine distance
(1 - cos) instead of L2. Leaves the L2 app unchanged.

Controls:
    space   start / stop recording a continuous signing segment
    r       send unique words to the LLM (clean + reply)
    c       clear current words / last result
    n       new conversation
    q       quit
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from chat import Chat
from dtw_match import load_word_templates, recognize_stream
from landmarks import SignTracker
from refine_prompt import ollama_available, refine

ROOT = Path(__file__).resolve().parent.parent
SIGNS_DIR = ROOT / "data" / "signs"
MIN_SCORE = 0.55
HOP_SECONDS = 0.20
MIN_FRAMES = 20
MAX_FRAMES = 900          # ~30s at 30 FPS safety cap
METRIC = "cosine"


def draw_ui(frame, recording, n_frames, status, words, user, reply, llm_up):
    h, w = frame.shape[:2]
    if recording:
        cv2.rectangle(frame, (0, 0), (w, 10), (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"REC {n_frames} frames  (space=stop)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            status[:70],
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    dot = (0, 200, 0) if llm_up else (0, 0, 255)
    cv2.putText(frame, "LLM", (w - 95, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.circle(frame, (w - 25, 20), 8, dot, -1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        "words: " + (" ".join(words) or "_"),
        (10, h - 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "you:   " + (user or "_"),
        (10, h - 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "agent: " + (reply or "_")[:60],
        (10, h - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (120, 220, 120),
        2,
    )
    cv2.putText(
        frame,
        "DTW-cos | space=start/stop  r=send  c=clear  n=new  q=quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )


def main():
    print("Loading DTW-cosine word templates ...")
    templates = load_word_templates(SIGNS_DIR, max_templates_per_word=4)
    print("Words:", ", ".join(sorted(templates)))

    tracker = SignTracker()
    chat = Chat()
    llm_up = ollama_available()
    print("Ollama:", "up" if llm_up else "down (fallback replies)")

    words, user, reply = [], "", ""
    status = "space = start continuous signing"
    recording = False
    buf = []
    rec_started = None

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feat, hand_list, _has_hand = tracker.process(frame)
        SignTracker.draw(frame, hand_list)

        if recording:
            buf.append(feat.astype(np.float32))
            if len(buf) >= MAX_FRAMES:
                status = "Hit max length; press space to stop & recognize"
                # keep recording until user stops, but don't grow forever
                buf = buf[-MAX_FRAMES:]

        draw_ui(frame, recording, len(buf), status, words, user, reply, llm_up)
        cv2.imshow("BNVIP - continuous DTW-cosine signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord(" "):
            if not recording:
                recording = True
                buf = []
                rec_started = time.monotonic()
                status = "Recording... press space to stop"
                print("Recording started")
            else:
                recording = False
                elapsed = max(0.001, time.monotonic() - (rec_started or time.monotonic()))
                fps = len(buf) / elapsed if buf else 30.0
                fps = float(np.clip(fps, 10.0, 60.0))
                print(f"Recording stopped: {len(buf)} frames, ~{fps:.1f} FPS")

                if len(buf) < MIN_FRAMES:
                    status = f"Too short ({len(buf)} frames); try again"
                    words = []
                else:
                    status = "Running DTW-cosine..."
                    draw_ui(frame, False, len(buf), status, words, user, reply, llm_up)
                    cv2.imshow("BNVIP - continuous DTW-cosine signs", frame)
                    cv2.waitKey(1)

                    stream = np.stack(buf, axis=0)
                    result = recognize_stream(
                        stream,
                        templates,
                        fps=fps,
                        min_similarity=MIN_SCORE,
                        hop_seconds=HOP_SECONDS,
                        metric="cosine",
                    )
                    words = result["unique_words"]
                    if words:
                        status = "Detected: " + " ".join(words)
                        print("Detections:")
                        for hit in result["detections"]:
                            print(
                                f"  {hit['word']:<8} "
                                f"{hit['start_seconds']:.2f}s -> {hit['end_seconds']:.2f}s "
                                f"score={hit['score']:.3f}"
                            )
                        print("Unique words:", " ".join(words))
                    else:
                        status = "No signs detected; try clearer signing / lower threshold"
                        print(status)
                buf = []
                rec_started = None

        elif key == ord("r") and words:
            user = refine(" ".join(words))
            reply = chat.send(user)
            print(f"you: {user!r}\nagent: {reply!r}\n")
            words = []
            status = "Sent. space = record next segment"

        elif key == ord("c"):
            words, user, reply = [], "", ""
            status = "Cleared. space = start continuous signing"

        elif key == ord("n"):
            chat.reset()
            words, user, reply = [], "", ""
            status = "New conversation. space = start continuous signing"

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
