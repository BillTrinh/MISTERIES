"""Webcam continuous multi-sign recognition with a more stable GRU decoder.

Usage:
    python src/app_gru_stable.py

Leaves app_gru_continuous.py / gru_match.py unchanged.

Controls:
    space   start / stop recording
    r       send unique words to the LLM
    c       clear
    n       new conversation
    q       quit
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from chat import Chat
from gru_match_stable import recognize_stream
from landmarks import SignTracker
from refine_prompt import ollama_available, refine
from sign_model import SignRecognizer

MIN_CONF = 0.72
HOP_SECONDS = 0.20
MIN_FRAMES = 20
MAX_FRAMES = 900


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
        "GRU stable | space=start/stop  r=send  c=clear  n=new  q=quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )


def main():
    print("Loading GRU model (stable decoder) ...")
    recognizer = SignRecognizer()
    print("Labels:", ", ".join(recognizer.labels))

    tracker = SignTracker()
    chat = Chat()
    llm_up = ollama_available()
    print("Ollama:", "up" if llm_up else "down (fallback replies)")

    words, user, reply = [], "", ""
    status = "GRU stable: space = start continuous signing"
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
                buf = buf[-MAX_FRAMES:]

        draw_ui(frame, recording, len(buf), status, words, user, reply, llm_up)
        cv2.imshow("BNVIP - continuous GRU stable", frame)

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
                    status = "Running stable GRU decode..."
                    draw_ui(frame, False, len(buf), status, words, user, reply, llm_up)
                    cv2.imshow("BNVIP - continuous GRU stable", frame)
                    cv2.waitKey(1)

                    stream = np.stack(buf, axis=0)
                    result = recognize_stream(
                        stream,
                        recognizer=recognizer,
                        fps=fps,
                        min_confidence=MIN_CONF,
                        hop_seconds=HOP_SECONDS,
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
                        status = "No signs detected; try clearer signing"
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
