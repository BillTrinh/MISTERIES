"""Collect fingerspelling training data.

Usage:
    python src/collect_data.py A

Controls (in the webcam window):
    c   toggle recording (captures every frame where a hand is detected)
    u   undo last 10 samples
    q   save and quit

Samples are appended to data/fingerspelling/<LABEL>.npy  (shape N x 63).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from landmarks import HandTracker, FEATURE_DIM

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fingerspelling"


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/collect_data.py <LABEL>   e.g. A")
        sys.exit(1)

    label = sys.argv[1].upper()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{label}.npy"

    samples = list(np.load(out_path)) if out_path.exists() else []
    print(f"Label '{label}': starting with {len(samples)} existing samples.")

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    recording = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)             # mirror for natural feel
        feat, hand = tracker.process(frame)

        if hand is not None:
            HandTracker.draw(frame, hand)
        if recording and feat is not None:
            samples.append(feat)

        color = (0, 0, 255) if recording else (0, 200, 0)
        status = "REC" if recording else "paused"
        cv2.putText(frame, f"[{label}] {status}  n={len(samples)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "c=rec  u=undo10  q=save+quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            recording = not recording
        elif key == ord("u"):
            samples = samples[:-10]
            print(f"undo -> {len(samples)} samples")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    if samples:
        arr = np.array(samples, dtype=np.float32)
        assert arr.shape[1] == FEATURE_DIM
        np.save(out_path, arr)
        print(f"Saved {len(arr)} samples -> {out_path}")
    else:
        print("No samples captured; nothing saved.")


if __name__ == "__main__":
    main()
