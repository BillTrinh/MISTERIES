"""Collect dynamic word-sign clips.

Usage:
    python src/collect_sequences.py HELLO

Controls:
    space   record ONE clip (auto-captures CLIP_LEN frames while you sign)
    u       undo last clip
    q       save and quit

Perform the sign during the ~1.3s recording window. Aim for 30-50 clips per
sign, varying speed and position a little. Clips are saved to
data/signs/<LABEL>.pkl as a list of (frames, 126) arrays.
"""
import sys
import pickle
from pathlib import Path

import cv2

from landmarks import SignTracker

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "signs"
CLIP_LEN = 40          # frames captured per clip (~1.3s at 30 fps)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/collect_sequences.py <LABEL>   e.g. HELLO")
        sys.exit(1)

    label = sys.argv[1].upper()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{label}.pkl"

    clips = []
    if out_path.exists():
        with open(out_path, "rb") as f:
            clips = pickle.load(f)
    print(f"Label '{label}': starting with {len(clips)} clips.")

    tracker = SignTracker()
    cap = cv2.VideoCapture(0)
    recording = False
    current = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feat, hand_list, has_hand = tracker.process(frame)
        SignTracker.draw(frame, hand_list)

        if recording:
            current.append(feat)
            # progress bar
            w = frame.shape[1]
            pct = len(current) / CLIP_LEN
            cv2.rectangle(frame, (0, 0), (int(w * pct), 8), (0, 0, 255), -1)
            if len(current) >= CLIP_LEN:
                import numpy as np
                clips.append(np.array(current, dtype="float32"))
                current = []
                recording = False

        color = (0, 0, 255) if recording else (0, 200, 0)
        status = f"REC {len(current)}/{CLIP_LEN}" if recording else "ready"
        cv2.putText(frame, f"[{label}] {status}  clips={len(clips)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "space=record  u=undo  q=save+quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("collect signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not recording:
            recording = True
            current = []
        elif key == ord("u") and clips:
            clips.pop()
            print(f"undo -> {len(clips)} clips")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    if clips:
        with open(out_path, "wb") as f:
            pickle.dump(clips, f)
        print(f"Saved {len(clips)} clips -> {out_path}")
    else:
        print("No clips captured; nothing saved.")


if __name__ == "__main__":
    main()
