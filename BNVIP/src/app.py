"""Real-time ASL fingerspelling -> text -> LLM-refined prompt.

Usage:
    python src/app.py

Controls:
    r      refine the current text into a clean prompt (via Ollama)
    b      backspace
    c      clear text
    q      quit

A letter is typed by holding the sign steady; a pause (hand out of frame)
inserts a space.
"""
import cv2

from landmarks import HandTracker
from classifier import FingerspellClassifier
from text_buffer import TextBuffer
from refine_prompt import refine, ollama_available


def draw_panel(frame, letter, conf, raw, prompt, llm_up):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if letter:
        cv2.putText(frame, f"{letter}  ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    dot = (0, 200, 0) if llm_up else (0, 0, 255)
    cv2.putText(frame, "LLM", (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    cv2.circle(frame, (w - 25, 20), 8, dot, -1)

    cv2.putText(frame, "raw:    " + (raw or "_"), (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "prompt: " + (prompt or "_"), (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "r=refine  b=back  c=clear  q=quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    tracker = HandTracker()
    clf = FingerspellClassifier()
    buf = TextBuffer()
    prompt = ""
    llm_up = ollama_available()
    print("Ollama:", "up" if llm_up else "down (refine will use fallback)")

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feat, hand = tracker.process(frame)

        letter, conf = None, 0.0
        if feat is not None:
            letter, conf = clf.predict(feat)
            HandTracker.draw(frame, hand)
        buf.update(letter, conf)

        draw_panel(frame, letter, conf, buf.text, prompt, llm_up)
        cv2.imshow("BNVIP - sign to prompt", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            prompt = refine(buf.text)
            print(f"raw: {buf.text!r}  ->  prompt: {prompt!r}")
        elif key == ord("b"):
            buf.backspace()
        elif key == ord("c"):
            buf.clear()
            prompt = ""

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
