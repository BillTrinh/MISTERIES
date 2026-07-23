"""Real-time dynamic word-sign -> conversation with a local LLM agent.

Usage:
    python src/app_dynamic.py

Flow: sign a few keyword words -> they gather in the current turn -> send the
turn -> the LLM cleans it into a sentence AND replies (with memory).

Controls:
    space   record ONE sign (~1.3s) and add the recognized word to the turn
    r       send the turn: clean into a sentence + get the agent's reply
    b       remove the last word
    c       clear the current turn
    n       new conversation (reset the agent's memory)
    q       quit
"""
import cv2

from landmarks import SignTracker
from sign_model import SignRecognizer
from refine_prompt import refine, ollama_available
from chat import Chat

CLIP_LEN = 40          # keep in sync with collect_sequences.py
CONF_THRESH = 0.5


def draw_ui(frame, armed, rec, prog, last_word, last_conf, words, user, reply, llm_up):
    h, w = frame.shape[:2]
    if rec:
        cv2.rectangle(frame, (0, 0), (int(w * prog), 8), (0, 0, 255), -1)
    if armed:
        cv2.putText(frame, "SIGN NOW", (w // 2 - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)

    dot = (0, 200, 0) if llm_up else (0, 0, 255)
    cv2.putText(frame, "LLM", (w - 95, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    cv2.circle(frame, (w - 25, 20), 8, dot, -1)

    if last_word:
        col = (0, 255, 0) if last_conf >= CONF_THRESH else (0, 165, 255)
        cv2.putText(frame, f"{last_word} ({last_conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "turn:  " + (" ".join(words) or "_"), (10, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "you:   " + (user or "_"), (10, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "agent: " + (reply or "_")[:60], (10, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 120), 2)
    cv2.putText(frame, "space=sign  r=send  b=back  c=clear  n=new  q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    tracker = SignTracker()
    rec_model = SignRecognizer()
    chat = Chat()
    llm_up = ollama_available()
    print("Ollama:", "up" if llm_up else "down (fallback replies)")

    words, user, reply = [], "", ""
    last_word, last_conf = "", 0.0
    armed, recording, buf = False, False, []

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feat, hand_list, has_hand = tracker.process(frame)
        SignTracker.draw(frame, hand_list)

        if armed and has_hand:               # start recording once a hand appears
            armed, recording, buf = False, True, []
        if recording:
            buf.append(feat)
            if len(buf) >= CLIP_LEN:
                lab, conf = rec_model.predict(buf)
                last_word, last_conf = lab, conf
                if conf >= CONF_THRESH:
                    words.append(lab)        # only commit confident predictions
                recording, buf = False, []

        draw_ui(frame, armed, recording, len(buf) / CLIP_LEN, last_word, last_conf,
                words, user, reply, llm_up)
        cv2.imshow("BNVIP - sign conversation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and not recording:
            armed = True
        elif key == ord("r") and words:
            user = refine(" ".join(words))
            reply = chat.send(user)
            print(f"you: {user!r}\nagent: {reply!r}\n")
            words = []
        elif key == ord("b") and words:
            words.pop()
        elif key == ord("c"):
            words = []
        elif key == ord("n"):
            chat.reset()
            words, user, reply = [], "", ""

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
