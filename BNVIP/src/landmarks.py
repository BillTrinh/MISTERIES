"""MediaPipe Hands wrappers.

- HandTracker  : single hand, normalized 63-dim vector (used by the static
                 fingerspelling classifier).
- SignTracker  : up to two hands, 126-dim per-frame vector with hands slotted
                 by handedness (used by the dynamic word-sign recognizer).
                 Raw normalized image coords are kept so hand *movement* across
                 a sequence is preserved.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FEATURE_DIM = 63          # single hand, 21 * 3  (static classifier)
SIGN_DIM = 126            # two hands, 2 * 21 * 3 (dynamic recognizer)


def _to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _draw(frame, hand_list):
    for lms in hand_list:
        mp_draw.draw_landmarks(
            frame, lms, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style())


# ---------------------------------------------------------------- static
def normalize(hand_landmarks) -> np.ndarray:
    """Normalized (63,) vector: wrist at origin, scaled by hand size."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                   dtype=np.float32)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    return (pts / scale).flatten()


class HandTracker:
    def __init__(self, max_num_hands=1, det_conf=0.6, track_conf=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=max_num_hands,
            min_detection_confidence=det_conf, min_tracking_confidence=track_conf)

    def process(self, frame_bgr):
        results = self.hands.process(_to_rgb(frame_bgr))
        if not results.multi_hand_landmarks:
            return None, None
        hand = results.multi_hand_landmarks[0]
        return normalize(hand), hand

    @staticmethod
    def draw(frame_bgr, hand_landmarks):
        _draw(frame_bgr, [hand_landmarks])

    def close(self):
        self.hands.close()


# ---------------------------------------------------------------- dynamic
class SignTracker:
    """Two-hand tracker: (126,) per-frame feature for dynamic signs."""

    def __init__(self, det_conf=0.6, track_conf=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=det_conf, min_tracking_confidence=track_conf)

    def process(self, frame_bgr):
        """Return (feature_126, hand_landmarks_list, has_hand)."""
        res = self.hands.process(_to_rgb(frame_bgr))
        feat = np.zeros(SIGN_DIM, dtype=np.float32)
        hand_list = []
        if res.multi_hand_landmarks:
            for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label      # 'Left' / 'Right'
                slot = 0 if label == "Right" else 1
                pts = np.array([[p.x, p.y, p.z] for p in lms.landmark],
                               dtype=np.float32).flatten()   # 63
                feat[slot * 63:(slot + 1) * 63] = pts
                hand_list.append(lms)
        return feat, hand_list, bool(hand_list)

    @staticmethod
    def draw(frame_bgr, hand_list):
        _draw(frame_bgr, hand_list)

    def close(self):
        self.hands.close()
