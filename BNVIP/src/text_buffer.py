"""Turn a stream of per-frame letter predictions into typed text.

Stabilization: a letter is only committed after it has been predicted with
high confidence for `hold_frames` consecutive frames. After committing, the
same letter won't repeat until the prediction changes (debounce). A gap with
no hand detected commits a space.
"""


class TextBuffer:
    def __init__(self, hold_frames=8, conf_thresh=0.6, space_frames=15):
        self.hold_frames = hold_frames
        self.conf_thresh = conf_thresh
        self.space_frames = space_frames

        self.text = ""
        self._candidate = None
        self._streak = 0
        self._last_committed = None   # blocks immediate repeats
        self._empty_streak = 0

    def update(self, label, confidence):
        """Call once per frame. label=None when no hand is detected."""
        if label is None or confidence < self.conf_thresh:
            self._candidate = None
            self._streak = 0
            self._last_committed = None      # a gap re-arms repeats
            self._empty_streak += 1
            if self._empty_streak == self.space_frames and \
                    self.text and not self.text.endswith(" "):
                self.text += " "
            return

        self._empty_streak = 0
        if label == self._candidate:
            self._streak += 1
        else:
            self._candidate = label
            self._streak = 1

        if self._streak == self.hold_frames and label != self._last_committed:
            self.text += label
            self._last_committed = label

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self._last_committed = None
