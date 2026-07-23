"""Load the trained fingerspelling model and predict a letter from keypoints."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fingerspell.pkl"


class FingerspellClassifier:
    def __init__(self, path=MODEL_PATH):
        if not Path(path).exists():
            raise SystemExit(f"Model not found: {path}. Run train_classifier.py first.")
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.labels = bundle["labels"]

    def predict(self, feature_vec):
        """Return (label, confidence) for a (63,) feature vector."""
        x = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            i = int(np.argmax(proba))
            return self.model.classes_[i], float(proba[i])
        return self.model.predict(x)[0], 1.0
