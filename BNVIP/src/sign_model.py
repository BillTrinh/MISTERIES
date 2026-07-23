"""Load the trained GRU and classify a keypoint sequence into a word."""
from pathlib import Path

import numpy as np
import torch

from sequence_utils import preprocess
from sign_net import SignGRU

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sign_gru.pt"


class SignRecognizer:
    def __init__(self, path=MODEL_PATH, device="cpu"):
        if not Path(path).exists():
            raise SystemExit(f"Model not found: {path}. Run train_lstm.py first.")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.labels = [str(l) for l in ckpt["labels"]]
        self.T = ckpt["T"]
        self.device = device
        self.model = SignGRU(len(self.labels), dim=ckpt["dim"],
                             hidden=ckpt["hidden"], layers=ckpt["layers"]).to(device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    def predict(self, seq):
        """Return (label, confidence) for a (frames, 126) sequence."""
        x = preprocess(seq, self.T)                       # (T, 126)
        t = torch.tensor(x, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prob = torch.softmax(self.model(t), dim=1)[0].cpu().numpy()
        i = int(np.argmax(prob))
        return self.labels[i], float(prob[i])
