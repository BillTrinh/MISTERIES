"""Train the dynamic word-sign GRU on collected clips.

Usage:
    python src/train_lstm.py

Loads data/signs/*.pkl, preprocesses to fixed-length sequences, trains a small
GRU, reports val accuracy + confusion matrix + per-window latency, and saves
the model to models/sign_gru.pt.
"""
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sequence_utils import preprocess, augment_clip, T_FIXED, SIGN_DIM
from sign_net import SignGRU

AUG = 5          # augmented copies per training clip

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "signs"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sign_gru.pt"

HIDDEN = 128
LAYERS = 1
EPOCHS = 80
LR = 1e-3


def load_data():
    """Return raw clips (list of variable-length arrays) + labels."""
    files = sorted(DATA_DIR.glob("*.pkl"))
    if not files:
        raise SystemExit(f"No data in {DATA_DIR}. Run collect_sequences.py first.")
    clips, y = [], []
    for f in files:
        with open(f, "rb") as fh:
            cl = pickle.load(fh)
        clips += list(cl)
        y += [f.stem] * len(cl)
        print(f"  {f.stem}: {len(cl)} clips")
    return clips, np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\nLoading data...")
    clips, y = load_data()
    labels = sorted(str(v) for v in set(y))
    lab2idx = {l: i for i, l in enumerate(labels)}
    yi = np.array([lab2idx[str(v)] for v in y])
    print(f"Total: {len(clips)} clips, {len(labels)} classes\n")

    idx_tr, idx_te = train_test_split(
        np.arange(len(clips)), test_size=0.2, random_state=42, stratify=yi)

    rng = np.random.default_rng(0)
    X_tr, y_tr = [], []
    for i in idx_tr:                                  # original + AUG copies
        X_tr.append(preprocess(clips[i], T_FIXED)); y_tr.append(yi[i])
        for _ in range(AUG):
            X_tr.append(preprocess(augment_clip(clips[i], rng), T_FIXED))
            y_tr.append(yi[i])
    X_tr = np.stack(X_tr); y_tr = np.array(y_tr)
    X_te = np.stack([preprocess(clips[i], T_FIXED) for i in idx_te])  # no aug
    y_te = yi[idx_te]
    print(f"Train (with x{AUG} aug): {len(X_tr)} | Val: {len(X_te)}\n")

    Xtr = torch.tensor(X_tr, device=device)
    ytr = torch.tensor(y_tr, device=device)
    Xte = torch.tensor(X_te, device=device)

    model = SignGRU(len(labels), dim=SIGN_DIM, hidden=HIDDEN, layers=LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    n, bs = len(Xtr), 64
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for k in range(0, n, bs):
            b = perm[k:k + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1:3d}  loss {total / n:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).cpu().numpy()
    acc = (pred == y_te).mean()
    print(f"\nVal accuracy: {acc:.3f}\n")

    print("Confusion matrix (rows=true, cols=pred):")
    print("      " + " ".join(f"{l[:4]:>4}" for l in labels))
    cm = confusion_matrix(y_te, pred, labels=list(range(len(labels))))
    for l, row in zip(labels, cm):
        print(f"{l[:5]:>5} " + " ".join(f"{v:>4}" for v in row))
    print("\n" + classification_report(
        y_te, pred, labels=list(range(len(labels))),
        target_names=labels, zero_division=0))

    # latency (one window forward)
    one = Xte[:1]
    with torch.no_grad():
        for _ in range(5):
            model(one)                       # warmup
        t0 = time.perf_counter()
        for _ in range(100):
            model(one)
        ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"Avg forward latency: {ms:.2f} ms/window ({device})")

    torch.save({
        "state": model.state_dict(),
        "labels": labels,
        "dim": SIGN_DIM, "hidden": HIDDEN, "layers": LAYERS, "T": T_FIXED,
    }, MODEL_PATH)
    print(f"\nSaved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
