"""Train a fingerspelling classifier on collected keypoints.

Usage:
    python src/train_classifier.py

Loads data/fingerspelling/*.npy, trains KNN and a small MLP, keeps the better
one on a held-out split, reports accuracy + confusion matrix + per-prediction
latency, and saves the model to models/fingerspell.pkl.
"""
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fingerspelling"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fingerspell.pkl"


def load_data():
    X, y = [], []
    files = sorted(DATA_DIR.glob("*.npy"))
    if not files:
        raise SystemExit(f"No data in {DATA_DIR}. Run collect_data.py first.")
    for f in files:
        arr = np.load(f)
        X.append(arr)
        y += [f.stem] * len(arr)
        print(f"  {f.stem}: {len(arr)} samples")
    return np.vstack(X), np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    X, y = load_data()
    print(f"Total: {len(X)} samples, {len(set(y))} classes\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"{name}: test accuracy = {acc:.3f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest: {best_name} ({best_acc:.3f})\n")

    y_pred = best_model.predict(X_te)
    labels = sorted(set(y))
    print("Confusion matrix (rows=true, cols=pred):")
    print("     " + " ".join(f"{l:>3}" for l in labels))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    for l, row in zip(labels, cm):
        print(f"{l:>3}: " + " ".join(f"{v:>3}" for v in row))
    print("\n" + classification_report(y_te, y_pred, zero_division=0))

    # latency check (assignment wants < 30 ms / prediction)
    one = X_te[:1]
    t0 = time.perf_counter()
    for _ in range(200):
        best_model.predict(one)
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"Avg prediction latency: {ms:.2f} ms  ({'OK' if ms < 30 else 'TOO SLOW'} for 30 FPS)")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "labels": labels}, f)
    print(f"\nSaved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
