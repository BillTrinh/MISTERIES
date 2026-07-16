from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


# ============================================================
# Configuration
# ============================================================

TEST_FEATURES_PATH = Path("cache/test_features.npz")
MODEL_PATH = Path("models/emotion_mlp.joblib")

OUTPUT_DIR = Path("outputs/evaluation")

REPORT_PATH = OUTPUT_DIR / "test_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "test_confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "test_confusion_matrix_normalized.png"
)
MISCLASSIFIED_PATH = OUTPUT_DIR / "misclassified_samples.npz"

NUM_TIMING_RUNS = 1000


# ============================================================
# Data loading
# ============================================================

def load_test_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load extracted test features and associated metadata.
    """
    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Test feature file does not exist: "
            f"{TEST_FEATURES_PATH.resolve()}"
        )

    data = np.load(
        TEST_FEATURES_PATH,
        allow_pickle=False,
    )

    required_keys = {
        "X",
        "y",
        "class_names",
        "paths",
    }

    missing_keys = required_keys.difference(data.files)

    if missing_keys:
        raise KeyError(
            f"Missing keys in test feature file: "
            f"{sorted(missing_keys)}"
        )

    X_test = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    y_test = np.asarray(
        data["y"],
        dtype=np.int64,
    )

    class_names = np.asarray(
        data["class_names"],
        dtype=str,
    )

    paths = np.asarray(
        data["paths"],
        dtype=str,
    )

    if X_test.ndim != 2:
        raise ValueError(
            f"Expected X_test to be 2D, got {X_test.shape}"
        )

    if y_test.ndim != 1:
        raise ValueError(
            f"Expected y_test to be 1D, got {y_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different sample counts"
        )

    if len(paths) != len(y_test):
        raise ValueError(
            "paths and y_test have different sample counts"
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Test features contain NaN or infinity"
        )

    return X_test, y_test, class_names, paths


def load_model_package() -> dict:
    """
    Load the trained MLP and scaler.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file does not exist: "
            f"{MODEL_PATH.resolve()}"
        )

    package = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "scaler",
        "class_names",
        "feature_size",
    }

    missing_keys = required_keys.difference(package.keys())

    if missing_keys:
        raise KeyError(
            f"Missing keys in model package: "
            f"{sorted(missing_keys)}"
        )

    return package


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    normalized: bool,
    output_path: Path,
) -> None:
    """
    Save a raw or row-normalized confusion matrix.
    """
    normalize_mode = "true" if normalized else None

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        normalize=normalize_mode,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format=".2f" if normalized else "d",
    )

    title = (
        "Normalized Test Confusion Matrix"
        if normalized
        else "Test Confusion Matrix"
    )

    axis.set_title(title)
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


# ============================================================
# Prediction timing
# ============================================================

def measure_prediction_time(
    model,
    scaler,
    X_test: np.ndarray,
) -> dict[str, float]:
    """
    Measure MLP prediction time per sample.

    This measures only:
        scaling + MLP prediction

    It does not include SCRFD or FAN.
    """
    sample_count = min(
        NUM_TIMING_RUNS,
        len(X_test),
    )

    indices = np.linspace(
        0,
        len(X_test) - 1,
        sample_count,
        dtype=np.int64,
    )

    timing_samples = X_test[indices]

    # Warm-up
    warmup = scaler.transform(
        timing_samples[:10]
    )

    model.predict(warmup)

    elapsed_times = []

    for sample in timing_samples:
        sample = sample.reshape(1, -1)

        start_time = time.perf_counter()

        scaled_sample = scaler.transform(sample)
        model.predict(scaled_sample)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        elapsed_times.append(elapsed_ms)

    times = np.asarray(
        elapsed_times,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "maximum_ms": float(np.max(times)),
    }


# ============================================================
# Failure cases
# ============================================================

def save_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    paths: np.ndarray,
) -> int:
    """
    Save metadata for all incorrectly classified test samples.
    """
    wrong_indices = np.flatnonzero(
        y_true != y_pred
    )

    np.savez_compressed(
        MISCLASSIFIED_PATH,
        indices=wrong_indices,
        paths=paths[wrong_indices],
        true_labels=y_true[wrong_indices],
        predicted_labels=y_pred[wrong_indices],
        probabilities=probabilities[wrong_indices],
        confidence=np.max(
            probabilities[wrong_indices],
            axis=1,
        ),
    )

    return len(wrong_indices)


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test, class_names, paths = load_test_data()
    package = load_model_package()

    model = package["model"]
    scaler = package["scaler"]

    model_feature_size = int(
        package["feature_size"]
    )

    if X_test.shape[1] != model_feature_size:
        raise ValueError(
            "Feature-size mismatch:\n"
            f"Model expects {model_feature_size} features, "
            f"but test data contains {X_test.shape[1]}."
        )

    model_class_names = np.asarray(
        package["class_names"],
        dtype=str,
    )

    if not np.array_equal(
        class_names,
        model_class_names,
    ):
        raise ValueError(
            "Class-name order differs between model and test data"
        )

    print("=" * 70)
    print("MLP Test Evaluation")
    print("=" * 70)

    print(f"Model path      : {MODEL_PATH.resolve()}")
    print(f"Test file       : {TEST_FEATURES_PATH.resolve()}")
    print(f"Test samples    : {len(X_test)}")
    print(f"Feature size    : {X_test.shape[1]}")
    print(f"Classes         : {len(class_names)}")

    X_test_scaled = scaler.transform(
        X_test
    )

    prediction_start = time.perf_counter()

    y_pred = model.predict(
        X_test_scaled
    )

    batch_prediction_seconds = (
        time.perf_counter() - prediction_start
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )

    test_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    timing = measure_prediction_time(
        model=model,
        scaler=scaler,
        X_test=X_test,
    )

    misclassified_count = save_misclassified_samples(
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        paths=paths,
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        normalized=False,
        output_path=CONFUSION_MATRIX_PATH,
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        normalized=True,
        output_path=NORMALIZED_CONFUSION_MATRIX_PATH,
    )

    validation_accuracy = package.get(
        "validation_accuracy"
    )

    validation_macro_f1 = package.get(
        "validation_macro_f1"
    )

    report_text = (
        "MLP Test Evaluation\n"
        "===================\n\n"
        f"Test samples              : {len(X_test)}\n"
        f"Feature size              : {X_test.shape[1]}\n"
        f"Test accuracy             : {test_accuracy:.4f}\n"
        f"Test macro F1             : {test_macro_f1:.4f}\n"
        f"Test weighted F1          : {test_weighted_f1:.4f}\n"
        f"Misclassified samples     : {misclassified_count}\n"
        f"Batch prediction time     : "
        f"{batch_prediction_seconds:.6f} seconds\n"
        f"Mean single prediction    : "
        f"{timing['mean_ms']:.4f} ms\n"
        f"Median single prediction  : "
        f"{timing['median_ms']:.4f} ms\n"
        f"95th percentile           : "
        f"{timing['p95_ms']:.4f} ms\n"
        f"Maximum single prediction : "
        f"{timing['maximum_ms']:.4f} ms\n"
    )

    if validation_accuracy is not None:
        report_text += (
            f"Validation accuracy       : "
            f"{validation_accuracy:.4f}\n"
        )

    if validation_macro_f1 is not None:
        report_text += (
            f"Validation macro F1       : "
            f"{validation_macro_f1:.4f}\n"
        )

    report_text += "\n" + report

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nTest results")
    print(f"Accuracy    : {test_accuracy:.4f}")
    print(f"Macro F1    : {test_macro_f1:.4f}")
    print(f"Weighted F1 : {test_weighted_f1:.4f}")
    print(f"Wrong       : {misclassified_count}/{len(y_test)}")

    print("\nPrediction time — scaler + MLP")
    print(f"Mean   : {timing['mean_ms']:.4f} ms")
    print(f"Median : {timing['median_ms']:.4f} ms")
    print(f"P95    : {timing['p95_ms']:.4f} ms")
    print(f"Maximum: {timing['maximum_ms']:.4f} ms")

    print("\nClassification report")
    print(report)

    print("Saved outputs")
    print(f"Report              : {REPORT_PATH.resolve()}")
    print(
        f"Confusion matrix    : "
        f"{CONFUSION_MATRIX_PATH.resolve()}"
    )
    print(
        f"Normalized matrix  : "
        f"{NORMALIZED_CONFUSION_MATRIX_PATH.resolve()}"
    )
    print(
        f"Failure metadata   : "
        f"{MISCLASSIFIED_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()