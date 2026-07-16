from collections import Counter
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

TRAIN_FEATURES_PATH = Path("cache/train_features.npz")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/training")

MODEL_PATH = MODEL_DIR / "emotion_mlp.joblib"
RESULTS_PATH = OUTPUT_DIR / "training_results.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "validation_confusion_matrix.png"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.txt"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.15

# Moderate oversampling.
# Minority classes are increased only up to this level.
OVERSAMPLE_TARGET = 2500

# Smaller network to reduce overfitting.
HIDDEN_LAYERS = (64, 32)

BATCH_SIZE = 128
LEARNING_RATE = 0.001
ALPHA = 0.005

# Train several candidates and select by external validation Macro F1.
ITERATION_CANDIDATES = [
    100,
    150,
    200,
    250,
]

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


# ============================================================
# Data loading
# ============================================================

def load_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load the extracted training feature file.
    """
    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file does not exist: "
            f"{TRAIN_FEATURES_PATH.resolve()}"
        )

    data = np.load(
        TRAIN_FEATURES_PATH,
        allow_pickle=False,
    )

    required_keys = {
        "X",
        "y",
        "class_names",
    }

    missing_keys = required_keys.difference(
        data.files
    )

    if missing_keys:
        raise KeyError(
            f"Missing keys: {sorted(missing_keys)}"
        )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    y = np.asarray(
        data["y"],
        dtype=np.int64,
    )

    class_names = np.asarray(
        data["class_names"],
        dtype=str,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D, got {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"Expected y to be 1D, got {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y have different sample counts: "
            f"{len(X)} and {len(y)}"
        )

    if X.shape[1] <= 0:
        raise ValueError(
            f"Invalid feature size: {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or infinity"
        )

    return X, y, class_names


# ============================================================
# Class statistics
# ============================================================

def print_class_distribution(
    y: np.ndarray,
    title: str,
) -> None:
    """
    Print the number of samples per class.
    """
    counts = Counter(
        y.tolist()
    )

    print(f"\n{title}")

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"{class_name:10s}: "
            f"{counts.get(class_index, 0):6d}"
        )


# ============================================================
# Moderate oversampling
# ============================================================

def moderate_oversample(
    X: np.ndarray,
    y: np.ndarray,
    target_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample only classes below target_count.

    Majority classes above the target are kept unchanged.
    Validation data must not be passed into this function.
    """
    rng = np.random.default_rng(
        random_seed
    )

    sampled_indices = []

    for class_index in np.unique(y):
        class_indices = np.flatnonzero(
            y == class_index
        )

        class_count = len(
            class_indices
        )

        if class_count < target_count:
            extra_indices = rng.choice(
                class_indices,
                size=target_count - class_count,
                replace=True,
            )

            class_indices = np.concatenate(
                [
                    class_indices,
                    extra_indices,
                ]
            )

        sampled_indices.append(
            class_indices
        )

    balanced_indices = np.concatenate(
        sampled_indices
    )

    rng.shuffle(
        balanced_indices
    )

    return (
        X[balanced_indices],
        y[balanced_indices],
    )


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_model(
    model: MLPClassifier,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[
    float,
    float,
    np.ndarray,
]:
    """
    Evaluate one candidate model.
    """
    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        macro_f1,
        predictions,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """
    Save validation confusion matrix as an image.
    """
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format="d",
    )

    axis.set_title(
        "Validation Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
    )

    plt.close(
        figure
    )


# ============================================================
# Main training pipeline
# ============================================================

def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y, class_names = load_training_data()

    feature_size = X.shape[1]

    print("=" * 70)
    print("MLP Facial Expression Training")
    print("=" * 70)

    print(
        f"Feature file         : "
        f"{TRAIN_FEATURES_PATH.resolve()}"
    )

    print(
        f"Samples              : "
        f"{X.shape[0]}"
    )

    print(
        f"Features             : "
        f"{feature_size}"
    )

    print(
        f"Classes              : "
        f"{len(class_names)}"
    )

    print(
        f"Validation size      : "
        f"{VALIDATION_SIZE:.0%}"
    )

    print(
        f"Hidden layers        : "
        f"{HIDDEN_LAYERS}"
    )

    print(
        f"Oversampling target  : "
        f"{OVERSAMPLE_TARGET}"
    )

    print(
        f"Alpha                : "
        f"{ALPHA}"
    )

    print(
        f"Iteration candidates : "
        f"{ITERATION_CANDIDATES}"
    )

    print_class_distribution(
        y,
        "Original class distribution",
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("\nData split")
    print(
        f"Training samples   : "
        f"{len(X_train)}"
    )
    print(
        f"Validation samples : "
        f"{len(X_validation)}"
    )

    print_class_distribution(
        y_train,
        "Training distribution before oversampling",
    )

    (
        X_train_balanced,
        y_train_balanced,
    ) = moderate_oversample(
        X=X_train,
        y=y_train,
        target_count=OVERSAMPLE_TARGET,
        random_seed=RANDOM_SEED,
    )

    print_class_distribution(
        y_train_balanced,
        "Training distribution after moderate oversampling",
    )

    print(
        f"\nBalanced training samples: "
        f"{len(X_train_balanced)}"
    )

    # Fit scaler only on the balanced training subset.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_balanced
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    best_model = None
    best_predictions = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0
    best_iterations = None
    best_training_seconds = None

    comparison_lines = [
        "MLP candidate comparison",
        "=" * 70,
        "",
    ]

    for max_iterations in ITERATION_CANDIDATES:
        print("\n" + "-" * 70)
        print(
            f"Training candidate: "
            f"max_iter={max_iterations}"
        )
        print("-" * 70)

        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=ALPHA,
            batch_size=BATCH_SIZE,
            learning_rate_init=LEARNING_RATE,
            max_iter=max_iterations,
            shuffle=True,
            early_stopping=False,
            tol=1e-4,
            random_state=RANDOM_SEED,
            verbose=False,
        )

        start_time = time.perf_counter()

        model.fit(
            X_train_scaled,
            y_train_balanced,
        )

        training_seconds = (
            time.perf_counter() - start_time
        )

        (
            validation_accuracy,
            validation_macro_f1,
            validation_predictions,
        ) = evaluate_model(
            model=model,
            X_validation=X_validation_scaled,
            y_validation=y_validation,
        )

        print(
            f"Training time : "
            f"{training_seconds:.2f} seconds"
        )

        print(
            f"Iterations    : "
            f"{model.n_iter_}"
        )

        print(
            f"Accuracy      : "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Macro F1      : "
            f"{validation_macro_f1:.4f}"
        )

        comparison_lines.extend(
            [
                f"max_iter={max_iterations}",
                f"actual_iterations={model.n_iter_}",
                f"training_seconds={training_seconds:.4f}",
                f"accuracy={validation_accuracy:.4f}",
                f"macro_f1={validation_macro_f1:.4f}",
                "",
            ]
        )

        if validation_macro_f1 > best_macro_f1:
            best_model = model
            best_predictions = validation_predictions
            best_macro_f1 = validation_macro_f1
            best_accuracy = validation_accuracy
            best_iterations = max_iterations
            best_training_seconds = training_seconds

    if best_model is None or best_predictions is None:
        raise RuntimeError(
            "No model candidate was successfully trained"
        )

    print("\n" + "=" * 70)
    print("Best validation model")
    print("=" * 70)

    print(
        f"Selected max_iter : "
        f"{best_iterations}"
    )

    print(
        f"Accuracy          : "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{best_macro_f1:.4f}"
    )

    report = classification_report(
        y_validation,
        best_predictions,
        labels=np.arange(
            len(class_names)
        ),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print()
    print(report)

    model_package = {
        "model": best_model,
        "scaler": scaler,
        "class_names": class_names,
        "feature_size": feature_size,
        "feature_version": (
            "selected_landmarks_plus_geometry_v1"
        ),
        "hidden_layers": HIDDEN_LAYERS,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "oversample_target": OVERSAMPLE_TARGET,
        "selected_max_iterations": best_iterations,
        "face_size": 160,
        "padding": 48,
        "random_seed": RANDOM_SEED,
        "validation_accuracy": best_accuracy,
        "validation_macro_f1": best_macro_f1,
        "training_seconds": best_training_seconds,
        "training_samples_original": len(
            X_train
        ),
        "training_samples_balanced": len(
            X_train_balanced
        ),
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    report_text = (
        "Best MLP validation results\n"
        "===========================\n\n"
        f"Selected max_iter : {best_iterations}\n"
        f"Feature size      : {feature_size}\n"
        f"Accuracy          : {best_accuracy:.4f}\n"
        f"Macro F1          : {best_macro_f1:.4f}\n\n"
        f"{report}"
    )

    RESULTS_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    COMPARISON_PATH.write_text(
        "\n".join(
            comparison_lines
        ),
        encoding="utf-8",
    )

    save_confusion_matrix(
        y_true=y_validation,
        y_pred=best_predictions,
        class_names=class_names,
    )

    print("\nSaved outputs")
    print(
        f"Model             : "
        f"{MODEL_PATH.resolve()}"
    )
    print(
        f"Validation report : "
        f"{RESULTS_PATH.resolve()}"
    )
    print(
        f"Model comparison  : "
        f"{COMPARISON_PATH.resolve()}"
    )
    print(
        f"Confusion matrix  : "
        f"{CONFUSION_MATRIX_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()