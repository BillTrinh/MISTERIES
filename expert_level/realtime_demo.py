from collections import deque
from pathlib import Path
import time

import cv2
import face_alignment
import joblib
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("models/emotion_mlp.joblib")

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Run the expensive SCRFD + FAN pipeline every N frames.
PROCESS_EVERY_N_FRAMES = 2

# Average probabilities over recent predictions to reduce flickering.
SMOOTHING_WINDOW = 7

CONFIDENCE_THRESHOLD = 0.30

COORDINATE_FEATURE_SIZE = 136
GEOMETRIC_FEATURE_SIZE = 12
FEATURE_SIZE = 148


# ============================================================
# Device
# ============================================================

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# Landmark normalization
# ============================================================

def normalize_landmarks(
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Normalize 68 landmarks for translation, rotation and scale.
    Returns shape (68, 2).
    """
    points = np.asarray(
        landmarks,
        dtype=np.float32,
    ).copy()

    if points.shape != (68, 2):
        raise ValueError(
            f"Expected landmarks shape (68, 2), got {points.shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "Landmarks contain NaN or infinity"
        )

    left_eye_center = points[36:42].mean(axis=0)
    right_eye_center = points[42:48].mean(axis=0)

    eye_midpoint = (
        left_eye_center + right_eye_center
    ) / 2.0

    eye_vector = (
        right_eye_center - left_eye_center
    )

    eye_distance = float(
        np.linalg.norm(eye_vector)
    )

    if eye_distance < 1e-6:
        raise ValueError(
            "Inter-eye distance is too small"
        )

    points -= eye_midpoint

    angle = np.arctan2(
        eye_vector[1],
        eye_vector[0],
    )

    rotation_matrix = np.array(
        [
            [
                np.cos(-angle),
                -np.sin(-angle),
            ],
            [
                np.sin(-angle),
                np.cos(-angle),
            ],
        ],
        dtype=np.float32,
    )

    points = points @ rotation_matrix.T
    points /= eye_distance

    if not np.isfinite(points).all():
        raise ValueError(
            "Normalized landmarks contain NaN or infinity"
        )

    return points.astype(np.float32)


# ============================================================
# Geometric features
# ============================================================

def point_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            point_a - point_b
        )
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    return float(
        numerator
        / (denominator + 1e-6)
    )


def extract_geometric_features(
    points: np.ndarray,
) -> np.ndarray:
    """
    Produce the same 12 geometric features used during training.
    """
    left_eye_width = point_distance(
        points[36],
        points[39],
    )

    left_eye_height = (
        point_distance(
            points[37],
            points[41],
        )
        + point_distance(
            points[38],
            points[40],
        )
    ) / 2.0

    right_eye_width = point_distance(
        points[42],
        points[45],
    )

    right_eye_height = (
        point_distance(
            points[43],
            points[47],
        )
        + point_distance(
            points[44],
            points[46],
        )
    ) / 2.0

    left_eye_ratio = safe_ratio(
        left_eye_height,
        left_eye_width,
    )

    right_eye_ratio = safe_ratio(
        right_eye_height,
        right_eye_width,
    )

    mouth_width = point_distance(
        points[48],
        points[54],
    )

    outer_mouth_height = (
        point_distance(
            points[50],
            points[58],
        )
        + point_distance(
            points[51],
            points[57],
        )
        + point_distance(
            points[52],
            points[56],
        )
    ) / 3.0

    inner_mouth_width = point_distance(
        points[60],
        points[64],
    )

    inner_mouth_height = (
        point_distance(
            points[61],
            points[67],
        )
        + point_distance(
            points[62],
            points[66],
        )
        + point_distance(
            points[63],
            points[65],
        )
    ) / 3.0

    outer_mouth_ratio = safe_ratio(
        outer_mouth_height,
        mouth_width,
    )

    inner_mouth_ratio = safe_ratio(
        inner_mouth_height,
        inner_mouth_width,
    )

    left_brow_center = points[17:22].mean(axis=0)
    right_brow_center = points[22:27].mean(axis=0)

    left_eye_center = points[36:42].mean(axis=0)
    right_eye_center = points[42:48].mean(axis=0)

    left_brow_eye_distance = point_distance(
        left_brow_center,
        left_eye_center,
    )

    right_brow_eye_distance = point_distance(
        right_brow_center,
        right_eye_center,
    )

    inner_brow_distance = point_distance(
        points[21],
        points[22],
    )

    mouth_center = points[48:60].mean(axis=0)

    left_corner_height = float(
        mouth_center[1] - points[48, 1]
    )

    right_corner_height = float(
        mouth_center[1] - points[54, 1]
    )

    upper_lip_height = point_distance(
        points[51],
        points[62],
    )

    lower_lip_height = point_distance(
        points[57],
        points[66],
    )

    features = np.array(
        [
            left_eye_ratio,
            right_eye_ratio,
            outer_mouth_ratio,
            inner_mouth_ratio,
            mouth_width,
            left_brow_eye_distance,
            right_brow_eye_distance,
            inner_brow_distance,
            left_corner_height,
            right_corner_height,
            upper_lip_height,
            lower_lip_height,
        ],
        dtype=np.float32,
    )

    if features.shape != (12,):
        raise ValueError(
            f"Expected 12 geometric features, got {features.shape}"
        )

    return features


def create_feature_vector(
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    136 normalized landmark coordinates + 12 geometry = 148.
    """
    normalized = normalize_landmarks(
        landmarks
    )

    coordinate_features = (
        normalized
        .flatten()
        .astype(np.float32)
    )

    geometric_features = extract_geometric_features(
        normalized
    )

    features = np.concatenate(
        [
            coordinate_features,
            geometric_features,
        ]
    ).astype(np.float32)

    if features.shape != (FEATURE_SIZE,):
        raise ValueError(
            f"Expected {FEATURE_SIZE} features, got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature vector contains NaN or infinity"
        )

    return features


# ============================================================
# Visualization
# ============================================================

def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
) -> None:
    for x, y in landmarks:
        cv2.circle(
            frame,
            (
                int(round(x)),
                int(round(y)),
            ),
            1,
            (0, 255, 0),
            -1,
        )


def draw_face_box(
    frame: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[int, int, int, int]:
    x_min = max(
        0,
        int(np.min(landmarks[:, 0])) - 20,
    )

    y_min = max(
        0,
        int(np.min(landmarks[:, 1])) - 30,
    )

    x_max = min(
        frame.shape[1] - 1,
        int(np.max(landmarks[:, 0])) + 20,
    )

    y_max = min(
        frame.shape[0] - 1,
        int(np.max(landmarks[:, 1])) + 20,
    )

    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2,
    )

    return x_min, y_min, x_max, y_max


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist: {MODEL_PATH.resolve()}"
        )

    package = joblib.load(
        MODEL_PATH
    )

    required_keys = {
        "model",
        "scaler",
        "class_names",
        "feature_size",
    }

    missing_keys = required_keys.difference(
        package.keys()
    )

    if missing_keys:
        raise KeyError(
            f"Missing model keys: {sorted(missing_keys)}"
        )

    model = package["model"]
    scaler = package["scaler"]

    class_names = np.asarray(
        package["class_names"],
        dtype=str,
    )

    model_feature_size = int(
        package["feature_size"]
    )

    if model_feature_size != FEATURE_SIZE:
        raise ValueError(
            f"Model expects {model_feature_size} features, "
            f"but realtime code creates {FEATURE_SIZE}."
        )

    device = pick_device()

    print("=" * 65)
    print("Realtime Facial Expression Demo")
    print("=" * 65)

    print(f"Model          : {MODEL_PATH.resolve()}")
    print(f"Feature size   : {model_feature_size}")
    print(f"Device         : {device}")
    print("Detector       : SCRFD")
    print("Landmarks      : FAN")
    print("Press Q or ESC to quit.")
    print()

    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector="scrfd",
        device=device,
        flip_input=False,
        compile=False,
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. "
            "Check macOS camera permissions."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    probability_buffer = deque(
        maxlen=SMOOTHING_WINDOW
    )

    frame_index = 0

    last_landmarks = None
    last_label = "No face"
    last_confidence = 0.0
    last_processing_ms = 0.0

    previous_display_time = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the webcam for a natural preview.
            frame = cv2.flip(
                frame,
                1,
            )

            if frame_index % PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                start_time = time.perf_counter()

                predictions = face_aligner.get_landmarks(
                    rgb
                )

                last_processing_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000.0

                if predictions is None or len(predictions) == 0:
                    last_landmarks = None
                    last_label = "No face"
                    last_confidence = 0.0
                    probability_buffer.clear()

                else:
                    # Choose the face closest to the frame center.
                    frame_center = np.array(
                        [
                            frame.shape[1] / 2.0,
                            frame.shape[0] / 2.0,
                        ],
                        dtype=np.float32,
                    )

                    valid_faces = [
                        np.asarray(
                            prediction,
                            dtype=np.float32,
                        )
                        for prediction in predictions
                        if np.asarray(prediction).shape == (68, 2)
                    ]

                    if valid_faces:
                        last_landmarks = min(
                            valid_faces,
                            key=lambda points: np.linalg.norm(
                                points.mean(axis=0)
                                - frame_center
                            ),
                        )

                        feature_vector = create_feature_vector(
                            last_landmarks
                        ).reshape(1, -1)

                        scaled_feature = scaler.transform(
                            feature_vector
                        )

                        probabilities = model.predict_proba(
                            scaled_feature
                        )[0]

                        probability_buffer.append(
                            probabilities
                        )

                        smoothed_probabilities = np.mean(
                            np.stack(
                                probability_buffer,
                                axis=0,
                            ),
                            axis=0,
                        )

                        predicted_index = int(
                            np.argmax(
                                smoothed_probabilities
                            )
                        )

                        last_confidence = float(
                            smoothed_probabilities[
                                predicted_index
                            ]
                        )

                        if last_confidence >= CONFIDENCE_THRESHOLD:
                            last_label = class_names[
                                predicted_index
                            ]
                        else:
                            last_label = "Uncertain"

            if last_landmarks is not None:
                draw_landmarks(
                    frame,
                    last_landmarks,
                )

                x_min, y_min, _, _ = draw_face_box(
                    frame,
                    last_landmarks,
                )

                label_text = (
                    f"{last_label}: "
                    f"{last_confidence * 100:.1f}%"
                )

                text_y = max(
                    30,
                    y_min - 10,
                )

                cv2.putText(
                    frame,
                    label_text,
                    (x_min, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            current_display_time = time.perf_counter()

            elapsed_display = (
                current_display_time
                - previous_display_time
            )

            if elapsed_display > 0:
                instant_fps = (
                    1.0 / elapsed_display
                )

                display_fps = (
                    0.90 * display_fps
                    + 0.10 * instant_fps
                    if display_fps > 0
                    else instant_fps
                )

            previous_display_time = current_display_time

            cv2.putText(
                frame,
                f"Display FPS: {display_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"SCRFD + FAN: "
                    f"{last_processing_ms:.1f} ms"
                ),
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Realtime Facial Expression",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in {
                ord("q"),
                27,
            }:
                break

            frame_index += 1

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()