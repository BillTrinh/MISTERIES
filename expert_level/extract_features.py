from collections import Counter
from pathlib import Path
import argparse
import time

import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("facial_expression_dataset")
CACHE_DIR = Path("cache")
FAILED_DIR = Path("outputs/extraction_failed")

FACE_SIZE = 160
PADDING = 48

COORDINATE_FEATURE_SIZE = 68 * 2
GEOMETRIC_FEATURE_SIZE = 12
FEATURE_SIZE = (
    COORDINATE_FEATURE_SIZE
    + GEOMETRIC_FEATURE_SIZE
)

FEATURE_VERSION = "all_landmarks_plus_geometry_v1"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# Device
# ============================================================

def pick_device() -> str:
    """
    Select the best available device for FAN.

    SCRFD does not support MPS inside face-alignment, so it will
    automatically fall back to CPU on Apple Silicon.
    """
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# Dataset
# ============================================================

def collect_images(split: str) -> list[Path]:
    """
    Collect all valid images from the requested dataset split.
    """
    split_dir = DATASET_ROOT / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: "
            f"{split_dir.resolve()}"
        )

    image_paths = [
        path
        for path in split_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(
            f"No images found inside: "
            f"{split_dir.resolve()}"
        )

    image_paths.sort()

    return image_paths


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read, resize and pad a FER-2013 image.

    Returns:
        processed_gray:
            Resized and padded grayscale image.

        rgb:
            RGB image passed to SCRFD and FAN.
    """
    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if gray is None:
        raise ValueError(
            f"OpenCV cannot read image: "
            f"{image_path}"
        )

    resized = cv2.resize(
        gray,
        (FACE_SIZE, FACE_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    background_value = int(
        np.median(resized)
    )

    processed_gray = cv2.copyMakeBorder(
        resized,
        top=PADDING,
        bottom=PADDING,
        left=PADDING,
        right=PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=background_value,
    )

    rgb = cv2.cvtColor(
        processed_gray,
        cv2.COLOR_GRAY2RGB,
    )

    rgb = np.ascontiguousarray(
        rgb,
        dtype=np.uint8,
    )

    return processed_gray, rgb


# ============================================================
# Landmark normalization
# ============================================================

def normalize_landmarks(
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Normalize 68 facial landmarks to reduce variation caused by
    translation, scale and in-plane head rotation.

    Steps:
        1. Use the midpoint between both eyes as the origin.
        2. Rotate the face so the eyes become horizontal.
        3. Scale coordinates by the distance between both eyes.

    Returns:
        Normalized landmarks with shape (68, 2).
    """
    points = np.asarray(
        landmarks,
        dtype=np.float32,
    ).copy()

    if points.shape != (68, 2):
        raise ValueError(
            f"Expected landmark shape (68, 2), "
            f"got {points.shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "Landmarks contain NaN or infinity"
        )

    left_eye_center = points[36:42].mean(
        axis=0
    )

    right_eye_center = points[42:48].mean(
        axis=0
    )

    eye_midpoint = (
        left_eye_center
        + right_eye_center
    ) / 2.0

    eye_vector = (
        right_eye_center
        - left_eye_center
    )

    eye_distance = float(
        np.linalg.norm(eye_vector)
    )

    if eye_distance < 1e-6:
        raise ValueError(
            "Inter-eye distance is too small"
        )

    # Translation normalization
    points -= eye_midpoint

    # Rotation normalization
    angle = np.arctan2(
        eye_vector[1],
        eye_vector[0],
    )

    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)

    rotation_matrix = np.array(
        [
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle],
        ],
        dtype=np.float32,
    )

    points = points @ rotation_matrix.T

    # Scale normalization
    points /= eye_distance

    if not np.isfinite(points).all():
        raise ValueError(
            "Normalized landmarks contain "
            "NaN or infinity"
        )

    return points.astype(np.float32)


# ============================================================
# Geometric features
# ============================================================

def point_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two landmark points.
    """
    return float(
        np.linalg.norm(
            point_a - point_b
        )
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide safely while avoiding division by zero.
    """
    return float(
        numerator
        / (denominator + 1e-6)
    )


def extract_geometric_features(
    points: np.ndarray,
) -> np.ndarray:
    """
    Extract 12 expression-related geometric features.

    Input:
        Normalized landmarks with shape (68, 2).

    Output:
        Geometric feature vector with shape (12,).
    """
    if points.shape != (68, 2):
        raise ValueError(
            f"Expected normalized landmarks "
            f"with shape (68, 2), got {points.shape}"
        )

    # --------------------------------------------------------
    # Eye aspect ratios
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mouth geometry
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Eyebrow geometry
    # --------------------------------------------------------

    left_brow_center = points[
        17:22
    ].mean(axis=0)

    right_brow_center = points[
        22:27
    ].mean(axis=0)

    left_eye_center = points[
        36:42
    ].mean(axis=0)

    right_eye_center = points[
        42:48
    ].mean(axis=0)

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

    # --------------------------------------------------------
    # Mouth corner geometry
    # --------------------------------------------------------

    mouth_center = points[
        48:60
    ].mean(axis=0)

    left_corner_height = float(
        mouth_center[1]
        - points[48, 1]
    )

    right_corner_height = float(
        mouth_center[1]
        - points[54, 1]
    )

    # --------------------------------------------------------
    # Lip geometry
    # --------------------------------------------------------

    upper_lip_height = point_distance(
        points[51],
        points[62],
    )

    lower_lip_height = point_distance(
        points[57],
        points[66],
    )

    geometric_features = np.array(
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

    if geometric_features.shape != (
        GEOMETRIC_FEATURE_SIZE,
    ):
        raise ValueError(
            "Unexpected geometric feature size: "
            f"{geometric_features.shape}"
        )

    if not np.isfinite(
        geometric_features
    ).all():
        raise ValueError(
            "Geometric features contain "
            "NaN or infinity"
        )

    return geometric_features


# ============================================================
# Final feature vector
# ============================================================

def create_feature_vector(
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the final feature vector.

    Feature composition:
        136 normalized landmark coordinates
        + 12 geometric measurements
        = 148 features

    Returns:
        features:
            Final vector with shape (148,).

        normalized_landmarks:
            Normalized landmarks with shape (68, 2).
    """
    normalized_landmarks = normalize_landmarks(
        landmarks
    )

    coordinate_features = (
        normalized_landmarks
        .flatten()
        .astype(np.float32)
    )

    geometric_features = extract_geometric_features(
        normalized_landmarks
    )

    features = np.concatenate(
        [
            coordinate_features,
            geometric_features,
        ]
    ).astype(np.float32)

    if features.shape != (FEATURE_SIZE,):
        raise ValueError(
            f"Expected feature shape "
            f"({FEATURE_SIZE},), "
            f"got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Final feature vector contains "
            "NaN or infinity"
        )

    return (
        features,
        normalized_landmarks,
    )


# ============================================================
# Face selection
# ============================================================

def select_main_face(
    predictions: list[np.ndarray],
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Select the detected face closest to the image center.

    FER-2013 normally contains one main face, but this handles
    rare multiple-detection cases more safely than predictions[0].
    """
    if not predictions:
        raise ValueError(
            "No landmark predictions provided"
        )

    image_height, image_width = image_shape[
        :2
    ]

    image_center = np.array(
        [
            image_width / 2.0,
            image_height / 2.0,
        ],
        dtype=np.float32,
    )

    best_landmarks = None
    best_distance = float("inf")

    for prediction in predictions:
        landmarks = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if landmarks.shape != (68, 2):
            continue

        face_center = landmarks.mean(
            axis=0
        )

        center_distance = float(
            np.linalg.norm(
                face_center
                - image_center
            )
        )

        if center_distance < best_distance:
            best_distance = center_distance
            best_landmarks = landmarks

    if best_landmarks is None:
        raise ValueError(
            "No valid 68-point face found"
        )

    return best_landmarks


# ============================================================
# Failure visualization
# ============================================================

def save_failed_image(
    split: str,
    image_path: Path,
    rgb_image: np.ndarray,
    index: int,
) -> None:
    """
    Save failed detector inputs for later failure-case analysis.
    """
    split_failed_dir = FAILED_DIR / split

    split_failed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_name = (
        f"{index:06d}_"
        f"{image_path.parent.name}_"
        f"{image_path.stem}.jpg"
    )

    bgr = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(
            split_failed_dir
            / output_name
        ),
        bgr,
    )

    if not success:
        raise IOError(
            f"Could not save failed image: "
            f"{split_failed_dir / output_name}"
        )


# ============================================================
# Extraction
# ============================================================

def extract_split(
    split: str,
    save_failed: bool,
) -> None:
    """
    Extract FAN landmark and geometric features from one split.
    """
    image_paths = collect_images(
        split
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = pick_device()

    print("=" * 70)
    print(f"Feature extraction: {split}")
    print("=" * 70)

    print(
        f"Dataset path         : "
        f"{(DATASET_ROOT / split).resolve()}"
    )

    print(
        f"Number of images     : "
        f"{len(image_paths)}"
    )

    print(
        f"Device               : "
        f"{device}"
    )

    print(
        "Face detector         : SCRFD"
    )

    print(
        "Landmark extractor    : FAN"
    )

    print(
        f"Face resize           : "
        f"{FACE_SIZE}x{FACE_SIZE}"
    )

    print(
        f"Padding               : "
        f"{PADDING}px each side"
    )

    print(
        f"Coordinate features   : "
        f"{COORDINATE_FEATURE_SIZE}"
    )

    print(
        f"Geometric features    : "
        f"{GEOMETRIC_FEATURE_SIZE}"
    )

    print(
        f"Total feature size    : "
        f"{FEATURE_SIZE}"
    )

    print(
        f"Feature version       : "
        f"{FEATURE_VERSION}"
    )

    print()

    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector="scrfd",
        device=device,
        flip_input=False,
        compile=False,
    )

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    paths_list: list[str] = []

    raw_landmarks_list: list[np.ndarray] = []
    normalized_landmarks_list: list[np.ndarray] = []

    failed_paths: list[str] = []
    failed_labels: list[int] = []
    failure_reasons: list[str] = []

    class_total = Counter()
    class_detected = Counter()

    total_times: list[float] = []
    success_times: list[float] = []
    failure_times: list[float] = []

    error_count = 0
    multiple_face_count = 0

    for index, image_path in enumerate(
        tqdm(
            image_paths,
            desc=f"Extracting {split}",
            unit="image",
        )
    ):
        class_name = image_path.parent.name

        if class_name not in CLASS_TO_INDEX:
            print(
                f"\n[SKIP] Unknown class folder: "
                f"{class_name}"
            )
            continue

        class_index = CLASS_TO_INDEX[
            class_name
        ]

        class_total[class_name] += 1

        rgb = None

        try:
            _, rgb = preprocess_image(
                image_path
            )

            start_time = time.perf_counter()

            predictions = face_aligner.get_landmarks(
                rgb
            )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            total_times.append(
                elapsed_ms
            )

            if (
                predictions is None
                or len(predictions) == 0
            ):
                failure_times.append(
                    elapsed_ms
                )

                failed_paths.append(
                    str(image_path)
                )

                failed_labels.append(
                    class_index
                )

                failure_reasons.append(
                    "no_face_detected"
                )

                if save_failed:
                    save_failed_image(
                        split=split,
                        image_path=image_path,
                        rgb_image=rgb,
                        index=index,
                    )

                continue

            if len(predictions) > 1:
                multiple_face_count += 1

            landmarks = select_main_face(
                predictions=predictions,
                image_shape=rgb.shape,
            )

            (
                features,
                normalized_landmarks,
            ) = create_feature_vector(
                landmarks
            )

            features_list.append(
                features
            )

            labels_list.append(
                class_index
            )

            paths_list.append(
                str(image_path)
            )

            raw_landmarks_list.append(
                landmarks.astype(
                    np.float32
                )
            )

            normalized_landmarks_list.append(
                normalized_landmarks.astype(
                    np.float32
                )
            )

            class_detected[
                class_name
            ] += 1

            success_times.append(
                elapsed_ms
            )

        except Exception as error:
            error_count += 1

            failed_paths.append(
                str(image_path)
            )

            failed_labels.append(
                class_index
            )

            failure_reasons.append(
                f"{type(error).__name__}: "
                f"{error}"
            )

            if (
                save_failed
                and rgb is not None
            ):
                try:
                    save_failed_image(
                        split=split,
                        image_path=image_path,
                        rgb_image=rgb,
                        index=index,
                    )
                except Exception:
                    pass

            if error_count <= 20:
                print(
                    f"\n[ERROR] {image_path} "
                    f"| {type(error).__name__}: "
                    f"{error}"
                )

    if not features_list:
        raise RuntimeError(
            f"No usable features were extracted "
            f"from split: {split}"
        )

    X = np.stack(
        features_list,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        labels_list,
        dtype=np.int64,
    )

    paths = np.asarray(
        paths_list,
        dtype=str,
    )

    raw_landmarks = np.stack(
        raw_landmarks_list,
        axis=0,
    ).astype(np.float32)

    normalized_landmarks = np.stack(
        normalized_landmarks_list,
        axis=0,
    ).astype(np.float32)

    failed_paths_array = np.asarray(
        failed_paths,
        dtype=str,
    )

    failed_labels_array = np.asarray(
        failed_labels,
        dtype=np.int64,
    )

    failure_reasons_array = np.asarray(
        failure_reasons,
        dtype=str,
    )

    class_names_array = np.asarray(
        CLASS_NAMES,
        dtype=str,
    )

    output_path = (
        CACHE_DIR
        / f"{split}_features.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        paths=paths,
        raw_landmarks=raw_landmarks,
        normalized_landmarks=normalized_landmarks,
        failed_paths=failed_paths_array,
        failed_labels=failed_labels_array,
        failure_reasons=failure_reasons_array,
        class_names=class_names_array,
        feature_size=np.asarray(
            [FEATURE_SIZE],
            dtype=np.int32,
        ),
        coordinate_feature_size=np.asarray(
            [COORDINATE_FEATURE_SIZE],
            dtype=np.int32,
        ),
        geometric_feature_size=np.asarray(
            [GEOMETRIC_FEATURE_SIZE],
            dtype=np.int32,
        ),
        feature_version=np.asarray(
            [FEATURE_VERSION],
            dtype=str,
        ),
        face_size=np.asarray(
            [FACE_SIZE],
            dtype=np.int32,
        ),
        padding=np.asarray(
            [PADDING],
            dtype=np.int32,
        ),
    )

    total_images = len(image_paths)
    detected_count = len(features_list)
    failed_count = len(failed_paths)

    print("\n" + "=" * 70)
    print(f"Extraction summary: {split}")
    print("=" * 70)

    print(
        f"Total images              : "
        f"{total_images}"
    )

    print(
        f"Extracted features        : "
        f"{detected_count}"
    )

    print(
        f"Failed images             : "
        f"{failed_count}"
    )

    print(
        f"Runtime errors            : "
        f"{error_count}"
    )

    print(
        f"Multiple detections       : "
        f"{multiple_face_count}"
    )

    print(
        f"Feature matrix shape      : "
        f"{X.shape}"
    )

    print(
        f"Label vector shape        : "
        f"{y.shape}"
    )

    print(
        f"Raw landmark array shape  : "
        f"{raw_landmarks.shape}"
    )

    print(
        f"Normalized landmark shape : "
        f"{normalized_landmarks.shape}"
    )

    if total_images > 0:
        success_rate = (
            detected_count
            / total_images
            * 100.0
        )

        print(
            f"Success rate              : "
            f"{success_rate:.2f}%"
        )

    if total_times:
        print(
            f"Average total time        : "
            f"{np.mean(total_times):.2f} ms"
        )

        print(
            f"Median total time         : "
            f"{np.median(total_times):.2f} ms"
        )

    if len(total_times) > 1:
        print(
            f"Average without first     : "
            f"{np.mean(total_times[1:]):.2f} ms"
        )

    if success_times:
        print(
            f"Average success time      : "
            f"{np.mean(success_times):.2f} ms"
        )

    if failure_times:
        print(
            f"Average failure time      : "
            f"{np.mean(failure_times):.2f} ms"
        )

    print("\nPer-class extraction results")

    for class_name in CLASS_NAMES:
        total_class = class_total[
            class_name
        ]

        detected_class = class_detected[
            class_name
        ]

        rate = (
            detected_class
            / total_class
            * 100.0
            if total_class > 0
            else 0.0
        )

        print(
            f"{class_name:10s}: "
            f"{detected_class:5d}/"
            f"{total_class:5d} "
            f"({rate:6.2f}%)"
        )

    print(
        f"\nSaved feature file: "
        f"{output_path.resolve()}"
    )

    if save_failed:
        print(
            f"Saved failed inputs: "
            f"{(FAILED_DIR / split).resolve()}"
        )


# ============================================================
# Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized FAN landmark coordinates "
            "and geometric facial-expression features "
            "from FER-2013 using SCRFD."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
        ],
        required=True,
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--save-failed",
        action="store_true",
        help=(
            "Save images for which SCRFD/FAN extraction failed. "
            "This may consume considerable disk space."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extract_split(
        split=args.split,
        save_failed=args.save_failed,
    )


if __name__ == "__main__":
    main()