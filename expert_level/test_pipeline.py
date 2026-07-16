from collections import defaultdict
from pathlib import Path
import random
import shutil
import time

import cv2
import face_alignment
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("facial_expression_dataset/train")

OUTPUT_DIR = Path("outputs")
PREVIEW_DIR = OUTPUT_DIR / "preview"
FAILED_DIR = OUTPUT_DIR / "failed"

NUM_IMAGES = 500
RANDOM_SEED = 42

# FER-2013 images are originally 48x48 and tightly cropped.
# Resize the original face and add a constant background around it.
FACE_SIZE = 160
PADDING = 48

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# Device selection
# ============================================================

def pick_device() -> str:
    """
    Select the best available device for FAN.

    SCRFD does not support MPS in face-alignment, so it will
    automatically fall back to CPU on Apple Silicon.
    """
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# Dataset loading
# ============================================================

def collect_images() -> list[Path]:
    """
    Collect and randomly select images from the training dataset.
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{DATASET_DIR.resolve()}"
        )

    image_paths = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(
            f"No images were found inside: "
            f"{DATASET_DIR.resolve()}"
        )

    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)

    return image_paths[:NUM_IMAGES]


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a FER-2013 image, resize it and add constant padding.

    Returns:
        original_gray:
            Original 48x48 grayscale image.

        processed_gray:
            Resized and padded grayscale image.

        rgb:
            RGB image passed to SCRFD and FAN.
    """
    original_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if original_gray is None:
        raise ValueError(
            f"OpenCV cannot read image: {image_path}"
        )

    resized = cv2.resize(
        original_gray,
        (FACE_SIZE, FACE_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    # Use the median intensity as the padding value.
    # This creates a neutral background without reflected facial parts.
    background_value = int(np.median(resized))

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

    return original_gray, processed_gray, rgb


# ============================================================
# Visualization
# ============================================================

def draw_landmarks(
    gray_image: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Draw 68 FAN landmarks on the processed grayscale image.
    """
    output = cv2.cvtColor(
        gray_image,
        cv2.COLOR_GRAY2BGR,
    )

    for x, y in landmarks:
        cv2.circle(
            output,
            center=(
                int(round(x)),
                int(round(y)),
            ),
            radius=2,
            color=(0, 255, 0),
            thickness=-1,
        )

    return output


def save_rgb_image(
    path: Path,
    rgb_image: np.ndarray,
) -> None:
    """
    Save an RGB image correctly using OpenCV.
    """
    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        bgr_image,
    )

    if not success:
        raise IOError(
            f"Could not save image: {path}"
        )


# ============================================================
# Output folders
# ============================================================

def prepare_output_directories() -> None:
    """
    Delete previous outputs and create clean output folders.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAILED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Statistics
# ============================================================

def print_class_statistics(
    class_total: dict[str, int],
    class_detected: dict[str, int],
) -> None:
    """
    Print detection results for each emotion class.
    """
    print("\nPer-class detection results")

    for class_name in sorted(class_total):
        total = class_total[class_name]
        detected = class_detected[class_name]

        rate = (
            detected / total * 100
            if total > 0
            else 0.0
        )

        print(
            f"{class_name:10s}: "
            f"{detected:3d}/{total:3d} "
            f"({rate:6.2f}%)"
        )


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    prepare_output_directories()

    device = pick_device()

    print(f"Working directory  : {Path.cwd()}")
    print(f"Dataset path       : {DATASET_DIR.resolve()}")
    print(f"Device             : {device}")
    print("Face detector       : SCRFD")
    print("Landmark extractor  : FAN")
    print(f"Number of images    : {NUM_IMAGES}")
    print(f"Face resize         : {FACE_SIZE}x{FACE_SIZE}")
    print(f"Padding             : {PADDING}px each side")
    print(
        f"Final input size    : "
        f"{FACE_SIZE + 2 * PADDING}x"
        f"{FACE_SIZE + 2 * PADDING}"
    )
    print()

    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector="scrfd",
        device=device,
        flip_input=False,
        compile=False,
    )

    image_paths = collect_images()

    detected_count = 0
    failed_count = 0
    error_count = 0
    multiple_face_count = 0

    total_times: list[float] = []
    success_times: list[float] = []
    failure_times: list[float] = []

    class_total: dict[str, int] = defaultdict(int)
    class_detected: dict[str, int] = defaultdict(int)

    for index, image_path in enumerate(image_paths):
        class_name = image_path.parent.name
        class_total[class_name] += 1

        try:
            original_gray, processed_gray, rgb = preprocess_image(
                image_path
            )

            if index == 0:
                print("First image debug")
                print(f"Path             : {image_path.resolve()}")
                print(f"Exists           : {image_path.exists()}")
                print(f"Original shape   : {original_gray.shape}")
                print(f"Original dtype   : {original_gray.dtype}")
                print(f"Original min     : {original_gray.min()}")
                print(f"Original max     : {original_gray.max()}")
                print(f"Original mean    : {original_gray.mean():.2f}")
                print(
                    f"Background value : "
                    f"{int(np.median(cv2.resize(original_gray, (FACE_SIZE, FACE_SIZE))))}"
                )
                print(f"Model input      : {rgb.shape}")
                print(f"Input dtype      : {rgb.dtype}")
                print()

            # Save the first ten exact inputs given to SCRFD.
            if index < 10:
                input_name = (
                    f"input_{index:03d}_"
                    f"{class_name}_"
                    f"{image_path.stem}.jpg"
                )

                save_rgb_image(
                    PREVIEW_DIR / input_name,
                    rgb,
                )

            start_time = time.perf_counter()

            predictions = face_aligner.get_landmarks(
                rgb
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            total_times.append(elapsed_ms)

            if predictions is None or len(predictions) == 0:
                failed_count += 1
                failure_times.append(elapsed_ms)

                failed_name = (
                    f"failed_{index:03d}_"
                    f"{class_name}_"
                    f"{image_path.stem}.jpg"
                )

                save_rgb_image(
                    FAILED_DIR / failed_name,
                    rgb,
                )

                print(
                    f"[FAIL] {image_path.name} "
                    f"| class={class_name} "
                    f"| time={elapsed_ms:.2f} ms"
                )

                continue

            if len(predictions) > 1:
                multiple_face_count += 1

                print(
                    f"[WARNING] {image_path.name} "
                    f"| detected_faces={len(predictions)}"
                )

            # FER-2013 normally contains one main face.
            landmarks = predictions[0]

            if landmarks.shape != (68, 2):
                failed_count += 1

                print(
                    f"[INVALID] {image_path.name} "
                    f"| class={class_name} "
                    f"| shape={landmarks.shape}"
                )

                continue

            if not np.isfinite(landmarks).all():
                failed_count += 1

                print(
                    f"[INVALID] {image_path.name} "
                    f"| class={class_name} "
                    f"| landmarks contain NaN or infinity"
                )

                continue

            detected_count += 1
            class_detected[class_name] += 1
            success_times.append(elapsed_ms)

            visualized = draw_landmarks(
                processed_gray,
                landmarks,
            )

            output_name = (
                f"landmarks_{index:03d}_"
                f"{class_name}_"
                f"{image_path.stem}.jpg"
            )

            success = cv2.imwrite(
                str(PREVIEW_DIR / output_name),
                visualized,
            )

            if not success:
                raise IOError(
                    f"Could not save visualization: "
                    f"{PREVIEW_DIR / output_name}"
                )

            print(
                f"[OK] {image_path.name} "
                f"| class={class_name} "
                f"| landmarks={landmarks.shape} "
                f"| time={elapsed_ms:.2f} ms"
            )

        except Exception as error:
            failed_count += 1
            error_count += 1

            print(
                f"[ERROR] {image_path} "
                f"| {type(error).__name__}: {error}"
            )

    total = len(image_paths)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"Total images        : {total}")
    print(f"Detected            : {detected_count}")
    print(f"Failed              : {failed_count}")
    print(f"Runtime errors      : {error_count}")
    print(f"Multiple detections : {multiple_face_count}")

    if total > 0:
        success_rate = detected_count / total * 100.0

        print(
            f"Success rate        : "
            f"{success_rate:.2f}%"
        )

    if total_times:
        print(
            f"Average including warm-up : "
            f"{np.mean(total_times):.2f} ms"
        )

        print(
            f"Median including warm-up  : "
            f"{np.median(total_times):.2f} ms"
        )

    if len(total_times) > 1:
        print(
            f"Average excluding warm-up : "
            f"{np.mean(total_times[1:]):.2f} ms"
        )

    if success_times:
        print(
            f"Average successful frame  : "
            f"{np.mean(success_times):.2f} ms"
        )

    if failure_times:
        print(
            f"Average failed frame      : "
            f"{np.mean(failure_times):.2f} ms"
        )

    print_class_statistics(
        class_total,
        class_detected,
    )

    print("\nOutput folders")
    print(f"Preview images : {PREVIEW_DIR.resolve()}")
    print(f"Failed images  : {FAILED_DIR.resolve()}")


if __name__ == "__main__":
    main()