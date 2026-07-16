from pathlib import Path
from collections import Counter

DATASET_DIR = Path("facial_expression_dataset")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def count_images(split_dir: Path) -> Counter:
    counts = Counter()

    if not split_dir.exists():
        print(f"Missing folder: {split_dir}")
        return counts

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for path in class_dir.rglob("*")
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        counts[class_dir.name] = count

    return counts


def print_counts(split_name: str, counts: Counter) -> None:
    print(f"\n{split_name.upper()}")

    total = 0

    for class_name, count in counts.items():
        print(f"{class_name:12s}: {count}")
        total += count

    print(f"{'Total':12s}: {total}")


def main() -> None:
    print(f"Dataset path: {DATASET_DIR.resolve()}")

    train_counts = count_images(DATASET_DIR / "train")
    test_counts = count_images(DATASET_DIR / "test")

    print_counts("train", train_counts)
    print_counts("test", test_counts)


if __name__ == "__main__":
    main()