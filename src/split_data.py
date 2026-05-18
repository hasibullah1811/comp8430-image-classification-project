"""
Shuffle and split raw class folders into train (80%) and val (20%) sets.

Expected layout before running:

    data/raw/
        dark_70_cocoa/
            photo1.jpg
            photo2.jpg
        sea_salt/
            ...

After running:

    data/train/<class_name>/...
    data/val/<class_name>/...

Run from the project root or from src/ — paths default to ../data/ when executed inside src/.
"""

import argparse
import os
import random
import shutil

# Common image extensions we expect from phone / camera captures.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default split ratio: 80% training, 20% validation.
TRAIN_RATIO = 0.8


def list_images(class_dir: str):
    """Return full paths to image files inside one class folder."""
    images = []
    for filename in os.listdir(class_dir):
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            images.append(os.path.join(class_dir, filename))
    return images


def split_class_images(
    image_paths,
    train_ratio: float,
    rng: random.Random,
):
    """
    Shuffle paths in place (via a copy) and return train / val partitions.

    For very small classes we still try to place at least one image in val
    when there are two or more images, so the validation loop can run.
    """
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total == 0:
        return [], []

    # Integer split; at least one training image when n >= 1.
    n_train = int(n_total * train_ratio)
    if n_train == 0:
        n_train = 1
    if n_train >= n_total and n_total > 1:
        # Leave at least one sample for validation when possible.
        n_train = n_total - 1

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:]
    return train_paths, val_paths


def copy_split_files(file_paths, destination_dir: str) -> None:
    """Copy a list of images into the target class folder."""
    os.makedirs(destination_dir, exist_ok=True)
    for src_path in file_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(destination_dir, filename)
        shutil.copy2(src_path, dst_path)


def prepare_output_dirs(train_dir: str, val_dir: str) -> None:
    """
    Remove previous train/val folders so re-running the script does not
    duplicate files or mix old and new splits.
    """
    for split_dir in (train_dir, val_dir):
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)


def split_dataset(
    raw_dir: str,
    train_dir: str,
    val_dir: str,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> None:
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: '{raw_dir}'. "
            "Create it and add one subfolder per chocolate class."
        )

    rng = random.Random(seed)
    prepare_output_dirs(train_dir, val_dir)

    class_names = sorted(
        name
        for name in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, name))
    )

    if not class_names:
        raise RuntimeError(f"No class subfolders found under '{raw_dir}'.")

    print(f"Splitting {len(class_names)} classes from '{raw_dir}'")
    print(f"  -> train: {train_dir}")
    print(f"  -> val:   {val_dir}")
    print(f"Train ratio: {train_ratio:.0%} | Random seed: {seed}\n")

    total_train = 0
    total_val = 0

    for class_name in class_names:
        class_raw_dir = os.path.join(raw_dir, class_name)
        images = list_images(class_raw_dir)

        if not images:
            print(f"  [skip] {class_name}: no images found")
            continue

        train_paths, val_paths = split_class_images(images, train_ratio, rng)

        copy_split_files(train_paths, os.path.join(train_dir, class_name))
        copy_split_files(val_paths, os.path.join(val_dir, class_name))

        total_train += len(train_paths)
        total_val += len(val_paths)
        print(
            f"  {class_name}: {len(images)} images "
            f"-> {len(train_paths)} train, {len(val_paths)} val"
        )

    print(f"\nDone. Copied {total_train} training and {total_val} validation images.")


def parse_args():
    # Default paths assume the script lives in src/ and data/ is one level up.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_data = os.path.join(project_root, "data")

    parser = argparse.ArgumentParser(
        description="Shuffle and split raw class folders into train/val sets."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(default_data, "raw"),
        help="Directory with one subfolder per class (default: data/raw/).",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=os.path.join(default_data, "train"),
        help="Output directory for training images (default: data/train/).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=os.path.join(default_data, "val"),
        help="Output directory for validation images (default: data/val/).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of images assigned to training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        raw_dir=args.raw_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
