"""Create transformed positive examples for the crack triplet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {directory}")
    return images


def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def shear_image(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    shear = np.array([[1.0, np.tan(np.deg2rad(degrees)), 0.0], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(
        image,
        shear,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        image,
        rotation,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def create_positive(
    image: np.ndarray,
    rng: np.random.Generator,
    rotation_range: tuple[int, int],
    shear_range: tuple[int, int],
    brightness_range: tuple[int, int],
) -> np.ndarray:
    rotation = int(rng.integers(rotation_range[0], rotation_range[1] + 1))
    shear = int(rng.integers(shear_range[0], shear_range[1] + 1))
    brightness = int(rng.integers(brightness_range[0], brightness_range[1] + 1))
    transformed = adjust_brightness(image, brightness)
    transformed = shear_image(transformed, shear)
    return rotate_image(transformed, rotation)


def parse_range(value: str) -> tuple[int, int]:
    try:
        first, second = (int(part.strip()) for part in value.split(",", maxsplit=1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a range such as -10,10") from error
    if first > second:
        raise argparse.ArgumentTypeError("Range start must not exceed range end")
    return first, second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--rotation", type=parse_range, default=(-10, 10))
    parser.add_argument("--shear", type=parse_range, default=(-20, 20))
    parser.add_argument("--brightness", type=parse_range, default=(-15, -15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchors = list_images(args.anchor_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("limit must be positive")
        anchors = anchors[:args.limit]
    args.positive_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    written = 0
    skipped = 0
    for anchor_path in tqdm(anchors, desc="Creating positives"):
        output_path = args.positive_dir / anchor_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {anchor_path}")
        positive = create_positive(image, rng, args.rotation, args.shear, args.brightness)
        if not cv2.imwrite(str(output_path), positive):
            raise OSError(f"Could not write image: {output_path}")
        written += 1
    print(f"Created {written} positives; skipped {skipped} existing files")


if __name__ == "__main__":
    main()
