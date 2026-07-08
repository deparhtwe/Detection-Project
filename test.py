#!/usr/bin/env python3
"""Show YOLO prediction result images from runs/detect/predict.

This script is for normal terminal use. IPython's display() only renders images
inside notebooks, so this creates a viewable gallery image instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def find_images(input_dir: Path) -> list[Path]:
    image_types = ("*.jpg", "*.jpeg", "*.png")
    images: list[Path] = []

    for image_type in image_types:
        images.extend(input_dir.glob(image_type))

    return sorted(images)


def resize_to_tile(image, tile_width: int, tile_height: int):
    height, width = image.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (new_width, new_height))


def make_gallery(images: list[Path], output_path: Path, columns: int, tile_size: int) -> Path:
    if not images:
        raise ValueError("no result images found")

    rows = (len(images) + columns - 1) // columns
    label_height = 34
    canvas_width = columns * tile_size
    canvas_height = rows * (tile_size + label_height)
    canvas = 255 * cv2.UMat(canvas_height, canvas_width, cv2.CV_8UC3).get()

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        resized = resize_to_tile(image, tile_size, tile_size)
        row = index // columns
        column = index % columns
        x = column * tile_size
        y = row * (tile_size + label_height)

        image_y = y + label_height
        offset_x = x + (tile_size - resized.shape[1]) // 2
        offset_y = image_y + (tile_size - resized.shape[0]) // 2
        canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized

        cv2.putText(
            canvas,
            image_path.name[:32],
            (x + 8, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not save gallery to {output_path}")

    return output_path


def show_images(images: list[Path]) -> None:
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        cv2.imshow(str(image_path.name), image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in (ord("q"), 27):
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or show a gallery of YOLO result images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/detect/predict"),
        help="Folder containing YOLO prediction images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detect/predict_gallery.jpg"),
        help="Gallery image to create.",
    )
    parser.add_argument("--columns", type=int, default=3, help="Number of columns in the gallery.")
    parser.add_argument("--tile-size", type=int, default=360, help="Size of each image tile.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show each image in an OpenCV window. Press any key for next, q to quit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = find_images(args.input_dir)

    if not images:
        print(f"No images found in {args.input_dir}")
        return 1

    gallery_path = make_gallery(images, args.output, args.columns, args.tile_size)
    print(f"Found {len(images)} result images.")
    print(f"Gallery saved to: {gallery_path.resolve()}")

    if args.show:
        show_images(images)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
