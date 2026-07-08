#!/usr/bin/env python3
"""General object detection from an Android IP Webcam stream.

Example:
    python3 testing.py 192.168.150.47:8080
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urlparse


WINDOW_NAME = "Phone IP Webcam - General Object Detection"


def normalize_base_url(address: str) -> str:
    address = address.strip()
    if not address:
        raise ValueError("phone address cannot be empty")

    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"

    parsed = urlparse(address)
    if not parsed.netloc:
        raise ValueError(f"invalid phone address: {address}")

    return address.rstrip("/")


def create_window(cv2) -> bool:
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        return True
    except cv2.error as exc:
        print(
            "OpenCV cannot open a popup window.\n\n"
            "Fix it with:\n"
            "  source .venv/bin/activate\n"
            "  pip uninstall -y opencv-python-headless opencv-python\n"
            "  pip install opencv-python\n",
            file=sys.stderr,
        )
        print(f"Original OpenCV error:\n{exc}", file=sys.stderr)
        return False


def run_detection(address: str, model_path: str, confidence: float) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        print(
            "Missing Python package. Run:\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Missing module: {exc.name}", file=sys.stderr)
        return 1

    base_url = normalize_base_url(address)
    video_url = f"{base_url}/video"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Opening phone camera: {video_url}")
    camera = cv2.VideoCapture(video_url)
    if not camera.isOpened():
        print(f"Could not open phone camera stream: {video_url}", file=sys.stderr)
        return 1

    if not create_window(cv2):
        camera.release()
        return 1

    print("Connected.")
    print("Press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("No frame received from phone camera.", file=sys.stderr)
                return 1

            results = model.predict(frame, conf=confidence, verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow(WINDOW_NAME, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run general YOLO object detection on a phone IP Webcam stream."
    )
    parser.add_argument(
        "address",
        help="Phone IP Webcam address, for example 192.168.150.47:8080",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Minimum detection confidence from 0.0 to 1.0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_detection(args.address, args.model, args.confidence)
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



