#!/usr/bin/env python3
"""Detect cubic objects from an Android IP Webcam stream.

Example:
    python3 cubic.py 192.168.150.47:8080
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_MODEL_PATH = "runs/detect/train-6/weights/best.pt"
WINDOW_NAME = "Phone IP Webcam - Cubic Detection"


@dataclass
class SharedFrame:
    frame: object | None = None
    frame_id: int = 0
    error: str | None = None
    stopped: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class DetectionState:
    detected: bool = False
    boxes: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


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


def draw_status(cv2, frame, detected: bool) -> None:
    status = "Detected: Cubic" if detected else "Normal"
    color = (0, 0, 255) if detected else (0, 180, 0)
    width = min(20 + len(status) * 18, frame.shape[1] - 10)

    cv2.rectangle(frame, (10, 10), (width, 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def capture_latest_frames(camera, shared_frame: SharedFrame) -> None:
    while True:
        with shared_frame.lock:
            if shared_frame.stopped:
                return

        ok, frame = camera.read()
        if not ok:
            with shared_frame.lock:
                shared_frame.error = "No frame received from phone camera."
            time.sleep(0.02)
            continue

        with shared_frame.lock:
            shared_frame.frame = frame
            shared_frame.frame_id += 1
            shared_frame.error = None


def detect_latest_frames(
    cv2,
    model,
    shared_frame: SharedFrame,
    detection_state: DetectionState,
    confidence: float,
    imgsz: int,
    process_every: int,
    detect_width: int,
    max_det: int,
    debug: bool,
) -> None:
    last_processed_id = -1

    while True:
        with shared_frame.lock:
            if shared_frame.stopped:
                return
            frame = None if shared_frame.frame is None else shared_frame.frame.copy()
            frame_id = shared_frame.frame_id

        if frame is None or frame_id == last_processed_id or frame_id % process_every != 0:
            time.sleep(0.005)
            continue

        last_processed_id = frame_id
        boxes = []
        source_height, source_width = frame.shape[:2]
        scale = min(detect_width / source_width, 1.0) if detect_width > 0 else 1.0
        detect_frame = frame

        if scale < 1.0:
            detect_height = max(1, int(source_height * scale))
            detect_frame = cv2.resize(frame, (detect_width, detect_height))

        scale_back = 1 / scale
        results = model.predict(
            detect_frame,
            conf=confidence,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )

        for result in results:
            if debug and frame_id % (process_every * 15) == 0:
                predictions = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = str(result.names[class_id])
                    score = float(box.conf[0])
                    predictions.append(f"{label} {score:.2f}")
                print("Predictions:", ", ".join(predictions) or "none")

            for box in result.boxes:
                class_id = int(box.cls[0])
                label = str(result.names[class_id]).lower()
                if label != "cubic":
                    continue

                x1, y1, x2, y2 = (int(value * scale_back) for value in box.xyxy[0])
                score = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, score))

        with detection_state.lock:
            detection_state.boxes = boxes
            detection_state.detected = bool(boxes)


def draw_boxes(cv2, frame, boxes: list[tuple[int, int, int, int, float]]) -> None:
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Cubic {score:.2f}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def run_detection(
    address: str,
    model_path: str,
    confidence: float,
    imgsz: int,
    process_every: int,
    detect_width: int,
    max_det: int,
    debug: bool,
    output_dir: Path,
) -> int:
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

    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    base_url = normalize_base_url(address)
    video_url = f"{base_url}/video"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    try:
        model.fuse()
    except (AttributeError, TypeError):
        pass

    print(f"Opening phone camera: {video_url}")
    camera = cv2.VideoCapture(video_url)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print(f"Could not open phone camera stream: {video_url}", file=sys.stderr)
        return 1

    if not create_window(cv2):
        camera.release()
        return 1

    print("Connected.")
    print("Status shows 'Normal' or 'Detected: Cubic'.")
    print(
        "Detection speed settings: "
        f"imgsz={imgsz}, detect_width={detect_width}, process_every={process_every}, max_det={max_det}"
    )
    print("Press 't' to save a test frame, 'q' or Esc to quit.")

    shared_frame = SharedFrame()
    detection_state = DetectionState()
    capture_thread = threading.Thread(
        target=capture_latest_frames,
        args=(camera, shared_frame),
        daemon=True,
    )
    detection_thread = threading.Thread(
        target=detect_latest_frames,
        args=(
            cv2,
            model,
            shared_frame,
            detection_state,
            confidence,
            imgsz,
            process_every,
            detect_width,
            max_det,
            debug,
        ),
        daemon=True,
    )
    capture_thread.start()
    detection_thread.start()

    try:
        while True:
            with shared_frame.lock:
                frame = None if shared_frame.frame is None else shared_frame.frame.copy()
                error = shared_frame.error

            if frame is None:
                if error:
                    print(error, file=sys.stderr)
                time.sleep(0.01)
                continue

            with detection_state.lock:
                boxes = list(detection_state.boxes)
                detected = detection_state.detected

            draw_boxes(cv2, frame, boxes)
            draw_status(cv2, frame, detected)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0

            if key == ord("t"):
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"cubic-phone-test-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                if cv2.imwrite(str(path), frame):
                    print(f"Saved test frame: {path}")
    finally:
        with shared_frame.lock:
            shared_frame.stopped = True
        capture_thread.join(timeout=1)
        detection_thread.join(timeout=1)
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cubic detection on a phone IP Webcam stream."
    )
    parser.add_argument(
        "address",
        help="Phone IP Webcam address, for example 192.168.150.47:8080",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="YOLO cubic model path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="YOLO inference image size. Smaller is faster.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=2,
        help="Run detection every N frames.",
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=320,
        help="Resize the frame sent to YOLO to this width. Use 0 for full size.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=3,
        help="Maximum number of cubic boxes YOLO should return.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print live model predictions in the terminal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snapshots"),
        help="Folder for saved test frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_detection(
            address=args.address,
            model_path=args.model,
            confidence=args.confidence,
            imgsz=args.imgsz,
            process_every=max(1, args.process_every),
            detect_width=args.detect_width,
            max_det=max(1, args.max_det),
            debug=args.debug,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
