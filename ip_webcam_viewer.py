#!/usr/bin/env python3
"""View and capture video from the Android IP Webcam app.

Typical IP Webcam app address:
    http://192.168.1.23:8080

Live video endpoint used by this script:
    http://192.168.1.23:8080/video
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_WEAPON_MODEL_PATH = "runs/detect/train-7/weights/best.pt"
DEFAULT_FIGHT_MODEL_PATH = "runs/detect/train-5/weights/best.pt"
WINDOW_NAME = "Phone IP Webcam - Safety Detection"
HISTORY_WINDOW_NAME = "Detection History - CRUD Console"

DEFAULT_WEAPON_LABELS = {
    "weapon",
}

DEFAULT_FIGHT_LABELS = {"fight"}


@dataclass
class ModelConfig:
    name: str
    enabled: bool
    model_path: str
    labels: set[str]
    color: tuple[int, int, int]


@dataclass
class AppConfig:
    confidence: float
    alert_dir: Path
    alert_cooldown: float
    imgsz: int
    process_every: int
    stream_width: int
    stream_height: int
    database_url: str | None
    sound_enabled: bool
    models: list[ModelConfig]


@dataclass
class Detection:
    category: str
    label: str
    confidence: float
    color: tuple[int, int, int]
    box: tuple[int, int, int, int]


@dataclass
class DetectionRecord:
    id: int
    created_at: datetime
    categories: list[str]
    labels: list[str]
    confidence: float
    image_path: str
    status: str
    notes: str
    image_data: bytes | None = None


@dataclass
class SharedFrame:
    frame: object | None = None
    frame_id: int = 0
    error: str | None = None
    stopped: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class DetectionState:
    detections: list[Detection] = field(default_factory=list)
    source_frame_id: int = -1
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class HistoryUiState:
    visible: bool = False
    selected_index: int = 0
    records: list[DetectionRecord] = field(default_factory=list)
    last_message: str = "Press h to open detection history."


def normalize_base_url(address: str) -> str:
    """Return a clean base URL from an IP, host:port, or full URL."""
    address = address.strip()
    if not address:
        raise ValueError("address cannot be empty")

    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"

    parsed = urlparse(address)
    if not parsed.netloc:
        raise ValueError(f"invalid address: {address}")

    return address.rstrip("/")


def check_connection(base_url: str, timeout: float = 4.0) -> None:
    """Check that the phone responds before opening the video stream."""
    snapshot_url = f"{base_url}/shot.jpg"
    request = urllib.request.Request(snapshot_url, headers={"User-Agent": "Python IP Webcam Viewer"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ConnectionError(f"camera returned HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"could not reach {snapshot_url}. Make sure the phone and computer "
            "are on the same Wi-Fi network and the IP Webcam server is running."
        ) from exc


def save_frame(cv2, frame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"ip-webcam-{timestamp}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise OSError(f"failed to save snapshot to {path}")

    return path


class DetectionHistoryRepository:
    def __init__(self, database_url: str | None):
        self.database_url = database_url
        self.connection = None

        if not database_url:
            print(
                "PostgreSQL history is disabled. Set DETECTION_DATABASE_URL or pass "
                "--database-url to store alert images in PostgreSQL."
            )
            return

        try:
            import psycopg
        except ModuleNotFoundError:
            print(
                "PostgreSQL history is disabled because psycopg is not installed. "
                "Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return

        try:
            self.connection = psycopg.connect(database_url)
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_images (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    categories TEXT[] NOT NULL,
                    labels TEXT[] NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    image_path TEXT NOT NULL,
                    image_data BYTEA NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL DEFAULT 'new',
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self.connection.commit()
            print("PostgreSQL detection history is enabled.")
        except Exception as exc:
            self.connection = None
            print(f"PostgreSQL history is disabled: {exc}", file=sys.stderr)

    @property
    def enabled(self) -> bool:
        return self.connection is not None

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def create_alert(self, image_path: Path, detections: list[Detection]) -> int | None:
        if not self.enabled:
            return None

        categories = sorted({detection.category for detection in detections})
        labels = [detection.label for detection in detections]
        confidence = max((detection.confidence for detection in detections), default=0.0)
        metadata = {
            "detections": [
                {
                    "category": detection.category,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "box": detection.box,
                }
                for detection in detections
            ]
        }

        try:
            from psycopg.types.json import Jsonb

            image_data = image_path.read_bytes()
            row = self.connection.execute(
                """
                INSERT INTO detection_images
                    (categories, labels, confidence, image_path, image_data, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (categories, labels, confidence, str(image_path), image_data, Jsonb(metadata)),
            ).fetchone()
            self.connection.commit()
            return int(row[0]) if row else None
        except Exception as exc:
            self.connection.rollback()
            print(f"Could not store alert in PostgreSQL: {exc}", file=sys.stderr)
            return None

    def list_records(self, limit: int = 12) -> list[DetectionRecord]:
        if not self.enabled:
            return []

        try:
            rows = self.connection.execute(
                """
                SELECT id, created_at, categories, labels, confidence, image_path, status, notes
                FROM detection_images
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            return [
                DetectionRecord(
                    id=int(row[0]),
                    created_at=row[1],
                    categories=list(row[2]),
                    labels=list(row[3]),
                    confidence=float(row[4]),
                    image_path=row[5],
                    status=row[6],
                    notes=row[7],
                )
                for row in rows
            ]
        except Exception as exc:
            self.connection.rollback()
            print(f"Could not read detection history: {exc}", file=sys.stderr)
            return []

    def get_record(self, record_id: int) -> DetectionRecord | None:
        if not self.enabled:
            return None

        try:
            row = self.connection.execute(
                """
                SELECT id, created_at, categories, labels, confidence, image_path,
                       status, notes, image_data
                FROM detection_images
                WHERE id = %s
                """,
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            return DetectionRecord(
                id=int(row[0]),
                created_at=row[1],
                categories=list(row[2]),
                labels=list(row[3]),
                confidence=float(row[4]),
                image_path=row[5],
                status=row[6],
                notes=row[7],
                image_data=bytes(row[8]),
            )
        except Exception as exc:
            self.connection.rollback()
            print(f"Could not load detection image: {exc}", file=sys.stderr)
            return None

    def update_record(self, record_id: int, status: str, notes: str) -> bool:
        if not self.enabled:
            return False

        try:
            self.connection.execute(
                """
                UPDATE detection_images
                SET status = %s, notes = %s
                WHERE id = %s
                """,
                (status, notes, record_id),
            )
            self.connection.commit()
            return True
        except Exception as exc:
            self.connection.rollback()
            print(f"Could not update detection record: {exc}", file=sys.stderr)
            return False

    def delete_record(self, record_id: int) -> bool:
        if not self.enabled:
            return False

        try:
            self.connection.execute("DELETE FROM detection_images WHERE id = %s", (record_id,))
            self.connection.commit()
            return True
        except Exception as exc:
            self.connection.rollback()
            print(f"Could not delete detection record: {exc}", file=sys.stderr)
            return False


def parse_labels(labels: str) -> set[str]:
    return {label.strip().lower() for label in labels.split(",") if label.strip()}


def load_detector(config: ModelConfig):
    if not config.enabled:
        return None

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        print(
            "Missing dependency: ultralytics. Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return None

    if not Path(config.model_path).exists() and config.model_path.endswith(".pt"):
        print(
            f"Warning: model file was not found: {config.model_path}",
            file=sys.stderr,
        )

    print(f"Loading {config.name} model: {config.model_path}")
    detector = YOLO(config.model_path)
    model_labels = {str(name).lower() for name in detector.names.values()}
    missing_labels = sorted(config.labels - model_labels)

    if missing_labels:
        print(
            f"Warning: the {config.name} model does not contain these requested labels: "
            f"{', '.join(missing_labels)}"
        )

    return detector


def load_detectors(config: AppConfig) -> list[tuple[ModelConfig, object]]:
    detectors = []
    for model_config in config.models:
        detector = load_detector(model_config)
        if detector is not None:
            detectors.append((model_config, detector))

    return detectors


def draw_model_detections(
    cv2,
    frame,
    model_config: ModelConfig,
    detector,
    confidence: float,
    imgsz: int,
) -> list[Detection]:
    if detector is None:
        return []

    detections = []
    source_height, source_width = frame.shape[:2]
    scale = min(imgsz / source_width, imgsz / source_height, 1.0)
    inference_frame = frame

    if scale < 1.0:
        inference_width = int(source_width * scale)
        inference_height = int(source_height * scale)
        inference_frame = cv2.resize(frame, (inference_width, inference_height))

    results = detector.predict(inference_frame, conf=confidence, imgsz=imgsz, verbose=False)
    scale_back = 1 / scale

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = str(names[class_id])
            label_key = label.lower()

            if label_key not in model_config.labels:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(value * scale_back) for value in box.xyxy[0])
            detections.append(
                Detection(
                    category=model_config.name,
                    label=label,
                    confidence=confidence,
                    color=model_config.color,
                    box=(x1, y1, x2, y2),
                )
            )

    return detections


def draw_detection_boxes(cv2, frame, detections: list[Detection]) -> None:
    height, width = frame.shape[:2]

    for detection in sorted(detections, key=lambda item: item.category == "weapon"):
        x1, y1, x2, y2 = detection.box
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 0, 255) if detection.category == "weapon" else detection.color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)


def draw_premium_panel(cv2, frame, detections: list[Detection], history_state: HistoryUiState) -> None:
    height, width = frame.shape[:2]
    panel_width = min(340, max(250, width // 3))
    x1 = width - panel_width - 14
    y1 = 14
    x2 = width - 14
    y2 = min(height - 14, 190)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 24, 38), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (86, 102, 133), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y1 + 4), (59, 130, 246), -1)

    cv2.putText(frame, "AI SAFETY COMMAND", (x1 + 14, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 241, 245), 2, cv2.LINE_AA)
    status = "SUSPICIOUS" if detections else "NORMAL"
    status_color = (46, 204, 113) if not detections else (0, 80, 255)
    cv2.putText(frame, status, (x1 + 14, y1 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)

    detail = "No active alerts"
    if detections:
        labels = ", ".join(sorted({detection.label for detection in detections}))
        detail = f"{len(detections)} hit(s): {labels[:28]}"
    cv2.putText(frame, detail, (x1 + 14, y1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

    controls = "s Snapshot  h History  m Sound  q Quit"
    cv2.putText(frame, controls, (x1 + 14, y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (148, 163, 184), 1, cv2.LINE_AA)
    message = history_state.last_message[:42]
    cv2.putText(frame, message, (x1 + 14, y1 + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (125, 211, 252), 1, cv2.LINE_AA)


def draw_status(cv2, frame, detections: list[Detection]) -> None:
    if detections:
        categories = []
        if any(detection.category == "weapon" for detection in detections):
            categories.append("Weapon")
        if any(detection.category == "fight" for detection in detections):
            categories.append("Fight")

        detected_type = "Both" if len(categories) > 1 else categories[0]
        status = f"Suspicious Found: {detected_type}"
        color = (0, 0, 255)
    else:
        status = "Normal"
        color = (0, 180, 0)

    cv2.rectangle(frame, (10, 10), (min(10 + len(status) * 18, frame.shape[1] - 10), 58), (0, 0, 0), -1)
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


def play_notification_sound(enabled: bool) -> None:
    if not enabled:
        return

    try:
        if sys.platform.startswith("win"):
            import winsound

            winsound.Beep(1200, 170)
            winsound.Beep(900, 170)
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def handle_alert(
    cv2,
    frame,
    detections: list[Detection],
    config: AppConfig,
    history_repository: DetectionHistoryRepository,
) -> Path:
    print("\aALERT: Suspicious Found")
    play_notification_sound(config.sound_enabled)
    return save_frame(cv2, frame, config.alert_dir)


def refresh_history(history_state: HistoryUiState, repository: DetectionHistoryRepository) -> None:
    history_state.records = repository.list_records()
    if history_state.records:
        history_state.selected_index = max(0, min(history_state.selected_index, len(history_state.records) - 1))
        history_state.last_message = f"Loaded {len(history_state.records)} detection record(s)."
    elif repository.enabled:
        history_state.selected_index = 0
        history_state.last_message = "No detection history yet."
    else:
        history_state.last_message = "PostgreSQL history is not connected."


def draw_history_window(cv2, history_state: HistoryUiState, repository: DetectionHistoryRepository) -> None:
    if not history_state.visible:
        try:
            cv2.destroyWindow(HISTORY_WINDOW_NAME)
        except cv2.error:
            pass
        return

    import numpy as np

    if not history_state.records:
        refresh_history(history_state, repository)

    canvas = np.zeros((650, 980, 3), dtype=np.uint8)
    canvas[:] = (15, 23, 42)
    cv2.rectangle(canvas, (0, 0), (980, 74), (24, 32, 48), -1)
    cv2.rectangle(canvas, (0, 73), (980, 76), (59, 130, 246), -1)
    cv2.putText(canvas, "Detection History", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (241, 245, 249), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CRUD: up/down select, v view image, u update, d delete, h close", (420, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    headers = ["ID", "TIME", "TYPE", "CONF", "STATUS", "NOTES"]
    positions = [30, 100, 270, 455, 555, 690]
    for header, x in zip(headers, positions):
        cv2.putText(canvas, header, (x, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (125, 211, 252), 1, cv2.LINE_AA)

    for index, record in enumerate(history_state.records[:12]):
        top = 135 + index * 38
        selected = index == history_state.selected_index
        row_color = (30, 41, 59) if not selected else (29, 78, 216)
        cv2.rectangle(canvas, (22, top - 24), (958, top + 8), row_color, -1)
        created = record.created_at.strftime("%m-%d %H:%M")
        category = ", ".join(record.categories)[:18]
        notes = record.notes[:28] if record.notes else "-"
        values = [
            str(record.id),
            created,
            category,
            f"{record.confidence:.2f}",
            record.status[:12],
            notes,
        ]
        for value, x in zip(values, positions):
            cv2.putText(canvas, value, (x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1, cv2.LINE_AA)

    cv2.putText(canvas, history_state.last_message[:90], (28, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (203, 213, 225), 1, cv2.LINE_AA)
    cv2.imshow(HISTORY_WINDOW_NAME, canvas)


def selected_history_record(history_state: HistoryUiState) -> DetectionRecord | None:
    if not history_state.records:
        return Noneq
    return history_state.records[max(0, min(history_state.selected_index, len(history_state.records) - 1))]


def update_selected_record(history_state: HistoryUiState, repository: DetectionHistoryRepository) -> None:
    record = selected_history_record(history_state)
    if record is None:
        history_state.last_message = "Select a record before updating."
        return

    print(f"\nUpdating detection record #{record.id}")
    status = input("Status [new/reviewed/escalated/false_positive]: ").strip() or record.status
    notes = input("Notes: ").strip() or record.notes
    if repository.update_record(record.id, status, notes):
        history_state.last_message = f"Updated record #{record.id}."
        refresh_history(history_state, repository)
    else:
        history_state.last_message = f"Could not update record #{record.id}."


def delete_selected_record(history_state: HistoryUiState, repository: DetectionHistoryRepository) -> None:
    record = selected_history_record(history_state)
    if record is None:
        history_state.last_message = "Select a record before deleting."
        return

    confirm = input(f"Delete detection record #{record.id}? Type DELETE to confirm: ").strip()
    if confirm != "DELETE":
        history_state.last_message = "Delete cancelled."
        return

    if repository.delete_record(record.id):
        history_state.last_message = f"Deleted record #{record.id}."
        history_state.selected_index = max(0, history_state.selected_index - 1)
        refresh_history(history_state, repository)
    else:
        history_state.last_message = f"Could not delete record #{record.id}."


def view_selected_record_image(cv2, history_state: HistoryUiState, repository: DetectionHistoryRepository) -> None:
    record = selected_history_record(history_state)
    if record is None:
        history_state.last_message = "Select a record before viewing."
        return

    stored = repository.get_record(record.id)
    if stored is None or stored.image_data is None:
        history_state.last_message = f"Could not load image for record #{record.id}."
        return

    import numpy as np

    image_array = np.frombuffer(stored.image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        history_state.last_message = f"Stored image for record #{record.id} is unreadable."
        return

    cv2.imshow(f"Detection Image #{record.id}", image)
    history_state.last_message = f"Viewing stored image #{record.id}."


def create_display_window(cv2) -> bool:
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        return True
    except cv2.error as exc:
        print(
            "OpenCV cannot open a popup window in this environment.\n\n"
            "Most likely, opencv-python-headless is installed. Fix it with:\n"
            "  source .venv/bin/activate\n"
            "  pip uninstall -y opencv-python-headless opencv-python\n"
            "  pip install opencv-python\n\n"
            "Then run this app again.",
            file=sys.stderr,
        )
        print(f"\nOriginal OpenCV error:\n{exc}", file=sys.stderr)
        return False


def capture_latest_frames(camera, shared_frame: SharedFrame) -> None:
    while True:
        with shared_frame.lock:
            if shared_frame.stopped:
                return

        ok, frame = camera.read()
        if not ok:
            with shared_frame.lock:
                shared_frame.error = "Lost connection or no frame received."
            time.sleep(0.02)
            continue

        with shared_frame.lock:
            shared_frame.frame = frame
            shared_frame.frame_id += 1
            shared_frame.error = None


def detect_latest_frames(
    cv2,
    detectors: list[tuple[ModelConfig, object]],
    shared_frame: SharedFrame,
    detection_state: DetectionState,
    confidence: float,
    imgsz: int,
    process_every: int,
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
        detections: list[Detection] = []

        with detection_state.lock:
            detection_state.running = True

        for model_config, detector in detectors:
            detections.extend(
                draw_model_detections(
                    cv2,
                    frame,
                    model_config,
                    detector,
                    confidence,
                    imgsz,
                )
            )

        with detection_state.lock:
            detection_state.detections = detections
            detection_state.source_frame_id = frame_id
            detection_state.running = False


def view_stream(
    base_url: str,
    output_dir: Path,
    skip_check: bool,
    app_config: AppConfig,
) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print(
            "Missing dependency: opencv-python. Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    detectors = load_detectors(app_config)
    history_repository = DetectionHistoryRepository(app_config.database_url)

    if not skip_check:
        check_connection(base_url)

    video_url = f"{base_url}/video"
    camera = cv2.VideoCapture(video_url)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if app_config.stream_width > 0:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, app_config.stream_width)
    if app_config.stream_height > 0:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, app_config.stream_height)

    if not camera.isOpened():
        print(f"Could not open video stream: {video_url}", file=sys.stderr)
        history_repository.close()
        return 1

    print("Connected.")
    print(
        "Controls: 's' snapshot, 'h' history CRUD, 'v' view selected image, "
        "'u' update selected, 'd' delete selected, 'm' sound, 'q' or Esc quit."
    )
    print(
        "Speed settings: "
        f"imgsz={app_config.imgsz}, process_every={app_config.process_every}, "
        f"stream={app_config.stream_width}x{app_config.stream_height}"
    )
    if detectors:
        for model_config, _detector in detectors:
            print(
                f"{model_config.name.title()} detection is enabled for: "
                f"{', '.join(sorted(model_config.labels))}"
            )
    else:
        print("No detection model is enabled. The popup will only show the camera.")
    print(f"Popup window: {WINDOW_NAME}")

    if not create_display_window(cv2):
        camera.release()
        return 1

    shared_frame = SharedFrame()
    detection_state = DetectionState()
    history_state = HistoryUiState()
    capture_thread = threading.Thread(
        target=capture_latest_frames,
        args=(camera, shared_frame),
        daemon=True,
    )
    detection_thread = threading.Thread(
        target=detect_latest_frames,
        args=(
            cv2,
            detectors,
            shared_frame,
            detection_state,
            app_config.confidence,
            app_config.imgsz,
            app_config.process_every,
        ),
        daemon=True,
    )
    capture_thread.start()
    if detectors:
        detection_thread.start()

    last_alert_time = 0.0

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
                detections = list(detection_state.detections)

            draw_detection_boxes(cv2, frame, detections)

            draw_status(cv2, frame, detections)
            draw_premium_panel(cv2, frame, detections, history_state)
            now = time.monotonic()
            if detections and now - last_alert_time >= app_config.alert_cooldown:
                path = handle_alert(cv2, frame, detections, app_config, history_repository)
                record_id = history_repository.create_alert(path, detections)
                if record_id is None:
                    print(f"Saved alert frame: {path}")
                    history_state.last_message = f"Saved alert locally: {path.name}"
                else:
                    print(f"Saved alert frame: {path} and PostgreSQL record #{record_id}")
                    history_state.last_message = f"Stored PostgreSQL alert #{record_id}"
                    if history_state.visible:
                        refresh_history(history_state, history_repository)
                last_alert_time = now

            cv2.imshow(WINDOW_NAME, frame)
            draw_history_window(cv2, history_state, history_repository)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                return 0

            if key == ord("s"):
                path = save_frame(cv2, frame, output_dir)
                print(f"Saved {path}")
                history_state.last_message = f"Saved snapshot: {path.name}"

            if key == ord("m"):
                app_config.sound_enabled = not app_config.sound_enabled
                state = "on" if app_config.sound_enabled else "off"
                history_state.last_message = f"Notification sound is {state}."
                print(f"Notification sound is {state}.")

            if key == ord("h"):
                history_state.visible = not history_state.visible
                if history_state.visible:
                    refresh_history(history_state, history_repository)
                else:
                    history_state.last_message = "Detection history closed."

            if history_state.visible and key in (82, 2490368):
                history_state.selected_index = max(0, history_state.selected_index - 1)

            if history_state.visible and key in (84, 2621440):
                if history_state.records:
                    history_state.selected_index = min(len(history_state.records) - 1, history_state.selected_index + 1)

            if history_state.visible and key == ord("r"):
                refresh_history(history_state, history_repository)

            if history_state.visible and key == ord("u"):
                update_selected_record(history_state, history_repository)

            if history_state.visible and key == ord("d"):
                delete_selected_record(history_state, history_repository)

            if history_state.visible and key == ord("v"):
                view_selected_record_image(cv2, history_state, history_repository)
    finally:
        with shared_frame.lock:
            shared_frame.stopped = True
        capture_thread.join(timeout=1.0)
        if detectors:
            detection_thread.join(timeout=1.0)
        camera.release()
        history_repository.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to an Android IP Webcam app stream over Wi-Fi."
    )
    parser.add_argument(
        "address",
        help="Phone camera address, for example 192.168.1.23:8080 or http://192.168.1.23:8080",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snapshots"),
        help="Folder where snapshots are saved when you press 's'.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Open the video stream without first checking /shot.jpg.",
    )
    parser.add_argument(
        "--detect-weapons",
        action="store_true",
        help="Draw alerts for possible weapons detected in the video stream.",
    )
    parser.add_argument(
        "--detect-fight",
        action="store_true",
        help="Draw alerts for possible fighting detected in the video stream.",
    )
    parser.add_argument(
        "--detect-all",
        action="store_true",
        help="Enable both weapon and fight detection. This is now the default when no detector flag is passed.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--weapon-model",
        default=DEFAULT_WEAPON_MODEL_PATH,
        help="YOLO weapon model file to use.",
    )
    parser.add_argument(
        "--fight-model",
        default=DEFAULT_FIGHT_MODEL_PATH,
        help="YOLO fight model file to use.",
    )
    parser.add_argument(
        "--weapon-labels",
        default=",".join(sorted(DEFAULT_WEAPON_LABELS)),
        help="Comma-separated model labels that should trigger weapon alerts.",
    )
    parser.add_argument(
        "--fight-labels",
        default=",".join(sorted(DEFAULT_FIGHT_LABELS)),
        help="Comma-separated model labels that should trigger fight alerts.",
    )
    parser.add_argument(
        "--alert-dir",
        type=Path,
        default=Path("alerts"),
        help="Folder where alert frames are saved.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=3.0,
        help="Seconds to wait between printed/saved alerts.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DETECTION_DATABASE_URL"),
        help=(
            "PostgreSQL connection URL used to store detection history images. "
            "Can also be set with DETECTION_DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable notification sound when suspicious action is detected.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="YOLO inference image size. Smaller is faster; 320 or 416 is good for low lag.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=2,
        help="Run detection every N frames. Higher is faster but less precise.",
    )
    parser.add_argument(
        "--stream-width",
        type=int,
        default=640,
        help="Requested stream width. Use 0 to leave unchanged.",
    )
    parser.add_argument(
        "--stream-height",
        type=int,
        default=480,
        help="Requested stream height. Use 0 to leave unchanged.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        base_url = normalize_base_url(args.address)
        detect_all_by_default = not (args.detect_weapons or args.detect_fight or args.detect_all)
        detect_weapons = args.detect_weapons or args.detect_all or detect_all_by_default
        detect_fight = args.detect_fight or args.detect_all or detect_all_by_default
        app_config = AppConfig(
            confidence=args.confidence,
            alert_dir=args.alert_dir,
            alert_cooldown=args.alert_cooldown,
            imgsz=args.imgsz,
            process_every=max(1, args.process_every),
            stream_width=args.stream_width,
            stream_height=args.stream_height,
            database_url=args.database_url,
            sound_enabled=not args.no_sound,
            models=[
                ModelConfig(
                    name="weapon",
                    enabled=detect_weapons,
                    model_path=args.weapon_model,
                    labels=parse_labels(args.weapon_labels),
                    color=(0, 0, 255),
                ),
                ModelConfig(
                    name="fight",
                    enabled=detect_fight,
                    model_path=args.fight_model,
                    labels=parse_labels(args.fight_labels),
                    color=(0, 165, 255),
                ),
            ],
        )
        return view_stream(base_url, args.output_dir, args.skip_check, app_config)
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
