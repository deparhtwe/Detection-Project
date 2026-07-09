#!/usr/bin/env python3
"""AEGIS threat intelligence dashboard for Android IP Webcam.

This single-file app keeps the UI, camera stream, detection logic, and
history actions together in one Python module.
"""

from __future__ import annotations

import argparse
import base64
import json
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
import tkinter as tk
from tkinter import messagebox, simpledialog

DEFAULT_WEAPON_MODEL_PATH = "runs/detect/train-1/weights/best.pt"
DEFAULT_FIGHT_MODEL_PATH = "runs/detect/train/weights/best.pt"

DEFAULT_WEAPON_LABELS = {"weapon"}
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
    location: str = ""
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


def normalize_base_url(address: str) -> str:
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
    snapshot_url = f"{base_url}/shot.jpg"
    request = urllib.request.Request(snapshot_url, headers={"User-Agent": "Python IP Webcam Viewer"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ConnectionError(f"camera returned HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"could not reach {snapshot_url}. Make sure the phone and computer are on the same Wi-Fi network."
        ) from exc


def save_frame(cv2, frame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ip-webcam-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"failed to save snapshot to {path}")
    return path


def metadata_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now()


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def fallback_records() -> list[DetectionRecord]:
    return [
        DetectionRecord(
            id=54713,
            created_at=datetime(2026, 7, 9, 14, 9, 14),
            categories=["weapon"],
            labels=["Weapon"],
            confidence=67.0,
            image_path="",
            status="pending",
            notes="Weapon-shaped object detected near the desk.",
            location="Parking Lot B",
        ),
        DetectionRecord(
            id=51035,
            created_at=datetime(2026, 7, 9, 14, 9, 11),
            categories=["weapon"],
            labels=["Weapon"],
            confidence=51.0,
            image_path="",
            status="false_positive",
            notes="Object matched a non-threatening silhouette.",
            location="Parking Lot B",
        ),
        DetectionRecord(
            id=47961,
            created_at=datetime(2026, 7, 9, 14, 9, 7),
            categories=["weapon"],
            labels=["Weapon"],
            confidence=84.0,
            image_path="",
            status="pending",
            notes="Sharp object motion in the live frame.",
            location="Parking Lot B",
        ),
    ]


def record_display_title(record: DetectionRecord) -> str:
    return record.labels[0] if record.labels else "Unknown"


def record_status_meta(status: str) -> tuple[str, str]:
    normalized = status.lower().strip()
    if normalized in {"verified", "confirmed"}:
        return "verified", "VERIFIED"
    if normalized in {"false", "false_positive", "false positive"}:
        return "false_positive", "FALSE POSITIVE"
    return "pending", "PENDING"


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
        print(f"Warning: model file was not found: {config.model_path}", file=sys.stderr)

    print(f"Loading {config.name} model: {config.model_path}")
    detector = YOLO(config.model_path)
    model_labels = {str(name).lower() for name in detector.names.values()}
    missing_labels = sorted(config.labels - model_labels)
    if missing_labels:
        print(
            f"Warning: the {config.name} model does not contain these requested labels: {', '.join(missing_labels)}",
            file=sys.stderr,
        )
    return detector


def load_detectors(config: AppConfig) -> list[tuple[ModelConfig, object]]:
    detectors = []
    for model_config in config.models:
        detector = load_detector(model_config)
        if detector is not None:
            detectors.append((model_config, detector))
    return detectors


def draw_model_detections(cv2, frame, model_config: ModelConfig, detector, confidence: float, imgsz: int) -> list[Detection]:
    if detector is None:
        return []

    detections: list[Detection] = []
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
            if label.lower() not in model_config.labels:
                continue

            x1, y1, x2, y2 = (int(value * scale_back) for value in box.xyxy[0])
            detections.append(
                Detection(
                    category=model_config.name,
                    label=label,
                    confidence=float(box.conf[0]),
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


def make_alert_placeholder(cv2, width: int = 960, height: int = 540):
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (8, 14, 28)
    cv2.rectangle(frame, (16, 16), (width - 16, height - 16), (34, 48, 72), 2)
    cv2.putText(
        frame,
        "No camera frame available",
        (40, height // 2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (220, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Displaying fallback incident image",
        (40, height // 2 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (140, 160, 190),
        1,
        cv2.LINE_AA,
    )
    return frame


def frame_to_photoimage(cv2, frame, max_width: int, max_height: int, master: tk.Misc) -> tk.PhotoImage | None:
    if frame is None:
        return None

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(max_width / width, max_height / height)
    if scale <= 0:
        return None

    if scale != 1.0:
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        frame = cv2.resize(frame, (new_width, new_height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, buffer = cv2.imencode(".png", rgb)
    if not ok:
        return None

    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return tk.PhotoImage(master=master, data=payload, format="png")


class DetectionHistoryRepository:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def _metadata_paths(self) -> list[Path]:
        return sorted(self.storage_dir.glob("*.json"))

    def _load_metadata(self, metadata_path: Path) -> dict | None:
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not read history metadata {metadata_path.name}: {exc}", file=sys.stderr)
            return None

    def _record_from_metadata(self, metadata: dict) -> DetectionRecord | None:
        try:
            image_path = Path(metadata.get("image_path", ""))
            if not image_path.is_absolute():
                image_path = self.storage_dir / image_path.name
            if not image_path.exists():
                image_path = self.storage_dir / f"{metadata.get('id')}.jpg"
            record = DetectionRecord(
                id=int(metadata["id"]),
                created_at=parse_iso_datetime(metadata.get("created_at")),
                categories=list(metadata.get("categories", [])),
                labels=list(metadata.get("labels", [])),
                confidence=float(metadata.get("confidence", 0.0)),
                image_path=str(image_path),
                status=str(metadata.get("status", "new")),
                notes=str(metadata.get("notes", "")),
                location=str(metadata.get("location", "Parking Lot B")),
            )
            return record
        except Exception as exc:
            print(f"Could not parse history record: {exc}", file=sys.stderr)
            return None

    def _metadata_for_record(self, record: DetectionRecord, image_path: str | None = None) -> dict:
        return {
            "id": record.id,
            "created_at": record.created_at.isoformat(timespec="seconds"),
            "categories": record.categories,
            "labels": record.labels,
            "confidence": record.confidence,
            "image_path": image_path or record.image_path,
            "status": record.status,
            "notes": record.notes,
            "location": record.location,
        }

    def _save_metadata(self, image_path: Path, metadata: dict) -> None:
        metadata_path = metadata_path_for_image(image_path)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _existing_ids(self) -> set[int]:
        ids: set[int] = set()
        for metadata_path in self._metadata_paths():
            metadata = self._load_metadata(metadata_path)
            if not metadata:
                continue
            try:
                ids.add(int(metadata["id"]))
            except Exception:
                continue
        return ids

    def _next_id(self) -> int:
        existing = self._existing_ids()
        candidate = int(time.time() * 1000)
        while candidate in existing:
            candidate += 1
        return candidate

    def create_alert(self, image_path: Path, detections: list[Detection]) -> int | None:
        try:
            record_id = self._next_id()
            categories = sorted({detection.category for detection in detections})
            labels = [detection.label for detection in detections]
            confidence = max((detection.confidence for detection in detections), default=0.0)
            if confidence <= 1.0:
                confidence *= 100.0
            inferred_location = (
                "East Corridor"
                if any(detection.category == "fight" for detection in detections)
                else "Parking Lot B"
                if any(detection.category == "weapon" for detection in detections)
                else "Live Camera Feed"
            )
            status = "verified" if confidence >= 85 else "pending" if confidence >= 60 else "false_positive"
            record = DetectionRecord(
                id=record_id,
                created_at=datetime.now(),
                categories=categories,
                labels=labels,
                confidence=confidence,
                image_path=str(image_path),
                status=status,
                notes=", ".join(labels),
                location=inferred_location,
            )
            metadata = self._metadata_for_record(record, image_path=str(image_path))
            metadata["detections"] = [
                {
                    "category": detection.category,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "box": list(detection.box),
                }
                for detection in detections
            ]
            metadata["image_file"] = image_path.name
            self._save_metadata(image_path, metadata)
            return record_id
        except Exception as exc:
            print(f"Could not store alert in image_data: {exc}", file=sys.stderr)
            return None

    def list_records(self, limit: int | None = None) -> list[DetectionRecord]:
        records: list[DetectionRecord] = []
        for metadata_path in self._metadata_paths():
            metadata = self._load_metadata(metadata_path)
            if not metadata:
                continue
            record = self._record_from_metadata(metadata)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.created_at, reverse=True)
        if limit is None or limit <= 0:
            return records
        return records[:limit]

    def get_record(self, record_id: int) -> DetectionRecord | None:
        for metadata_path in self._metadata_paths():
            metadata = self._load_metadata(metadata_path)
            if not metadata:
                continue
            try:
                if int(metadata.get("id")) != record_id:
                    continue
            except Exception:
                continue
            return self._record_from_metadata(metadata)
        return None

    def update_record(self, record_id: int, status: str, notes: str) -> bool:
        for metadata_path in self._metadata_paths():
            metadata = self._load_metadata(metadata_path)
            if not metadata:
                continue
            try:
                if int(metadata.get("id")) != record_id:
                    continue
            except Exception:
                continue
            metadata["status"] = status
            metadata["notes"] = notes
            metadata["updated_at"] = iso_now()
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            return True
        return False

    def delete_record(self, record_id: int) -> bool:
        for metadata_path in self._metadata_paths():
            metadata = self._load_metadata(metadata_path)
            if not metadata:
                continue
            try:
                if int(metadata.get("id")) != record_id:
                    continue
            except Exception:
                continue
            image_path = Path(str(metadata.get("image_path", "")))
            if not image_path.is_absolute():
                image_path = self.storage_dir / image_path.name
            try:
                if image_path.exists():
                    image_path.unlink()
            except Exception as exc:
                print(f"Could not delete image file {image_path.name}: {exc}", file=sys.stderr)
            try:
                metadata_path.unlink()
            except Exception as exc:
                print(f"Could not delete metadata file {metadata_path.name}: {exc}", file=sys.stderr)
            return True
        return False


class StreamController:
    def __init__(
        self,
        cv2,
        base_url: str,
        app_config: AppConfig,
        detectors: list[tuple[ModelConfig, object]],
    ) -> None:
        self.cv2 = cv2
        self.base_url = base_url
        self.app_config = app_config
        self.detectors = detectors
        self.camera = None
        self.shared_frame = SharedFrame()
        self.detection_state = DetectionState()
        self.capture_thread: threading.Thread | None = None
        self.detection_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.running = False
        self.last_error: str | None = None

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True

            video_url = f"{self.base_url}/video"
            camera = self.cv2.VideoCapture(video_url)
            camera.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.app_config.stream_width > 0:
                camera.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.app_config.stream_width)
            if self.app_config.stream_height > 0:
                camera.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.app_config.stream_height)

            if not camera.isOpened():
                camera.release()
                self.last_error = f"Could not open video stream: {video_url}"
                return False

            self.shared_frame = SharedFrame()
            self.detection_state = DetectionState()
            self.camera = camera
            self.capture_thread = threading.Thread(
                target=capture_latest_frames,
                args=(camera, self.shared_frame),
                daemon=True,
            )
            self.capture_thread.start()

            if self.detectors:
                self.detection_thread = threading.Thread(
                    target=detect_latest_frames,
                    args=(
                        self.cv2,
                        self.detectors,
                        self.shared_frame,
                        self.detection_state,
                        self.app_config,
                    ),
                    daemon=True,
                )
                self.detection_thread.start()
            else:
                self.detection_thread = None

            self.running = True
            self.last_error = None
            return True

    def stop(self) -> None:
        with self.lock:
            if not self.running:
                return

            with self.shared_frame.lock:
                self.shared_frame.stopped = True

            if self.capture_thread is not None:
                self.capture_thread.join(timeout=1.0)
            if self.detection_thread is not None:
                self.detection_thread.join(timeout=1.0)
            if self.camera is not None:
                self.camera.release()
                self.camera = None

            with self.shared_frame.lock:
                self.shared_frame.frame = None
                self.shared_frame.error = None

            with self.detection_state.lock:
                self.detection_state.detections = []
                self.detection_state.source_frame_id = -1
                self.detection_state.running = False

            self.running = False

    def close(self) -> None:
        self.stop()

    def get_frame(self):
        with self.shared_frame.lock:
            if self.shared_frame.frame is None:
                return None, self.shared_frame.frame_id, self.shared_frame.error
            return self.shared_frame.frame.copy(), self.shared_frame.frame_id, self.shared_frame.error

    def get_detections(self) -> list[Detection]:
        with self.detection_state.lock:
            return list(self.detection_state.detections)


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
    app_config: AppConfig,
) -> None:
    last_processed_id = -1

    while True:
        with shared_frame.lock:
            if shared_frame.stopped:
                return
            frame = None if shared_frame.frame is None else shared_frame.frame.copy()
            frame_id = shared_frame.frame_id

        if frame is None or frame_id == last_processed_id or frame_id % max(1, app_config.process_every) != 0:
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
                    app_config.confidence,
                    app_config.imgsz,
                )
            )

        with detection_state.lock:
            detection_state.detections = detections
            detection_state.source_frame_id = frame_id
            detection_state.running = False


def handle_alert(
    cv2,
    frame,
    detections: list[Detection],
    config: AppConfig,
    history_repository: DetectionHistoryRepository,
) -> Path:
    play_notification_sound(config.sound_enabled)
    return save_frame(cv2, frame, config.alert_dir)


class DashboardApp:
    BG = "#040817"
    PANEL = "#0a1224"
    PANEL_2 = "#0d172b"
    BORDER = "#1d2a44"
    TEXT = "#f4f7ff"
    MUTED = "#8da0c5"
    CYAN = "#49e3ff"
    PINK = "#ff4d7d"
    RED = "#ff5a76"
    GREEN = "#14d38f"
    AMBER = "#f9a825"

    def __init__(
        self,
        cv2,
        app_config: AppConfig,
        output_dir: Path,
        history_repository: DetectionHistoryRepository,
        stream_controller: StreamController,
    ) -> None:
        self.cv2 = cv2
        self.app_config = app_config
        self.output_dir = output_dir
        self.history_repository = history_repository
        self.stream_controller = stream_controller
        self.root = tk.Tk()
        self.root.title("Weapon and Violence Detection System")
        self.root.configure(bg=self.BG)
        self.root.geometry("1850x980")
        self.root.minsize(1240, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.closed = False
        self.detection_active = True
        self.confidence_value = max(1, min(100, int(round(app_config.confidence * 100))))
        self.last_alert_time = 0.0
        self.last_alert_frame_id = -1
        self.last_frame_id = -1
        self.last_detection_signature: tuple[str, ...] | None = None
        self.photo_image: tk.PhotoImage | None = None
        self.incidents: list[DetectionRecord] = []
        self.selected_record_id: int | None = None

        self.capture_time_var = tk.StringVar(value="2026-07-09 14:09:14")
        self.operator_notes_var = tk.StringVar(value="Weapon")
        self.timeline_var = tk.StringVar(value=f"{self.confidence_value}%")
        self.tracking_var = tk.StringVar(value="Active Tracking")
        self.footer_var = tk.StringVar(value="Ready for live detection events.")
        self.toggle_label_var = tk.StringVar(value="Stop Detection")
        self.pipeline_title_var = tk.StringVar(value="Pipeline Processing")
        self.pipeline_desc_var = tk.StringVar(value="Evaluating multi-spectral raw frames...")

        self.build_ui()
        self.load_initial_incidents()
        self.refresh_history()
        self.sync_toggle_button_text()
        self.render_stage(force=True)
        self.root.after(60, self.poll)

    def build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=24, pady=24)

        topbar = tk.Frame(outer, bg=self.BG)
        topbar.pack(fill="x", pady=(0, 18))

        brand = tk.Frame(topbar, bg=self.BG)
        brand.pack(side="left", anchor="w")
        brand_mark = tk.Canvas(brand, width=6, height=48, bg=self.BG, highlightthickness=0, bd=0)
        brand_mark.create_rectangle(1, 2, 5, 46, fill="#7e143a", outline="#7e143a")
        brand_mark.pack(side="left", padx=(0, 14))
        brand_copy = tk.Frame(brand, bg=self.BG)
        brand_copy.pack(side="left")
        tk.Label(
            brand_copy,
            text="Weapon and Violence Detection System",
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand_copy,
            text="",
            bg=self.BG,
            fg=self.CYAN,
            font=("Courier New", 10, "bold"),
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(
            topbar,
            bg="#09101f",
            highlightthickness=1,
            highlightbackground=self.BORDER,
            width=430,
            height=58,
        )
        controls.pack(side="right", anchor="e")
        controls.pack_propagate(False)

        confidence_frame = tk.Frame(controls, bg="#09101f")
        confidence_frame.pack(side="left", padx=16, pady=10)
        tk.Label(
            confidence_frame,
            text="CHOOSE CONFIDENCE LEVEL",
            bg="#09101f",
            fg=self.MUTED,
            font=("Courier New", 8, "bold"),
        ).pack(anchor="w")
        slider_row = tk.Frame(confidence_frame, bg="#09101f")
        slider_row.pack(anchor="w", pady=(2, 0))
        self.confidence_scale = tk.Scale(
            slider_row,
            from_=0,
            to=100,
            orient="horizontal",
            length=150,
            showvalue=False,
            troughcolor="#24314c",
            bg="#09101f",
            fg=self.TEXT,
            highlightthickness=0,
            activebackground=self.PINK,
            command=self.on_confidence_change,
        )
        self.confidence_scale.set(self.confidence_value)
        self.confidence_scale.pack(side="left")
        self.confidence_value_label = tk.Label(
            slider_row,
            text=self.timeline_var.get(),
            bg="#09101f",
            fg=self.RED,
            font=("Segoe UI", 10, "bold"),
        )
        self.confidence_value_label.pack(side="left", padx=(8, 0))

        tk.Frame(controls, bg=self.BORDER, width=1).pack(side="left", fill="y", padx=(4, 0), pady=8)
        self.toggle_button = tk.Button(
            controls,
            textvariable=self.toggle_label_var,
            command=self.toggle_detection,
            bg=self.PINK,
            fg="white",
            activebackground="#ff6a90",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
        )
        self.toggle_button.pack(side="left", padx=12, pady=8)

        content = tk.Frame(outer, bg=self.BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1, uniform="cols")
        content.columnconfigure(1, weight=1, uniform="cols")
        content.rowconfigure(0, weight=1)

        left_panel = tk.Frame(content, bg=self.PANEL, highlightthickness=1, highlightbackground=self.BORDER)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)

        left_head = tk.Frame(left_panel, bg=self.PANEL)
        left_head.grid(row=0, column=0, sticky="ew")
        tk.Label(
            left_head,
            text="LIVE STREAM FEED",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Courier New", 10, "bold"),
        ).pack(side="left", padx=16, pady=14)
        tk.Label(
            left_head,
            textvariable=self.tracking_var,
            bg=self.PANEL,
            fg=self.CYAN,
            font=("Courier New", 8, "bold"),
        ).pack(side="right", padx=16)

        stage_wrap = tk.Frame(left_panel, bg=self.PANEL, padx=16, pady=0)
        stage_wrap.grid(row=1, column=0, sticky="nsew")
        stage_wrap.rowconfigure(0, weight=1)
        stage_wrap.columnconfigure(0, weight=1)

        self.stage_canvas = tk.Canvas(stage_wrap, bg="#040812", highlightthickness=1, highlightbackground=self.BORDER)
        self.stage_canvas.grid(row=0, column=0, sticky="nsew")
        self.status_badge = tk.Label(
            self.stage_canvas,
            text="PIPELINE PROCESSING\nEvaluating multi-spectral raw frames...",
            justify="left",
            bg="#5b1125",
            fg="#ffd5df",
            font=("Courier New", 9, "bold"),
            padx=10,
            pady=8,
            relief="flat",
        )
        self.badge_window = self.stage_canvas.create_window(18, 18, anchor="nw", window=self.status_badge)
        self.stage_note = tk.Label(
            self.stage_canvas,
            text="Continuous camera overlay",
            bg="#040812",
            fg=self.MUTED,
            font=("Courier New", 8, "bold"),
        )
        self.note_window = self.stage_canvas.create_window(760, 18, anchor="ne", window=self.stage_note)

        timeline = tk.Frame(left_panel, bg=self.PANEL, padx=16, pady=14)
        timeline.grid(row=2, column=0, sticky="ew")
        timeline.columnconfigure(0, weight=1)
        self.timeline_track = tk.Canvas(timeline, height=4, bg=self.PANEL, highlightthickness=0)
        self.timeline_track.grid(row=0, column=0, sticky="ew")
        self.timeline_fill = self.timeline_track.create_rectangle(0, 0, 0, 4, fill=self.PINK, outline=self.PINK)
        tk.Label(
            timeline,
            textvariable=self.timeline_var,
            bg=self.PANEL,
            fg=self.RED,
            font=("Courier New", 9, "bold"),
        ).place(relx=1.0, rely=0.5, x=-4, y=-10, anchor="e")

        footer = tk.Frame(left_panel, bg=self.PANEL_2, highlightthickness=1, highlightbackground=self.BORDER)
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure((0, 1, 2), weight=1)
        self.footer_clip = self.make_meta_item(footer, 0, "CLIP LENGTH", "5.00 Seconds Loop", self.TEXT)
        self.footer_capture = self.make_meta_item(footer, 1, "CAPTURE TIME", self.capture_time_var.get(), "#ff7c94")
        self.footer_notes = self.make_meta_item(footer, 2, "OPERATOR NOTES", self.operator_notes_var.get(), self.TEXT)

        right_panel = tk.Frame(content, bg=self.PANEL, highlightthickness=1, highlightbackground=self.BORDER)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        right_head = tk.Frame(right_panel, bg=self.PANEL)
        right_head.grid(row=0, column=0, sticky="ew")
        tk.Label(
            right_head,
            text="DETECTION INCIDENT HISTORY",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left", padx=16, pady=14)

        history_wrap = tk.Frame(right_panel, bg=self.PANEL, padx=12, pady=12)
        history_wrap.grid(row=1, column=0, sticky="nsew")
        history_wrap.rowconfigure(0, weight=1)
        history_wrap.columnconfigure(0, weight=1)

        self.history_canvas = tk.Canvas(history_wrap, bg=self.PANEL, highlightthickness=0)
        self.history_scrollbar = tk.Scrollbar(history_wrap, orient="vertical", command=self.history_canvas.yview)
        self.history_canvas.configure(yscrollcommand=self.history_scrollbar.set)
        self.history_scrollbar.grid(row=0, column=1, sticky="ns")
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_inner = tk.Frame(self.history_canvas, bg=self.PANEL)
        self.history_inner_id = self.history_canvas.create_window((0, 0), window=self.history_inner, anchor="nw")
        self.history_inner.bind("<Configure>", lambda _event: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")))
        self.history_canvas.bind("<Configure>", self.on_history_resize)

        header_row = tk.Frame(self.history_inner, bg="#09101f", highlightthickness=1, highlightbackground=self.BORDER)
        header_row.pack(fill="x", padx=4, pady=(4, 8))
        header_specs = [
            ("THREAT EVENT / ID", 300, "w"),
            ("CONFIDENCE", 110, "center"),
            ("STATUS", 130, "center"),
            ("ACTIONS", 120, "center"),
        ]
        for text, width, align in header_specs:
            cell = tk.Frame(header_row, bg="#09101f", width=width, height=38)
            cell.pack(side="left", fill="y", padx=6, pady=0)
            cell.pack_propagate(False)
            tk.Label(
                cell,
                text=text,
                bg="#09101f",
                fg=self.MUTED,
                font=("Courier New", 8, "bold"),
                anchor=align,
                padx=10 if align == "w" else 0,
                pady=10,
            ).pack(fill="both", expand=True)

        self.history_rows_container = tk.Frame(self.history_inner, bg=self.PANEL)
        self.history_rows_container.pack(fill="both", expand=True)

        self.status_footer = tk.Label(
            right_panel,
            textvariable=self.footer_var,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Courier New", 8, "bold"),
            anchor="w",
            padx=16,
            pady=10,
        )
        self.status_footer.grid(row=2, column=0, sticky="ew")

    def make_meta_item(self, parent: tk.Widget, column: int, label: str, value: str, value_color: str) -> tk.StringVar:
        frame = tk.Frame(parent, bg=self.PANEL_2, padx=14, pady=12)
        frame.grid(row=0, column=column, sticky="nsew")
        tk.Label(frame, text=label, bg=self.PANEL_2, fg=self.MUTED, font=("Courier New", 8, "bold")).pack(anchor="w")
        text_var = tk.StringVar(value=value)
        tk.Label(frame, textvariable=text_var, bg=self.PANEL_2, fg=value_color, font=("Segoe UI", 10, "italic")).pack(anchor="e", fill="x", pady=(4, 0))
        return text_var

    def load_initial_incidents(self) -> None:
        records = self.history_repository.list_records()
        self.incidents = [self.normalize_record(record) for record in records]
        if self.incidents:
            self.selected_record_id = self.incidents[0].id
            self.update_footer_from_selected()
        else:
            self.selected_record_id = None

    def normalize_record(self, record: DetectionRecord) -> DetectionRecord:
        if record.confidence <= 1.0:
            record.confidence *= 100.0
        return record

    def sync_toggle_button_text(self) -> None:
        self.toggle_label_var.set("Stop Detection" if self.detection_active else "Start Detection")

    def on_confidence_change(self, value: str) -> None:
        self.confidence_value = int(float(value))
        self.app_config.confidence = self.confidence_value / 100.0
        self.timeline_var.set(f"{self.confidence_value}%")
        self.confidence_value_label.configure(text=self.timeline_var.get())
        self.footer_var.set(f"Confidence updated to {self.confidence_value}%")
        self.render_stage(force=True)

    def toggle_detection(self) -> None:
        if self.detection_active:
            self.detection_active = False
            self.stream_controller.stop()
            self.tracking_var.set("Detection Paused")
            self.footer_var.set("Detection is paused. Camera stream stopped.")
        else:
            if self.stream_controller.start():
                self.detection_active = True
                self.last_alert_frame_id = -1
                self.tracking_var.set("Active Tracking")
                self.footer_var.set("Detection resumed with current confidence threshold.")
            else:
                self.footer_var.set(self.stream_controller.last_error or "Could not start camera stream.")
                self.tracking_var.set("Detection Paused")
        self.sync_toggle_button_text()
        self.render_stage(force=True)

    def on_history_resize(self, event) -> None:
        self.history_canvas.itemconfigure(self.history_inner_id, width=event.width)

    def get_current_frame(self):
        return self.stream_controller.get_frame()

    def get_current_detections(self) -> list[Detection]:
        return self.stream_controller.get_detections()

    def selected_record(self) -> DetectionRecord | None:
        if self.selected_record_id is None:
            return self.incidents[0] if self.incidents else None
        for record in self.incidents:
            if record.id == self.selected_record_id:
                return record
        return self.incidents[0] if self.incidents else None

    def refresh_history(self) -> None:
        records = self.history_repository.list_records()
        self.incidents = [self.normalize_record(record) for record in records]

        if self.selected_record_id is None or all(record.id != self.selected_record_id for record in self.incidents):
            self.selected_record_id = self.incidents[0].id if self.incidents else None

        self.refresh_history_view()
        self.update_footer_from_selected()

    def update_footer_from_selected(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        self.capture_time_var.set(format_timestamp(record.created_at))
        self.operator_notes_var.set(record.notes or "No notes available.")
        self.footer_capture.set(self.capture_time_var.get())
        self.footer_notes.set(self.operator_notes_var.get())

    def select_record(self, record_id: int) -> None:
        self.selected_record_id = record_id
        self.update_footer_from_selected()
        self.refresh_history()
        self.footer_var.set(f"Selected incident #{record_id}.")

    def refresh_history_view(self) -> None:
        for child in self.history_rows_container.winfo_children():
            child.destroy()

        if not self.incidents:
            empty = tk.Label(
                self.history_rows_container,
                text="No incidents available.",
                bg=self.PANEL,
                fg=self.MUTED,
                font=("Segoe UI", 11),
                padx=20,
                pady=20,
            )
            empty.pack(fill="x", padx=8, pady=8)
            return

        for record in self.incidents:
            self.build_history_row(record, record.id == self.selected_record_id)

    def build_history_row(self, record: DetectionRecord, selected: bool) -> None:
        bg = "#0e1930" if selected else "#09101f"
        row = tk.Frame(
            self.history_rows_container,
            bg=bg,
            height=70,
            highlightthickness=1,
            highlightbackground="#ff4d7d" if selected else self.BORDER,
        )
        row.pack(fill="x", padx=4, pady=4)
        row.pack_propagate(False)
        row.bind("<Button-1>", lambda _event, rid=record.id: self.select_record(rid))

        name_col = tk.Frame(row, bg=bg, width=300)
        name_col.pack(side="left", fill="y", padx=(10, 6), pady=10)
        name_col.pack_propagate(False)
        dot_color = self.AMBER if record.status == "pending" else self.GREEN if record.status == "verified" else self.MUTED
        dot = tk.Canvas(name_col, width=10, height=10, bg=bg, highlightthickness=0, bd=0)
        dot.create_oval(2, 2, 8, 8, fill=dot_color, outline=dot_color)
        dot.pack(side="left", padx=(0, 10), pady=16)
        text_wrap = tk.Frame(name_col, bg=bg)
        text_wrap.pack(side="left", fill="both", expand=True, pady=4)
        tk.Label(
            text_wrap,
            text=record_display_title(record),
            bg=bg,
            fg=self.TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            text_wrap,
            text=f"{format_timestamp(record.created_at)} | TRK-{record.id}",
            bg=bg,
            fg=self.MUTED,
            font=("Courier New", 8),
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        conf_col = tk.Frame(row, bg=bg, width=110)
        conf_col.pack(side="left", fill="y", padx=6, pady=10)
        conf_col.pack_propagate(False)
        conf_bg = "#5d2f00" if record.confidence < 70 else "#6c142a" if record.confidence >= 85 else "#744200"
        conf_fg = "#ffd18c" if record.confidence < 85 else "#ffb6c3"
        tk.Label(
            conf_col,
            text=f"{record.confidence:.0f}%",
            bg=conf_bg,
            fg=conf_fg,
            font=("Courier New", 9, "bold"),
            padx=12,
            pady=4,
        ).pack(anchor="center", expand=True)

        status_col = tk.Frame(row, bg=bg, width=130)
        status_col.pack(side="left", fill="y", padx=6, pady=10)
        status_col.pack_propagate(False)
        status_bg = "#1a2c16" if record.status == "verified" else "#5f3508" if record.status == "pending" else "#121f39"
        status_fg = "#8df3cd" if record.status == "verified" else "#ffcf71" if record.status == "pending" else "#aeb9ce"
        status_text = record_status_meta(record.status)[1]
        tk.Label(
            status_col,
            text=status_text,
            bg=status_bg,
            fg=status_fg,
            font=("Courier New", 8, "bold"),
            padx=12,
            pady=4,
        ).pack(anchor="center", expand=True)

        action_col = tk.Frame(row, bg=bg, width=120)
        action_col.pack(side="left", fill="y", padx=6, pady=10)
        action_col.pack_propagate(False)
        button_row = tk.Frame(action_col, bg=bg)
        button_row.place(relx=0.5, rely=0.5, anchor="center")
        self.make_action_button(button_row, "👁", lambda rec=record: self.view_record_image(rec))
        self.make_action_button(button_row, "✎", lambda rec=record: self.edit_record(rec))
        self.make_action_button(button_row, "🗑", lambda rec=record: self.delete_record(rec), danger=True)

    def make_action_button(self, parent: tk.Widget, text: str, command, danger: bool = False) -> None:
        tk.Button(
            parent,
            text=text,
            command=command,
            bg="#0f172a",
            fg="#d7e4ff" if not danger else "#ffb6c3",
            activebackground="#1d2b46",
            activeforeground="white",
            relief="flat",
            width=2,
            font=("Segoe UI Symbol", 9, "bold"),
            padx=0,
            pady=0,
        ).pack(side="left", padx=1, pady=0)

    def render_stage(self, force: bool = False) -> None:
        frame, frame_id, error = self.get_current_frame()
        detections = self.get_current_detections()
        detection_signature = tuple(sorted(f"{d.category}:{d.label}:{d.confidence:.2f}" for d in detections))

        if not force and frame_id == self.last_frame_id and detection_signature == self.last_detection_signature:
            return
        self.last_frame_id = frame_id
        self.last_detection_signature = detection_signature

        width = max(780, self.stage_canvas.winfo_width())
        height = max(470, self.stage_canvas.winfo_height())
        self.stage_canvas.delete("frame")
        self.stage_canvas.delete("overlay")
        self.stage_canvas.coords(self.badge_window, 18, 18)
        self.stage_canvas.coords(self.note_window, width - 18, 18)

        self.stage_canvas.configure(scrollregion=(0, 0, width, height))
        self.stage_canvas.create_rectangle(0, 0, width, height, fill="#040812", outline="", tags=("frame",))
        for x in range(0, width, 72):
            self.stage_canvas.create_line(x, 0, x, height, fill="#11203b", width=1, tags=("frame",))
        for y in range(0, height, 72):
            self.stage_canvas.create_line(0, y, width, y, fill="#11203b", width=1, tags=("frame",))

        if frame is not None:
            if self.detection_active and detections:
                draw_detection_boxes(self.cv2, frame, detections)
            photo = frame_to_photoimage(self.cv2, frame, width - 30, height - 40, self.root)
            if photo is not None:
                self.photo_image = photo
                self.stage_canvas.create_image(width // 2, height // 2, image=self.photo_image, tags=("frame",))
        elif error:
            self.stage_canvas.create_text(
                width // 2,
                height // 2,
                text=error,
                fill="#ffd5df",
                font=("Segoe UI", 12, "bold"),
                tags=("overlay",),
            )
        else:
            self.stage_canvas.create_text(
                width // 2,
                height // 2,
                text="Camera paused",
                fill="#d4d9e4",
                font=("Segoe UI", 14, "bold"),
                tags=("overlay",),
            )

        self.stage_canvas.create_rectangle(
            18,
            18,
            348,
            88,
            fill="#5b1125" if self.detection_active else "#2d3344",
            outline="#ff4d7d" if self.detection_active else "#566583",
            width=1,
            tags=("overlay",),
        )
        self.stage_canvas.create_text(
            30,
            30,
            anchor="nw",
            fill="#ff6f91" if self.detection_active else "#aeb9ce",
            text="PIPELINE PROCESSING" if self.detection_active else "DETECTION PAUSED",
            font=("Courier New", 10, "bold"),
            tags=("overlay",),
        )
        self.stage_canvas.create_text(
            30,
            52,
            anchor="nw",
            fill="#ffd5df" if self.detection_active else "#d4d9e4",
            text=self.pipeline_desc_var.get(),
            font=("Courier New", 8),
            tags=("overlay",),
        )
        self.stage_canvas.create_text(
            width - 18,
            20,
            anchor="ne",
            fill=self.CYAN,
            text="ACTIVE TRACKING" if self.detection_active else "MONITOR HOLD",
            font=("Courier New", 8, "bold"),
            tags=("overlay",),
        )

        progress = max(0, min(100, self.confidence_value))
        self.timeline_track.configure(width=width - 32)
        self.timeline_track.coords(self.timeline_fill, 0, 0, int((width - 32) * (progress / 100)), 4)
        self.timeline_track.itemconfigure(self.timeline_fill, fill=self.PINK if self.detection_active else "#566583")
        self.timeline_var.set(f"{progress}%")
        self.confidence_value_label.configure(text=self.timeline_var.get())

        if detections:
            labels = ", ".join(sorted({det.label for det in detections}))[:48]
            self.pipeline_desc_var.set(f"{len(detections)} active detection(s): {labels}")
            self.status_badge.configure(
                text=f"SUSPICIOUS EVENT\n{self.pipeline_desc_var.get()}",
                bg="#6e1a35",
                fg="#ffd9e3",
            )
        else:
            self.pipeline_desc_var.set("Evaluating multi-spectral raw frames...")
            self.status_badge.configure(
                text=f"{'PIPELINE PROCESSING' if self.detection_active else 'DETECTION PAUSED'}\n{self.pipeline_desc_var.get()}",
                bg="#5b1125" if self.detection_active else "#2d3344",
                fg="#ffd5df" if self.detection_active else "#d4d9e4",
            )

        self.sync_toggle_button_text()

    def build_record_from_event(
        self,
        detections: list[Detection],
        image_path: str,
        record_id: int | None = None,
        location: str = "",
    ) -> DetectionRecord:
        categories = sorted({det.category for det in detections}) or ["unknown"]
        labels = [det.label for det in detections] or ["Unknown"]
        confidence = max((det.confidence for det in detections), default=self.app_config.confidence)
        if confidence <= 1.0:
            confidence *= 100.0
        status = "verified" if confidence >= 85 else "pending" if confidence >= 60 else "false_positive"
        inferred_location = location or (
            "East Corridor"
            if any(det.category == "fight" for det in detections)
            else "Parking Lot B"
            if any(det.category == "weapon" for det in detections)
            else "Live Camera Feed"
        )
        return DetectionRecord(
            id=record_id or int(time.time() * 1000) % 100000,
            created_at=datetime.now(),
            categories=categories,
            labels=labels,
            confidence=confidence,
            image_path=image_path,
            status=status,
            notes=", ".join(labels),
            location=inferred_location,
            image_data=Path(image_path).read_bytes() if image_path and Path(image_path).exists() else None,
        )

    def append_record(self, record: DetectionRecord) -> None:
        self.incidents = [record] + [item for item in self.incidents if item.id != record.id]
        self.selected_record_id = record.id
        self.refresh_history()
        self.update_footer_from_selected()

    def save_alert_frame(self, frame) -> Path:
        if frame is None:
            frame = make_alert_placeholder(self.cv2)
        return save_frame(self.cv2, frame, self.output_dir)

    def add_alert(self, detections: list[Detection], frame, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_alert_time < self.app_config.alert_cooldown:
            return

        inferred_location = (
            "East Corridor"
            if any(det.category == "fight" for det in detections)
            else "Parking Lot B"
            if any(det.category == "weapon" for det in detections)
            else "Live Camera Feed"
        )
        path = self.save_alert_frame(frame)
        stored_id = self.history_repository.create_alert(path, detections)
        if stored_id is not None:
            record = self.history_repository.get_record(stored_id)
            if record is None:
                record = self.build_record_from_event(detections, str(path), stored_id, inferred_location)
            else:
                record = self.normalize_record(record)
                record.location = record.location or inferred_location
        else:
            record = self.build_record_from_event(detections, str(path), location=inferred_location)

        self.append_record(record)
        self.footer_var.set(f"Alert stored as {path.name}")
        self.last_alert_time = now
        play_notification_sound(self.app_config.sound_enabled)

    def view_record_image(self, record: DetectionRecord) -> None:
        image = None
        if record.image_data:
            import numpy as np

            buffer = np.frombuffer(record.image_data, dtype=np.uint8)
            image = self.cv2.imdecode(buffer, self.cv2.IMREAD_COLOR)
        elif record.image_path and Path(record.image_path).exists():
            image = self.cv2.imread(record.image_path)
        if image is None:
            image = make_alert_placeholder(self.cv2)

        top = tk.Toplevel(self.root)
        top.title(f"Detection Image #{record.id}")
        top.configure(bg=self.BG)
        photo = frame_to_photoimage(self.cv2, image, 1100, 760, top)
        if photo is None:
            tk.Label(top, text="Could not render image.", bg=self.BG, fg=self.TEXT).pack(padx=20, pady=20)
            return
        label = tk.Label(top, image=photo, bg=self.BG)
        label.image = photo
        label.pack(padx=12, pady=12)

    def edit_record(self, record: DetectionRecord) -> None:
        new_status = simpledialog.askstring("Edit Incident", "Status:", initialvalue=record.status, parent=self.root)
        if new_status is None:
            return
        new_notes = simpledialog.askstring("Edit Incident", "Notes:", initialvalue=record.notes, parent=self.root)
        if new_notes is None:
            return

        record.status = new_status.strip() or record.status
        record.notes = new_notes.strip() or record.notes
        self.history_repository.update_record(record.id, record.status, record.notes)
        records = self.history_repository.list_records()
        self.incidents = [self.normalize_record(item) for item in records]
        self.refresh_history()
        self.footer_var.set(f"Updated incident #{record.id}.")

    def delete_record(self, record: DetectionRecord) -> None:
        if not messagebox.askyesno("Delete Incident", f"Delete incident #{record.id}?", parent=self.root):
            return
        self.history_repository.delete_record(record.id)
        records = self.history_repository.list_records()
        self.incidents = [self.normalize_record(item) for item in records]
        self.selected_record_id = self.incidents[0].id if self.incidents else None
        self.refresh_history()
        self.update_footer_from_selected()
        self.footer_var.set(f"Deleted incident #{record.id}.")

    def poll(self) -> None:
        if self.closed:
            return

        frame, frame_id, error = self.get_current_frame()
        detections = self.get_current_detections()
        if self.detection_active and detections and frame is not None and frame_id != self.last_alert_frame_id:
            self.add_alert(detections, frame)
            self.last_alert_frame_id = frame_id
        elif error:
            self.footer_var.set(error)

        self.render_stage()
        self.root.after(60, self.poll)

    def close(self) -> None:
        self.closed = True
        self.stream_controller.close()
        self.history_repository.close()
        self.root.after(30, self.root.destroy)

    def run(self) -> int:
        self.root.mainloop()
        return 0


def view_stream(base_url: str, output_dir: Path, skip_check: bool, app_config: AppConfig) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print(
            "Missing dependency: opencv-python. Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    detectors = load_detectors(app_config)
    history_repository = DetectionHistoryRepository(output_dir)

    if not skip_check:
        check_connection(base_url)

    print("Connected.")
    print(
        "Speed settings: "
        f"imgsz={app_config.imgsz}, process_every={app_config.process_every}, "
        f"stream={app_config.stream_width}x{app_config.stream_height}"
    )
    if detectors:
        for model_config, _detector in detectors:
            print(f"{model_config.name.title()} detection is enabled for: {', '.join(sorted(model_config.labels))}")
    else:
        print("No detection model is enabled. The dashboard will still show camera status and simulated incidents.")

    controller = StreamController(
        cv2=cv2,
        base_url=base_url,
        app_config=app_config,
        detectors=detectors,
    )
    if not controller.start():
        print(controller.last_error or f"Could not open video stream: {base_url}/video", file=sys.stderr)
        history_repository.close()
        return 1

    app = DashboardApp(
        cv2=cv2,
        app_config=app_config,
        output_dir=output_dir,
        history_repository=history_repository,
        stream_controller=controller,
    )

    try:
        return app.run()
    finally:
        controller.close()
        history_repository.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to an Android IP Webcam app stream over Wi-Fi.")
    parser.add_argument(
        "address",
        help="Phone camera address, for example 192.168.1.23:8080 or http://192.168.1.23:8080",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("image_data"),
        help="Folder where detection screenshots and history metadata are saved.",
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
        default=0.50,
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
        default=Path("image_data"),
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
        help="Legacy option kept for compatibility; detection history now uses the local image_data folder.",
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
